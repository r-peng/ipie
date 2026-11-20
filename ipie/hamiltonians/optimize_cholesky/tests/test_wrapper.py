"""Step 5: QCSOR.optimize_from_cholesky options, defaults and limits."""

from contextlib import ExitStack
from unittest.mock import patch
import os
import tempfile
import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions, OptimizerOptions
from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, prepare_problem
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import molecule, synthetic
from ipie.hamiltonians.sor_base import QCSOR

try:
    import h5py  # noqa: F401
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

OPTIMIZERS = "ipie.hamiltonians.optimize_cholesky.optimizers.optimize_gauge"
SAVE = "ipie.hamiltonians.optimize_cholesky.io.save_decomposition"


def tensors():
    h1, eri, chol = synthetic(3, 2, 23)
    return h1, eri, chol, (2, 1)


def captured(**kwargs):
    """Run the wrapper with optimize_gauge and save_decomposition replaced; return their calls."""
    with ExitStack() as stack:
        optimize = stack.enter_context(patch(OPTIMIZERS, return_value="result"))
        save = stack.enter_context(patch(SAVE))
        result = QCSOR.optimize_from_cholesky(*tensors(), .03, .04, "unused.h5", **kwargs)
    assert result == "result"
    return optimize.call_args, save.call_args


class WrapperOptionTests(unittest.TestCase):
    def test_defaults(self):
        call, save = captured()
        inner, outer = call.kwargs["inner_options"], call.kwargs["options"]
        self.assertEqual(inner, InnerOptions(optimize_shift=True))
        self.assertEqual((inner.max_sweeps, inner.gradient_atol, inner.gradient_rtol), (500, 1e-11, 1e-9))
        self.assertEqual((inner.amplitude_limit, inner.one_body_limit), (None, None))
        self.assertEqual(outer, LBFGSOptions(verbosity=2, line_search="weak_wolfe"))
        self.assertEqual((outer.max_iterations, outer.gradient_atol, outer.gradient_rtol), (100, 1e-8, 1e-6))
        self.assertEqual(call.kwargs["solver"], "lbfgs")
        self.assertEqual(save.kwargs, {"drop_identity": False, "overwrite": False, "metadata": None})

    def test_every_option_reaches_its_field(self):
        call, save = captured(maxiter=7, verbosity=0, gradient_atol=1e-9, gradient_rtol=0.0,
                              maxsweep=50, inner_atol=1e-12, inner_rtol=0.0,
                              amplitude_limit=.8, one_body_limit=.3,
                              drop_identity=True, overwrite=True, metadata={"tag": 1})
        inner, outer = call.kwargs["inner_options"], call.kwargs["options"]
        self.assertEqual(inner, InnerOptions(max_sweeps=50, gradient_atol=1e-12, gradient_rtol=0.0,
                                             amplitude_limit=.8, one_body_limit=.3, optimize_shift=True))
        self.assertEqual(outer, LBFGSOptions(max_iterations=7, verbosity=0, gradient_atol=1e-9,
                                             gradient_rtol=0.0, line_search="weak_wolfe"))
        self.assertEqual(save.kwargs, {"drop_identity": True, "overwrite": True, "metadata": {"tag": 1}})

    def test_option_objects_are_the_base_and_keywords_override(self):
        call, _ = captured(inner_options=InnerOptions(max_sweeps=40, optimize_shift=False, max_abs_t=5.),
                           inner_atol=1e-12,
                           options=LBFGSOptions(memory=4, line_search="armijo", verbosity=1), maxiter=3)
        inner, outer = call.kwargs["inner_options"], call.kwargs["options"]
        self.assertEqual(inner, InnerOptions(max_sweeps=40, optimize_shift=False, max_abs_t=5., gradient_atol=1e-12))
        self.assertEqual(outer, LBFGSOptions(memory=4, line_search="armijo", verbosity=1, max_iterations=3))

    def test_unknown_names_and_wrong_types_are_rejected(self):
        for kwargs in ({"amplitude_limt": .5}, {"maxsweeps": 10}, {"tol": 1e-8}):
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                captured(**kwargs)
        with self.assertRaises(TypeError):
            QCSOR.optimize_from_cholesky(*tensors(), .03, .04, "unused.h5", .1)    # theta is keyword-only
        with self.assertRaisesRegex(TypeError, "LBFGSOptions"):
            captured(options=OptimizerOptions())
        with self.assertRaisesRegex(TypeError, "InnerOptions"):
            captured(inner_options={"max_sweeps": 3})

    def test_invalid_values_fail_before_numerical_work(self):
        for kwargs in ({"amplitude_limit": 1.5}, {"one_body_limit": 0.}, {"maxsweep": -1},
                       {"inner_atol": -1.}, {"gradient_rtol": float("nan")}, {"maxiter": 2.5}):
            with self.subTest(kwargs=kwargs), \
                 patch("ipie.hamiltonians.optimize_cholesky.problem.prepare_problem",
                       side_effect=AssertionError("prepare_problem ran")), \
                 self.assertRaises(ValueError):
                captured(**kwargs)

    def test_seeded_and_global_random_starts(self):
        count = 1                                                   # r = 2 factors
        first, _ = captured(seed=3, theta=.2)
        second, _ = captured(seed=3, theta=.2)
        expected = .2*(np.random.RandomState(3).rand(count)*2-1)
        np.testing.assert_array_equal(first.kwargs["theta0"], expected)
        np.testing.assert_array_equal(second.kwargs["theta0"], expected)
        np.random.seed(5)
        unseeded, _ = captured()
        np.testing.assert_array_equal(unseeded.kwargs["theta0"], .1*(np.random.RandomState(5).rand(count)*2-1))
        zero, _ = captured(theta=0.)
        np.testing.assert_array_equal(zero.kwargs["theta0"], np.zeros(count))


@unittest.skipUnless(HAVE_H5PY, "h5py is required to write and read decomposition files")
class WrapperLimitTests(unittest.TestCase):
    def test_limits_bind_and_the_file_reads_back(self):
        m = molecule("h4_sto3g")
        problem = prepare_problem(m["h1"], m["eri"], m["chol"], nelec=m["nelec"])
        r = problem.data.nchol
        from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
        evaluator = GaugeEvaluator(problem, tau1=.03, tau2=.04)
        start = .1*(np.random.RandomState(1).rand(r*(r-1)//2)*2-1)
        ell = evaluator.decomposition(exponential_gauge(start, r)).two_body.gauge.eigenvalues
        tightest = max(np.sqrt(abs(ell[s.bix, s.modes[0]]*ell[s.bix, s.modes[1]])*.04) for s in evaluator.sources)
        limit, one_body_limit = 1.02*tightest, .01
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "decomposition.h5")
            result = QCSOR.optimize_from_cholesky(m["h1"], m["eri"], m["chol"], m["nelec"], .03, .04, path,
                                                  seed=1, maxiter=2, verbosity=0, amplitude_limit=limit,
                                                  one_body_limit=one_body_limit)
            ham = QCSOR.from_file(path, thresh=None)
        inner = result.partition_result
        self.assertEqual(inner.amplitude_limit, limit)
        self.assertEqual(inner.one_body_limit, one_body_limit)
        self.assertTrue(inner.converged)
        self.assertLessEqual(inner.projected_gradient_norm, inner.gradient_threshold)
        amplitudes = np.abs(result.decomposition.two_body.amplitudes)
        self.assertLessEqual(amplitudes.max(), limit*(1+1e-12))
        self.assertGreater(amplitudes.max(), limit*(1-1e-6))                  # the amplitude wall binds
        d1 = result.decomposition.one_body.table.d
        self.assertGreaterEqual(d1.min(), -one_body_limit*(1+1e-12))
        self.assertLess(d1.min(), -one_body_limit*(1-1e-9))                   # the one-body wall binds
        self.assertEqual(ham.nterms, result.decomposition.nterms)


if __name__ == "__main__":
    unittest.main()
