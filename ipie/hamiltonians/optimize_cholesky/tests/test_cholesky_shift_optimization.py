"""Step 11: outer optimization over the gauge and the Cholesky shift.

ExponentialGaugeObjective(optimize_cholesky_shift='tied' or 'split') works on
x = (theta, mu/s): the gauge angles, then the shift divided by a fixed scale
s (cholesky_shift_scale, the eigenvalue spread of each rotated factor). Tied
shifts have r entries, split ones 2r (r for aa_only). frozen_theta keeps the
gauge fixed. optimize_gauge adds the start ('median', 'zero' or an array) and
the modes: theta only (with the evaluator's or a fixed shift), theta and mu,
or mu only. The tests check:
  * the coordinate layouts, the start and the scale;
  * the coordinate gradient against central differences, and its pieces;
  * the cache, the stopping norms, and that theta-only runs are unchanged;
  * every optimize_gauge mode and its result fields, Fock-space exactness of
    the final decomposition, and validation;
  * near_zero_eigenvalues on shifted decompositions;
  * the evaluator copies used by the objective.
"""

import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
from ipie.hamiltonians.optimize_cholesky.optimizers import (
    ExponentialGaugeObjective,
    LBFGSOptimizer,
    cholesky_shift_scale,
    factor_eigenvalues,
    median_cholesky_shift,
    optimize_gauge,
    resolve_cholesky_shift,
)
from ipie.hamiltonians.optimize_cholesky.problem import (
    build_gauge_state,
    exponential_gauge,
    near_zero_eigenvalues,
    prepare_problem,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import check_norm, example_tensors, physical2MB
from ipie.hamiltonians.optimize_cholesky.tests.test_cholesky_shift import (
    MAPS,
    SECTORS,
    TAU1,
    TAU2,
    build,
    fixture,
    green_function,
    sector,
    shift_of,
)
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import assert_close

INNER = InnerOptions(optimize_shift=True, max_sweeps=20000, gradient_atol=1e-13, gradient_rtol=1e-12,
                     shift_rtol=1e-13)
THETA = np.array([.3])
LAYOUTS = (("tied", False), ("split", False), ("tied", True), ("split", True))


def objective(problem, mode=False, frozen=False, partition="anchored_log", **kwargs):
    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
    return ExponentialGaugeObjective(evaluator, inner_options=INNER, optimize_cholesky_shift=mode,
                                     frozen_theta=THETA if frozen else None, **kwargs)


def exact_on_sector(case, problem, decomposition):
    basis, basis_map = sector(problem)
    H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
    check_norm(green_function(decomposition, basis, basis_map, problem.data.nbasis),
               decomposition.lambda_electronic*np.eye(len(basis)) - H)


class LayoutTests(unittest.TestCase):
    def test_pack_and_unpack(self):
        for nelec in SECTORS:
            problem, _ = fixture(nelec)
            r = problem.data.nchol
            aa = problem.policy == "aa_only"
            for mode, frozen in LAYOUTS:
                with self.subTest(nelec=nelec, mode=mode, frozen=frozen):
                    f = objective(problem, mode, frozen)
                    nshift = r if (mode == "tied" or aa) else 2*r
                    self.assertEqual((f.ntheta, f.nshift, f.nparams),
                                     (0 if frozen else 1, nshift, (0 if frozen else 1) + nshift))
                    mu = np.asarray(shift_of(mode, problem))
                    x = f.pack(THETA, mu)
                    theta, unpacked = f.unpack(x)
                    np.testing.assert_array_equal(theta, THETA)
                    expected = resolve_cholesky_shift(problem, mu)
                    assert_close(unpacked, expected)
                    self.assertTrue(np.all(unpacked[:, 1] == 0) if aa else True)
                    assert_close(x[f.ntheta:f.ntheta + r], expected[:, 0]/f.cholesky_scale)
                    if mode == "tied" and not aa:
                        with self.assertRaisesRegex(ValueError, "tied layout"):
                            f.pack(THETA, shift_of("split", problem))

    def test_theta_only_objective_is_today_s(self):
        problem, _ = fixture((2, 1))
        f = objective(problem)
        self.assertEqual((f.nparams, f.ntheta, f.nshift), (1, 1, 0))
        self.assertIsNone(f.cholesky_scale)
        point = f.evaluate(THETA)
        self.assertIs(point.coordinates, point.theta)
        self.assertIs(point.gradient, point.gradient_theta)
        self.assertIsNone(point.cholesky_shift)
        self.assertEqual(point.tangent_gradient_norm,
                         float(np.linalg.norm(point.gauge_gradient.tangent_gradient, "fro")))
        np.testing.assert_array_equal(f.pack(THETA), THETA)
        with self.assertRaisesRegex(ValueError, "does not optimize"):
            f.pack(THETA, np.ones(2))
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        for bad in (dict(cholesky_scale=np.ones(2)), dict(frozen_theta=THETA)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "need optimize_cholesky_shift"):
                ExponentialGaugeObjective(evaluator, **bad)

    def test_median_and_scale(self):
        for n in (3, 4):
            h, eri, chol = example_tensors(n=n, r=2)
            problem = prepare_problem(h, eri, chol, nelec=(1, 1))
            O = exponential_gauge(THETA, 2)
            ell = factor_eigenvalues(problem, O)
            np.testing.assert_array_equal(ell, np.linalg.eigh(build_gauge_state(problem.data, O).chol)[0])
            median = median_cholesky_shift(problem, O)
            with self.subTest(n=n):
                if n % 2 == 0:
                    np.testing.assert_array_equal(median, np.median(ell, axis=1))
                else:
                    for g in range(2):
                        gaps = np.diff(ell[g])
                        wider = 1 if gaps[1] > gaps[0] else 0
                        self.assertEqual(median[g], .5*(ell[g, wider] + ell[g, wider + 1]))
                        self.assertGreater(np.min(np.abs(ell[g] - median[g])), 0.)
                np.testing.assert_array_equal(cholesky_shift_scale(problem, O), ell[:, -1] - ell[:, 0])
        flat = prepare_problem(np.eye(2), np.zeros((2,)*4), np.zeros((1, 2, 2)), nelec=(1, 1))
        np.testing.assert_array_equal(cholesky_shift_scale(flat), [1.])

    def test_resolve_cholesky_shift(self):
        problem, _ = fixture((2, 1))
        triplet, _ = fixture((2, 0))
        r = problem.data.nchol
        np.testing.assert_array_equal(resolve_cholesky_shift(problem, "zero"), np.zeros((r, 2)))
        median = median_cholesky_shift(problem, exponential_gauge(THETA, r))
        np.testing.assert_array_equal(resolve_cholesky_shift(problem, "median", exponential_gauge(THETA, r)),
                                      np.stack([median, median], 1))
        np.testing.assert_array_equal(resolve_cholesky_shift(triplet, np.ones(r)),
                                      np.stack([np.ones(r), np.zeros(r)], 1))
        mu = shift_of("split", problem)
        np.testing.assert_array_equal(resolve_cholesky_shift(problem, mu), mu)
        with self.assertRaisesRegex(ValueError, "'median', 'zero'"):
            resolve_cholesky_shift(problem, "mean")
        with self.assertRaisesRegex(ValueError, "down-spin"):
            resolve_cholesky_shift(triplet, np.ones((r, 2)))


class GradientTests(unittest.TestCase):
    def test_coordinate_gradient_matches_central_differences(self):
        rng = np.random.default_rng(21)
        for nelec in SECTORS:
            problem, _ = fixture(nelec)
            for partition in MAPS:
                for mode, frozen in LAYOUTS:
                    with self.subTest(nelec=nelec, partition=partition, mode=mode, frozen=frozen):
                        f = objective(problem, mode, frozen, partition)
                        x = f.pack(THETA, shift_of(mode, problem))
                        point = f.evaluate(x)
                        E = rng.normal(size=f.nparams)
                        analytic = float(np.dot(point.gradient, E))
                        scale = float(np.linalg.norm(point.gradient)*np.linalg.norm(E))
                        errors = []
                        for h in (1e-4, 1e-5, 1e-6):
                            finite = (f.evaluate(x + h*E).loss - f.evaluate(x - h*E).loss)/(2*h)
                            errors.append(abs(finite - analytic)/max(abs(finite), abs(analytic), scale))
                        self.assertLess(min(errors), 1e-6)

    def test_gradient_pieces(self):
        """theta part: exponential_gauge_vjp of gradient_O; shift part: s times the evaluator's cholesky_gradient."""
        for nelec in ((2, 1), (2, 0)):
            problem, _ = fixture(nelec)
            r = problem.data.nchol
            for mode in ("tied", "split"):
                with self.subTest(nelec=nelec, mode=mode):
                    f = objective(problem, mode)
                    mu = resolve_cholesky_shift(problem, shift_of(mode, problem))
                    point = f.evaluate(f.pack(THETA, mu))
                    tied = mode == "tied" or problem.policy == "aa_only"
                    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition="anchored_log",
                                               cholesky_gradient="tied" if tied else "split")
                    direct = evaluator.relaxed(exponential_gauge(THETA, r), options=INNER, cholesky_shift=mu)
                    g = np.asarray(direct.cholesky_gradient)
                    expected = g*f.cholesky_scale if tied else np.concatenate([g[:, 0], g[:, 1]])*np.tile(
                        f.cholesky_scale, 2)
                    assert_close(point.gradient_scaled_shift, expected)
                    assert_close(point.gauge_gradient.gradient_O, direct.gradient_O)
                    assert_close(point.cholesky_shift, direct.decomposition.cholesky_shift)   # via mu/s

    def test_cache_is_keyed_on_coordinates(self):
        problem, _ = fixture((2, 1))
        f = objective(problem, "split")
        x = f.pack(THETA, shift_of("split", problem))
        first = f.evaluate(x)
        self.assertIs(f.evaluate(x.copy()), first)
        self.assertEqual((f.evaluations, f.cache_hits), (1, 1))
        moved = x.copy()
        moved[-1] += 1e-3
        self.assertIsNot(f.evaluate(moved), first)
        self.assertEqual(f.evaluations, 2)

    def test_stopping_norms_include_the_shift(self):
        problem, _ = fixture((2, 1))
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                f = objective(problem, "split", frozen)
                point = f.evaluate(f.pack(THETA, shift_of("split", problem)))
                tangent = float(np.linalg.norm(point.gauge_gradient.tangent_gradient, "fro"))
                shift = float(np.linalg.norm(point.gradient_scaled_shift))
                if frozen:
                    np.testing.assert_array_equal(point.coordinates, point.scaled_shift)
                    self.assertEqual(point.gradient_norm, float(np.max(np.abs(point.gradient_scaled_shift))))
                    self.assertEqual(point.tangent_gradient_norm, shift)
                else:
                    np.testing.assert_array_equal(point.gradient,
                                                  np.concatenate([point.gradient_theta, point.gradient_scaled_shift]))
                    self.assertEqual(point.gradient_norm, float(np.max(np.abs(point.gradient))))
                    self.assertEqual(point.tangent_gradient_norm, float(np.hypot(tangent, shift)))


class OptimizeGaugeTests(unittest.TestCase):
    OPTIONS = LBFGSOptions(max_iterations=12, line_search="weak_wolfe")

    def run_mode(self, problem, **kwargs):
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition="anchored_log")
        return evaluator, optimize_gauge(evaluator, theta0=THETA, options=self.OPTIONS, inner_options=INNER,
                                         **kwargs)

    def test_theta_only_run_is_unchanged(self):
        problem, _ = fixture((2, 1))
        evaluator, result = self.run_mode(problem)
        reference = LBFGSOptimizer(ExponentialGaugeObjective(evaluator, inner_options=INNER),
                                   options=self.OPTIONS).optimize(THETA)
        self.assertEqual(result.loss, reference.loss)
        np.testing.assert_array_equal(result.theta, reference.theta)
        self.assertEqual([row.loss for row in result.history], [row.loss for row in reference.history])
        self.assertIsNone(result.cholesky_shift)
        self.assertIsNone(result.initial_cholesky_shift)
        self.assertIsNone(result.cholesky_scale)

    def test_modes(self):
        for nelec in ((2, 1), (2, 0)):
            problem, _ = fixture(nelec)
            r = problem.data.nchol
            start = exponential_gauge(THETA, r)
            median = resolve_cholesky_shift(problem, "median", start)
            with self.subTest(nelec=nelec, mode="fixed median"):
                _, result = self.run_mode(problem, cholesky_shift0="median")
                np.testing.assert_array_equal(result.cholesky_shift, median)
                np.testing.assert_array_equal(result.initial_cholesky_shift, median)
                self.assertIsNone(result.cholesky_scale)
                self.assertLess(result.loss, result.initial_loss)
            for mode in ("tied", "split"):
                with self.subTest(nelec=nelec, mode=mode):
                    _, result = self.run_mode(problem, optimize_cholesky_shift=mode)
                    np.testing.assert_array_equal(result.initial_cholesky_shift, median)
                    np.testing.assert_array_equal(result.cholesky_scale, cholesky_shift_scale(problem, start))
                    self.assertLess(result.loss, result.initial_loss)
                    self.assertFalse(np.array_equal(result.cholesky_shift, median))
                    self.assertFalse(np.array_equal(result.theta, THETA))
                    if problem.policy == "aa_only":
                        np.testing.assert_array_equal(result.cholesky_shift[:, 1], 0.)
                    exact_on_sector(self, problem, result.decomposition)
            with self.subTest(nelec=nelec, mode="mu only"):
                _, result = self.run_mode(problem, optimize_cholesky_shift="tied", optimize_theta=False,
                                          cholesky_shift0="zero")
                np.testing.assert_array_equal(result.theta, THETA)
                np.testing.assert_array_equal(result.initial_cholesky_shift, np.zeros((r, 2)))
                self.assertLess(result.loss, result.initial_loss)

    def test_armijo_runs_with_a_shift(self):
        problem, _ = fixture((2, 1))
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        result = optimize_gauge(evaluator, theta0=THETA, options=LBFGSOptions(max_iterations=5),
                                inner_options=INNER, optimize_cholesky_shift="tied")
        self.assertLess(result.loss, result.initial_loss)
        self.assertEqual(result.history[-1].iteration, result.iterations)

    def test_validation(self):
        problem, _ = fixture((2, 1))
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        r = problem.data.nchol
        cases = (
            (dict(optimize_theta=False), "Nothing to optimize"),
            (dict(optimize_cholesky_shift="both"), "optimize_cholesky_shift"),
            (dict(optimize_cholesky_shift="tied", theta0=np.zeros(2)), "theta0"),
            (dict(optimize_cholesky_shift="tied", cholesky_shift0=shift_of("split", problem)), "tied layout"),
            (dict(optimize_cholesky_shift="split", cholesky_shift0="mean"), "'median', 'zero'"),
            (dict(optimize_cholesky_shift="split", cholesky_shift0=np.zeros(r + 1)), "cholesky_shift"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                optimize_gauge(evaluator, options=LBFGSOptions(max_iterations=1), **kwargs)
        h, eri, chol = example_tensors(n=3, r=2)
        fock = prepare_problem(h, eri, chol)
        with self.assertRaisesRegex(ValueError, "particle numbers"):
            optimize_gauge(GaugeEvaluator(fock, tau1=TAU1, tau2=TAU2), optimize_cholesky_shift="tied")
        with self.assertRaisesRegex(ValueError, "positive"):
            objective(problem, "tied", cholesky_scale=np.array([1., 0.]))


class NearZeroTests(unittest.TestCase):
    def test_shifted_kinks_are_reported_per_spin(self):
        for nelec in ((2, 1), (2, 0)):
            problem, gauge = fixture(nelec)
            ell = np.asarray(gauge.eigenvalues)
            r = problem.data.nchol
            mu = np.zeros((r, 2))
            mu[1, 0] = ell[1, 2]
            if nelec[1]:
                mu[0, 1] = ell[0, 0]
            found = near_zero_eigenvalues(build(problem, gauge, mu, "anchored_log"), rtol=1e-12)
            expected = {(1, 2, 0)} | ({(0, 0, 1)} if nelec[1] else set())
            with self.subTest(nelec=nelec):
                self.assertEqual({(g, a, s) for g, a, _, s in found}, expected)
                self.assertTrue(all(abs(value) < 1e-15 for _, _, value, _ in found))
                plain = near_zero_eigenvalues(build(problem, gauge, None, "anchored_log"))
                self.assertEqual(plain, near_zero_eigenvalues(gauge))
                self.assertTrue(all(len(entry) == 3 for entry in plain))


class EvaluatorCopyTests(unittest.TestCase):
    def test_copies_do_not_change_the_evaluator(self):
        problem, gauge = fixture((2, 1))
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        mu = shift_of("split", problem)
        shifted = evaluator.with_cholesky_shift(mu)
        split = evaluator.with_cholesky_gradient("split")
        self.assertIsNone(evaluator.cholesky_shift)
        self.assertIsNone(evaluator.cholesky_gradient)
        self.assertIs(shifted.problem, evaluator.problem)
        fresh = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=mu)
        for a, b in ((shifted.relaxed(gauge.O, options=INNER), fresh.relaxed(gauge.O, options=INNER)),):
            assert_close(a.gradient_O, b.gradient_O)
            assert_close(a.cholesky_gradient, b.cholesky_gradient)
        fresh = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_gradient="split")
        a, b = split.fixed(gauge.O, t=.1), fresh.fixed(gauge.O, t=.1)
        assert_close(a.cholesky_gradient, b.cholesky_gradient)
        self.assertTrue(a.decomposition.one_body.split)
        self.assertIsNone(evaluator.fixed(gauge.O, t=.1).cholesky_gradient)
        with self.assertRaisesRegex(ValueError, "cholesky_gradient"):
            evaluator.with_cholesky_gradient("both")


if __name__ == "__main__":
    unittest.main()
