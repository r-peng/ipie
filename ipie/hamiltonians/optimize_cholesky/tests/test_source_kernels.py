"""Step 4: batched per-source kernels, early regularity checks, one expm and one mode space per gauge.

The batched amplitude map, partition intervals, amplitude/bound pullbacks and
eigh pullback are compared with the step-3 loops in source_reference.py:
values, and the source or factor that an error names.
"""

from unittest.mock import patch
import unittest

import numpy as np
from scipy.linalg import expm as scipy_expm, expm_frechet

from ipie.hamiltonians.optimize_cholesky import gradients
from ipie.hamiltonians.optimize_cholesky.channels import fixed_partition_amplitudes, source_table
from ipie.hamiltonians.optimize_cholesky.gradients import (
    GaugeEvaluator,
    _amplitude_pullback,
    _bound_pullback,
    _check_eigengaps,
    _eigh_pullback,
    check_gauge_regularity,
    exponential_gauge_vjp,
)
from ipie.hamiltonians.optimize_cholesky.loss import ModeSpace
from ipie.hamiltonians.optimize_cholesky.models import (
    DensitySource,
    GaugeFeasibilityError,
    GaugeRegularityError,
    GradientOptions,
    InnerOptions,
    ValidationError,
)
from ipie.hamiltonians.optimize_cholesky.optimizers import ExponentialGaugeObjective
from ipie.hamiltonians.optimize_cholesky.partitions import _partition_intervals
from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, prepare_problem, skew_from_parameters
from ipie.hamiltonians.optimize_cholesky.tests import source_reference as ref
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import molecule, synthetic, theta
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import assert_close

TAU1, TAU2 = 0.03, 0.04
POLICIES = {"hybrid": (2, 1), "ab_only": (1, 1), "aa_only": (2, 0)}
MAPS = ("balanced_log", "anchored_log")


def problem_of(nelec, n=4, r=3, seed=5):
    h1, eri, chol = synthetic(n, r, seed)
    return prepare_problem(h1, eri, chol, nelec=nelec)


def small_problem(chol, nelec=(1, 1)):
    chol = np.asarray(chol, dtype=float)
    h = np.array([[.2, .1], [.1, -.1]])
    return prepare_problem(h, np.einsum("gpr,gqs->prqs", chol, chol), chol, nelec=nelec)


ZERO_EIGENVALUE = [np.diag([0., .7]), [[.3, .2], [.2, -.4]]]       # factor 0 has ell = 0
DEGENERATE = [np.diag([.5, .5]), [[.3, .2], [.2, -.4]]]            # factor 0 has ell = (.5, .5)


def random_gauge(r, seed):
    return exponential_gauge(theta(r, .3, seed), r)


def gauge_of(ev, O):
    """The evaluator's GaugeState at O, through the public decomposition builder."""
    return ev.decomposition(O).two_body.gauge


def cases():
    """(label, evaluator, O, gauge) over policies, maps and two larger systems."""
    for policy, nelec in POLICIES.items():
        problem = problem_of(nelec)
        for partition in MAPS:
            ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
            O = random_gauge(problem.data.nchol, 7)
            yield f"{policy}/{partition}", ev, O, gauge_of(ev, O)
    for partition in MAPS:
        problem = small_problem(ZERO_EIGENVALUE, nelec=(2, 1))
        ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
        yield f"zero-eigenvalue/{partition}", ev, np.eye(2), gauge_of(ev, np.eye(2))
    m = molecule("lih_sto3g")
    problem = prepare_problem(m["h1"], m["eri"], m["chol"], nelec=m["nelec"])
    for partition in MAPS:
        ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
        O = random_gauge(problem.data.nchol, 3)
        yield f"lih/{partition}", ev, O, gauge_of(ev, O)


def caught(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as exc:          # noqa: BLE001 - the tests compare type and message
        return type(exc), str(exc)
    return None


class AmplitudeMapTests(unittest.TestCase):
    def test_matches_the_per_source_loop_bitwise(self):
        for label, ev, O, gauge in cases():
            bounds = _partition_intervals(gauge.eigenvalues, ev.sources, TAU2, ev.partition, 1.0)
            rng = np.random.default_rng(len(label))
            for t in (np.zeros(len(ev.sources)), bounds[:, 0] + rng.uniform(size=len(ev.sources))*(bounds[:, 1]-bounds[:, 0])):
                with self.subTest(case=label):
                    _, actual = fixed_partition_amplitudes(gauge.eigenvalues, ev.sources, TAU2, t, ev.partition)
                    expected = ref.amplitudes(gauge.eigenvalues, ev.sources, TAU2, t, ev.partition)
                    assert_close(actual, expected, label=label)
                    self.assertFalse(actual.flags.writeable)

    def test_absent_legs_are_positive_zero(self):
        problem = small_problem(ZERO_EIGENVALUE, nelec=(2, 1))
        for partition in MAPS:
            ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
            _, d = fixed_partition_amplitudes(gauge_of(ev, None).eigenvalues, ev.sources, TAU2, .3, partition)
            zero = d == 0
            self.assertTrue(np.any(zero))
            self.assertFalse(np.any(np.signbit(d[zero])))

    def test_errors_name_the_same_source(self):
        sources = source_table([DensitySource(0, (0, 1), "two_branch", (0, 1)),
                                DensitySource(0, (2, 1), "two_branch", (0, 1)),
                                DensitySource(1, (0, 2), "spin_quartet", None),
                                DensitySource(1, (1, 2), "spin_quartet", None)])
        # Source 1 has a subnormal left eigenvalue: its anchored left leg underflows at t=0,
        # its balanced legs underflow near t=-372.3.
        ell = np.array([[.5, -.4, 1e-323], [.3, 1e-30, -.2]])
        expected_errors = {
            ("balanced_log", (0., 0., 4., 5.)): "Source 2 (bix=1, modes=(0, 2)): partition leaves",
            ("balanced_log", (3., 0., 0., 0.)): "Source 0 (bix=0, modes=(0, 1)): partition leaves",
            ("balanced_log", (0., -372.3, 0., 0.)): "Source 1: amplitude underflow",
            ("balanced_log", (0., 0., 0., 0.)): None,
            ("anchored_log", (0., 0., 0., 0.)): "Source 1: amplitude underflow",
            ("anchored_log", (0., -372.3, 0., 0.)): "Source 1 (bix=0, modes=(2, 1)): partition leaves",
        }
        for (partition, t), message in expected_errors.items():
            with self.subTest(partition=partition, t=t):
                actual = caught(fixed_partition_amplitudes, ell, sources, TAU2, np.array(t), partition)
                self.assertEqual(actual, caught(ref.amplitudes, ell, sources, TAU2, np.array(t), partition))
                if message is None:
                    self.assertIsNone(actual)
                else:
                    self.assertIs(actual[0], ValueError)
                    self.assertTrue(actual[1].startswith(message), actual[1])


class IntervalTests(unittest.TestCase):
    SETTINGS = ({}, {"amplitude_limit": .3}, {"max_abs_t": .05}, {"amplitude_limit": .5, "max_abs_t": .7})

    def test_matches_the_per_source_loop(self):
        for label, ev, O, gauge in cases():
            for settings in self.SETTINGS:
                with self.subTest(case=label, **settings):
                    args = (gauge.eigenvalues, ev.sources, TAU2, ev.partition, 1.0)
                    actual = caught(_partition_intervals, *args, **settings)
                    expected = caught(ref.partition_intervals, *args, **settings)
                    self.assertEqual(actual, expected)
                    if actual is None:
                        # np.log against math.log: at most a few ulp of the log ends.
                        np.testing.assert_allclose(_partition_intervals(*args, **settings),
                                                   ref.partition_intervals(*args, **settings),
                                                   rtol=1e-15, atol=4e-15)

    def test_infeasible_source_is_the_first_one(self):
        for label, ev, O, gauge in cases():
            args = (gauge.eigenvalues, ev.sources, TAU2, ev.partition, 1.0)
            with self.subTest(case=label):
                actual = caught(_partition_intervals, *args, amplitude_limit=1e-6)
                self.assertEqual(actual, caught(ref.partition_intervals, *args, amplitude_limit=1e-6))
                if actual is not None:
                    self.assertIs(actual[0], GaugeFeasibilityError)


class PullbackTests(unittest.TestCase):
    def test_amplitude_pullback_matches_the_per_source_loop(self):
        options = GradientOptions()
        for label, ev, O, gauge in cases():
            with self.subTest(case=label):
                two = ev.decomposition(O, t=.1).two_body
                g_amplitude = np.random.default_rng(1).normal(size=(len(two.table), 2))
                expected = caught(ref.amplitude_pullback, two, g_amplitude, options)
                actual = caught(_amplitude_pullback, two, g_amplitude, options)
                self.assertEqual(actual, expected)
                if expected is not None:
                    continue
                gell, gt = _amplitude_pullback(two, g_amplitude, options)
                gell_ref, gt_ref = ref.amplitude_pullback(two, g_amplitude, options)
                assert_close(gt, gt_ref, label="gt")
                assert_close(gell, gell_ref, label="gell")

    def test_bound_pullback_matches_the_per_source_loop(self):
        options = GradientOptions()
        for label, ev, O, gauge in cases():
            if label.startswith("zero-eigenvalue/balanced"):
                continue                                    # rejected before the bound pullback
            for settings in ({}, {"amplitude_limit": .5, "max_abs_t": .7}):
                with self.subTest(case=label, **settings):
                    two = ev.decomposition(O, t=0.).two_body
                    rng = np.random.default_rng(2)
                    count = len(two.sources)
                    gt = rng.normal(size=count)
                    gt[::5] = 0.
                    z = rng.uniform(-1, 1, size=count)
                    z[1::7], z[2::7] = 1., -1.
                    limit, cap = settings.get("amplitude_limit"), settings.get("max_abs_t", 30.)
                    assert_close(_bound_pullback(two, gt, z, limit, cap, options),
                                 ref.bound_pullback(two, gt, z, limit, cap, options), label="bound pullback")

    def test_bound_ties_name_the_same_source(self):
        options = GradientOptions()
        ties = set()
        for label, ev, O, gauge in cases():
            if label.startswith("zero-eigenvalue/balanced"):
                continue
            two = ev.decomposition(O, t=0.).two_body
            bounds = _partition_intervals(gauge.eigenvalues, ev.sources, TAU2, ev.partition, 1.0,
                                          amplitude_limit=.5)
            gt = np.ones(len(two.sources))
            for z_value, end_name in ((0., "lower"), (1., "upper")):   # z=1: only the upper end moves
                z = np.full(len(two.sources), z_value)
                for end, k in ((1, len(two.sources)//2), (0, len(two.sources)//3)):
                    cap = abs(float(bounds[k, end]))
                    if not np.isfinite(cap) or cap == 0:
                        continue
                    with self.subTest(case=label, end=end, z=z_value):
                        actual = caught(_bound_pullback, two, gt, z, .5, cap, options)
                        self.assertEqual(actual, caught(ref.bound_pullback, two, gt, z, .5, cap, options))
                        if actual is not None:
                            self.assertIn(f"{end_name} amplitude bound ties", actual[1])
                            ties.add(end_name)
        self.assertEqual(ties, {"lower", "upper"})

    def test_eigh_pullback_matches_the_per_factor_formula(self):
        rng = np.random.default_rng(4)
        r, n = 5, 6
        A = rng.normal(size=(r, n, n))
        ell, X = np.linalg.eigh(A + np.swapaxes(A, 1, 2))
        ge, gX = rng.normal(size=(r, n)), rng.normal(size=(r, n, n))
        batched = _eigh_pullback(ell, X, ge, gX)
        for g in range(r):
            expected = ref.eigh_pullback(ell[g], X[g], ge[g], gX[g])
            assert_close(batched[g], expected, label=f"factor {g}")
            assert_close(gradients.eigh_vjp(ell[g], X[g], ge[g], gX[g]), expected, label=f"eigh_vjp {g}")

    def test_first_degenerate_factor_is_reported(self):
        ell = np.array([[1., 2., 3.], [1., 1., 2.], [0., 1., 2.], [3., 3., 3.]])
        with self.assertRaisesRegex(GaugeRegularityError, r"^Factor 1: Repeated or near-degenerate"):
            _check_eigengaps(ell, GradientOptions())
        _check_eigengaps(ell[[0, 2]], GradientOptions())


class EarlyRegularityTests(unittest.TestCase):
    def test_degenerate_gauge_fails_before_any_inner_work(self):
        problem = small_problem(DEGENERATE)
        ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        with patch.object(gradients, "_solve_partitions", side_effect=AssertionError("inner solve ran")), \
             patch.object(gradients, "_assemble", side_effect=AssertionError("loss pullback ran")):
            for call in (ev.relaxed, ev.fixed, ev.fixed_z):
                with self.subTest(call=call.__name__), \
                     self.assertRaisesRegex(GaugeRegularityError, "^Factor 0: Repeated or near-degenerate"):
                    call(np.eye(2))

    def test_balanced_zero_eigenvalue_fails_before_any_inner_work(self):
        problem = small_problem(ZERO_EIGENVALUE, nelec=(2, 1))
        balanced = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        with patch.object(gradients, "_solve_partitions", side_effect=AssertionError("inner solve ran")):
            with self.assertRaisesRegex(GaugeRegularityError, "zero/near-zero"):
                balanced.relaxed(np.eye(2))
        anchored = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition="anchored_log")
        check_gauge_regularity(gauge_of(anchored, np.eye(2)), anchored.sources, "anchored_log")
        self.assertTrue(np.isfinite(anchored.relaxed(np.eye(2)).gradient_O).all())

    def test_rejected_trial_records_no_inner_work(self):
        problem = small_problem(DEGENERATE)
        objective = ExponentialGaugeObjective(GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2))
        with self.assertRaises(GaugeRegularityError):
            objective.evaluate([0.])
        self.assertEqual((objective.failed_evaluations, objective.inner_sweeps), (1, 0))

    def test_public_gradients_still_check(self):
        from ipie.hamiltonians.optimize_cholesky.channels import build_fixed_decomposition
        from ipie.hamiltonians.optimize_cholesky.problem import build_gauge_state
        problem = small_problem(DEGENERATE)
        decomposition = build_fixed_decomposition(problem, build_gauge_state(problem.data), TAU1, TAU2)
        with self.assertRaisesRegex(GaugeRegularityError, "^Factor 0: Repeated"):
            gradients.fixed_partition_gradient(problem, decomposition)


class OncePerGaugeTests(unittest.TestCase):
    def test_one_expm_per_outer_evaluation(self):
        m = molecule("lih_sto3g")
        problem = prepare_problem(m["h1"], m["eri"], m["chol"], nelec=m["nelec"])
        objective = ExponentialGaugeObjective(GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2))
        with patch("ipie.hamiltonians.optimize_cholesky.problem.expm", wraps=scipy_expm) as counted:
            objective.evaluate(theta(problem.data.nchol, .1, 0))
        self.assertEqual(counted.call_count, 1)

    def test_exponential_gauge_vjp_is_the_frechet_adjoint_and_still_validates(self):
        r = 4
        rng = np.random.default_rng(8)
        th, E = rng.normal(size=r*(r-1)//2), rng.normal(size=(r, r))
        reference = exponential_gauge(rng.normal(size=r*(r-1)//2), r)
        adjoint = expm_frechet(skew_from_parameters(th, r).T, E@reference.T, compute_expm=False)
        rows, cols = np.triu_indices(r, 1)
        assert_close(exponential_gauge_vjp(th, r, E, reference), adjoint[rows, cols]-adjoint[cols, rows])
        with self.assertRaises(ValidationError):
            exponential_gauge_vjp(th, r, E, 2*reference)
        with self.assertRaises(ValueError):
            exponential_gauge_vjp(th[:-1], r, E)
        with self.assertRaises(ValueError):
            exponential_gauge_vjp(th, r, E[:-1])

    def test_one_mode_space_per_relaxed_evaluation(self):
        for policy, nelec in POLICIES.items():
            ev = GaugeEvaluator(problem_of(nelec), tau1=TAU1, tau2=TAU2)
            O = random_gauge(ev.problem.data.nchol, 2)
            built = []
            original = ModeSpace.__init__

            def counting(self, *args, **kwargs):
                built.append(1)
                original(self, *args, **kwargs)

            with self.subTest(policy=policy), patch.object(ModeSpace, "__init__", counting):
                ev.relaxed(O, options=InnerOptions(optimize_shift=True))
                self.assertEqual(len(built), 1)


if __name__ == "__main__":
    unittest.main()
