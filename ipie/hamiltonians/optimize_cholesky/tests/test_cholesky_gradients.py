"""Step 10: gauge and Cholesky-shift gradients with a shift mu[gamma, sigma].

With a shift the channels use lambda[s, g, a] = ell[g, a] - mu[g, s], and the
one-body matrices h_s = h + sum_g (m_g - mu[g, s]) L_g(O), m = mu @ N, depend
on the gauge. With P[g, s] = <dL/dh_s, L_g>:

    dL/dL_g += sum_s (m_g - mu[g, s]) dL/dh_s,
    dL/dmu[g, s] = -sum_a dL/dlambda[s, g, a] + N_s sum_s' P[g, s'] - P[g, s].

GaugeGradient.cholesky_gradient reports the second line, (r, 2) per spin or
(r,) tied (column 0 for aa_only, whose mu_down is 0). The tests check:
  * central differences in O and in mu for fixed_t, fixed_z and relaxed, on
    every policy, both maps, tied and split shifts;
  * a relaxed one-body shift held on its lower bound, and sources on
    amplitude walls (the moving-bound terms);
  * the tied gradient is the sum of the split columns;
  * zero shift: gradient_O is unchanged, and a requested mu gradient is right;
  * at a tied shift the split fixed-z gradient is the mean of the one-sided
    derivatives (the interval end takes a maximum over spins);
  * public functions agree with the evaluator, shared and per-spin one-body
    bases agree;
  * regularity and validation errors, and optimize_gauge at a fixed shift.
"""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from ipie.hamiltonians.optimize_cholesky import gradients
from ipie.hamiltonians.optimize_cholesky.channels import build_fixed_decomposition, leg_log_bases
from ipie.hamiltonians.optimize_cholesky.gradients import (
    GaugeEvaluator,
    _shift_wall,
    fixed_partition_gradient,
    normalized_partition_gradient,
    relaxed_gauge_gradient,
)
from ipie.hamiltonians.optimize_cholesky.models import (
    FixedDecomposition,
    GaugeRegularityError,
    InnerOptions,
    LBFGSOptions,
)
from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
from ipie.hamiltonians.optimize_cholesky.partitions import optimize_partitions
from ipie.hamiltonians.optimize_cholesky.problem import build_gauge_state, exponential_gauge, prepare_problem
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import check_norm, example_tensors, physical2MB
from ipie.hamiltonians.optimize_cholesky.tests.test_cholesky_shift import (
    MAPS,
    SECTORS,
    TAU1,
    TAU2,
    build,
    fixture,
    green_function,
    per_spin,
    sector,
    shift_of,
)
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import assert_close, assert_same_evaluation

MODES = ("fixed", "fixed_z", "relaxed")
TIGHT = InnerOptions(optimize_shift=True, max_sweeps=20000, gradient_atol=1e-13, gradient_rtol=1e-12,
                     shift_rtol=1e-13)
FD_RTOL = 1e-6


def evaluate(evaluator, mode, O, *, t0=None, inner=TIGHT, **kwargs):
    if mode == "fixed":
        return evaluator.fixed(O, t=.1, **kwargs)
    if mode == "fixed_z":
        return evaluator.fixed_z(O, z=.3, **kwargs)
    return evaluator.relaxed(O, t0=t0, options=inner, **kwargs)


def skew(rng, r):
    K = rng.normal(size=(r, r))
    return K - K.T


def direction(problem, kind, rng):
    """An admissible (r, 2) shift direction: tied columns for 'tied', mu_down fixed at 0 for aa_only."""
    r = problem.data.nchol
    E = rng.normal(size=(r, 2))
    if kind == "tied":
        E[:, 1] = E[:, 0]
    if problem.policy == "aa_only":
        E[:, 1] = 0.
    return E


def along(gradient, E):
    """Directional derivative from a reported gradient: (r, 2) per spin, or (r,) tied (column 0 of E)."""
    g = np.asarray(gradient)
    return float(np.sum(g*E)) if g.ndim == 2 else float(np.sum(g*E[:, 0]))


def scale_of(gradient, E):
    """|g| |E| in the reported form (column 0 of E for a tied gradient)."""
    g = np.asarray(gradient)
    return float(np.linalg.norm(g)*np.linalg.norm(E if g.ndim == 2 else E[:, 0]))


def central_error(f, analytic, scale=0., steps=(1e-4, 1e-5, 1e-6)):
    """Smallest relative error of central differences of f at 0 against analytic.

    scale: |gradient| |direction|, a floor for the denominator when the
    directional derivative cancels.
    """
    errors = []
    for h in steps:
        finite = (f(h) - f(-h))/(2*h)
        errors.append(abs(finite - analytic)/max(abs(finite), abs(analytic), scale, 1e-300))
    return min(errors)


def one_sided(f, h=1e-5):
    """Second-order forward and backward differences of f at 0."""
    f0 = f(0.)
    return (-3*f0 + 4*f(h) - f(2*h))/(2*h), (3*f0 - 4*f(-h) + f(-2*h))/(2*h)


def stored(evaluator):
    """The engine's (r, 2) shift as an array (zeros for none)."""
    mu = evaluator.cholesky_shift
    return np.zeros((evaluator.problem.data.nchol, 2)) if mu is None else np.asarray(mu)


def minimum_limit(evaluator, O):
    """Smallest amplitude limit that leaves every source a feasible t."""
    two = evaluator.decomposition(O).two_body
    logs, _ = leg_log_bases(two.spin_eigenvalues, two.sources, np.log(two.tau2), two.partition)
    top = logs.max(axis=2)
    return float(np.exp(np.max(top.sum(axis=1)/2)))


def check_gradients(case, evaluator, mode, O, kind, rng, inner=TIGHT):
    """Central differences of the loss in O (direction K O) and in mu (direction E)."""
    center = evaluate(evaluator, mode, O, inner=inner)
    t0 = center.decomposition.two_body.t if mode == "relaxed" else None
    r = evaluator.problem.data.nchol
    K = skew(rng, r)
    error = central_error(lambda s: evaluate(evaluator, mode, expm(s*K) @ O, t0=t0, inner=inner).loss,
                          float(np.sum(center.gradient_O*(K @ O))),
                          np.linalg.norm(center.gradient_O)*np.linalg.norm(K @ O))
    case.assertLess(error, FD_RTOL, "gauge direction")
    mu, E = stored(evaluator), direction(evaluator.problem, kind, rng)
    error = central_error(lambda s: evaluate(evaluator, mode, O, t0=t0, inner=inner,
                                             cholesky_shift=mu + s*E).loss,
                          along(center.cholesky_gradient, E), scale_of(center.cholesky_gradient, E))
    case.assertLess(error, FD_RTOL, "shift direction")
    return center


class FiniteDifferenceTests(unittest.TestCase):
    def test_gauge_and_shift_gradients_match_central_differences(self):
        rng = np.random.default_rng(10)
        for nelec in SECTORS:
            problem, gauge = fixture(nelec)
            r = problem.data.nchol
            for partition in MAPS:
                for kind in ("tied", "split"):
                    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                               shift=(.05, 0.) if nelec[1] == 0 else (.05, -.02),
                                               cholesky_shift=shift_of(kind, problem))
                    for mode in MODES:
                        with self.subTest(nelec=nelec, partition=partition, shift=kind, mode=mode):
                            center = check_gradients(self, evaluator, mode, np.asarray(gauge.O), kind, rng)
                            tied = kind == "tied" or problem.policy == "aa_only"
                            self.assertEqual(np.shape(center.cholesky_gradient), (r,) if tied else (r, 2))
                            self.assertEqual(center.mode, "fixed_t" if mode == "fixed" else mode)

    def test_one_body_shift_on_its_lower_bound(self):
        """mu_s = max(eps_s) - rho_1/tau1 moves with h_s; without that term the gradient is wrong."""
        rng = np.random.default_rng(11)
        inner = replace(TIGHT, one_body_limit=.002)
        for nelec in ((2, 1), (2, 0)):
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            for partition in MAPS:
                for kind in ("tied", "split"):
                    with self.subTest(nelec=nelec, partition=partition, shift=kind):
                        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                                   cholesky_shift=shift_of(kind, problem))
                        center = check_gradients(self, evaluator, "relaxed", O, kind, rng, inner)
                        result = center.inner_result
                        wall = _shift_wall(result)
                        self.assertEqual(wall, tuple(s in (0,) + ((1,) if nelec[1] else ()) for s in (0, 1)))
                        t, bounds = result.t, result.bounds
                        self.assertFalse(np.any((t == bounds[:, 0]) | (t == bounds[:, 1])))
                        self.assertGreater(np.linalg.norm(center.bound_correction_O), 1e-4)
                        with patch.object(gradients, "_shift_wall", return_value=None):
                            without = evaluator.relaxed(O, options=inner)
                        K = skew(rng, problem.data.nchol)
                        error = central_error(
                            lambda s: evaluator.relaxed(expm(s*K) @ O, t0=t, options=inner).loss,
                            float(np.sum(without.gradient_O*(K @ O))))
                        self.assertGreater(error, 1e-4)

    def test_sources_on_amplitude_walls(self):
        rng = np.random.default_rng(12)
        for nelec in SECTORS:
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            for partition in MAPS:
                active = 0
                for kind in ("tied", "split"):
                    with self.subTest(nelec=nelec, partition=partition, shift=kind):
                        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                                   cholesky_shift=shift_of(kind, problem))
                        inner = replace(TIGHT, amplitude_limit=1.025*minimum_limit(evaluator, O))
                        center = check_gradients(self, evaluator, "relaxed", O, kind, rng, inner)
                        t, bounds = center.inner_result.t, center.inner_result.bounds
                        active += int(np.sum((t == bounds[:, 0]) | (t == bounds[:, 1])))
                with self.subTest(nelec=nelec, partition=partition):
                    self.assertGreater(active, 0)


class ConsistencyTests(unittest.TestCase):
    def test_tied_gradient_is_the_sum_of_split_columns(self):
        for nelec in SECTORS:
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            for partition in MAPS:
                mu = shift_of("tied", problem)
                tied = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition, cholesky_shift=mu)
                split = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition, cholesky_shift=mu,
                                       cholesky_gradient="split")
                for mode in MODES:
                    with self.subTest(nelec=nelec, partition=partition, mode=mode):
                        a, b = evaluate(tied, mode, O), evaluate(split, mode, O)
                        columns = np.asarray(b.cholesky_gradient)
                        self.assertEqual(columns.shape, (problem.data.nchol, 2))
                        if problem.policy == "aa_only":
                            np.testing.assert_array_equal(columns[:, 1], 0.)
                            assert_close(columns[:, 0], a.cholesky_gradient)
                        else:
                            assert_close(columns.sum(axis=1), a.cholesky_gradient)
                        assert_close(b.gradient_O, a.gradient_O)
                        assert_close(b.loss, a.loss)

    def test_zero_shift(self):
        """gradient_O is unchanged when a shift gradient is requested at mu = 0, and that gradient is right."""
        rng = np.random.default_rng(13)
        for nelec in SECTORS:
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            for partition in MAPS:
                plain = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition)
                for want in ("tied", "split"):
                    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                               cholesky_gradient=want)
                    for mode in MODES:
                        with self.subTest(nelec=nelec, partition=partition, want=want, mode=mode):
                            a, b = evaluate(plain, mode, O), evaluate(evaluator, mode, O)
                            self.assertIsNone(a.cholesky_gradient)
                            assert_same_evaluation(b, a)
                            if want == "split" and mode == "fixed_z" and problem.policy == "hybrid":
                                continue          # a kink: test_split_gradient_at_a_tied_shift_is_a_mean
                            E = direction(problem, want, rng)
                            t0 = b.decomposition.two_body.t if mode == "relaxed" else None
                            error = central_error(
                                lambda s: evaluate(plain, mode, O, t0=t0, cholesky_shift=s*E).loss,
                                along(b.cholesky_gradient, E), scale_of(b.cholesky_gradient, E))
                            self.assertLess(error, FD_RTOL)

    def test_split_gradient_at_a_tied_shift_is_a_mean(self):
        """At equal spins a quartet leg's interval end is a maximum over two equal spins.

        The fixed-z loss then has a kink in each spin's direction (not in the
        tied one); the split gradient is the mean of the one-sided derivatives.
        """
        for nelec in ((2, 1), (2, 2)):
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            r = problem.data.nchol
            for partition in MAPS:
                for base in ("zero", "tied"):
                    mu = np.zeros((r, 2)) if base == "zero" else np.repeat(shift_of("tied", problem)[:, None], 2, 1)
                    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                               cholesky_shift=mu, cholesky_gradient="split")
                    columns = np.asarray(evaluator.fixed_z(O, z=.3).cholesky_gradient)
                    for spin in (0, 1):
                        with self.subTest(nelec=nelec, partition=partition, base=base, spin=spin):
                            E = np.zeros((r, 2))
                            E[:, spin] = 1.
                            forward, backward = one_sided(
                                lambda s: evaluator.fixed_z(O, z=.3, cholesky_shift=mu + s*E).loss)
                            self.assertGreater(abs(forward - backward), 1.)
                            analytic = float(columns[:, spin].sum())
                            self.assertLess(abs(analytic - (forward + backward)/2), 1e-5*abs(analytic))

    def test_public_functions_match_the_evaluator(self):
        inner = InnerOptions(optimize_shift=True, max_sweeps=20000)
        for nelec in SECTORS:
            problem, gauge = fixture(nelec)
            O = np.asarray(gauge.O)
            for partition in MAPS:
                for kind in ("tied", "split"):
                    for want in (None, "tied", "split"):
                        with self.subTest(nelec=nelec, partition=partition, shift=kind, want=want):
                            evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=partition,
                                                       cholesky_shift=shift_of(kind, problem),
                                                       cholesky_gradient=want)
                            a = evaluator.fixed(O, t=.1)
                            b = fixed_partition_gradient(problem, a.decomposition, cholesky_gradient=want)
                            c = evaluator.fixed_z(O, z=.3)
                            d = normalized_partition_gradient(problem, c.decomposition, cholesky_gradient=want)
                            e = evaluator.relaxed(O, options=inner)
                            f = relaxed_gauge_gradient(problem, e.inner_result, cholesky_gradient=want)
                            for x, y in ((a, b), (c, d), (e, f)):
                                assert_close(y.gradient_O, x.gradient_O)
                                assert_close(y.cholesky_gradient, x.cholesky_gradient)
                                self.assertEqual(y.mode, x.mode)

    def test_shared_and_per_spin_bases_agree(self):
        """A per-spin request on a shared-basis decomposition equals the per-spin representation."""
        for nelec in ((2, 1), (1, 1), (2, 2)):
            problem, gauge = fixture(nelec)
            for partition in MAPS:
                for kind in (None, "tied"):
                    with self.subTest(nelec=nelec, partition=partition, shift=kind):
                        mu = None if kind is None else shift_of(kind, problem)
                        decomposition = build(problem, gauge, mu, partition)
                        self.assertFalse(decomposition.one_body.split)
                        built = build_fixed_decomposition(problem, gauge, TAU1, TAU2, .1, partition=partition,
                                                          shift=decomposition.one_body.shift,
                                                          cholesky_shift=mu, per_spin=True)
                        self.assertTrue(built.one_body.split and built.two_body.split)
                        for function in (fixed_partition_gradient, normalized_partition_gradient):
                            a = function(problem, decomposition, cholesky_gradient="split")
                            for other in (per_spin(decomposition), built):
                                b = function(problem, other, cholesky_gradient="split")
                                assert_close(a.gradient_O, b.gradient_O)
                                assert_close(a.cholesky_gradient, b.cholesky_gradient)

    def test_optimize_gauge_at_a_fixed_shift(self):
        """The outer loop runs with a fixed Cholesky shift; the result stays exact on the sector."""
        problem, gauge = fixture((2, 1))
        mu = shift_of("split", problem)
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=mu)
        result = optimize_gauge(evaluator, theta0=np.full(1, .3), options=LBFGSOptions(max_iterations=5),
                                inner_options=InnerOptions(optimize_shift=True))
        self.assertLess(result.loss, result.initial_loss)
        final = result.decomposition
        np.testing.assert_array_equal(final.cholesky_shift, evaluator.cholesky_shift)
        assert_close(result.evaluation.gauge_gradient.gradient_O,
                     evaluator.relaxed(result.O, options=InnerOptions(optimize_shift=True)).gradient_O,
                     rtol=1e-9)
        basis, basis_map = sector(problem)
        H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
        check_norm(green_function(final, basis, basis_map, problem.data.nbasis),
                   final.lambda_electronic*np.eye(len(basis)) - H)


class RegularityTests(unittest.TestCase):
    @staticmethod
    def degenerate_problem(nelec=(2, 1)):
        """example_tensors with h replaced by one with a repeated eigenvalue."""
        h, eri, chol = example_tensors(n=3, r=2)
        Q, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(3, 3)))
        return prepare_problem((Q*np.array([-.2, -.2, .3])) @ Q.T, eri, chol, nelec=nelec)

    def test_degenerate_h_has_no_shift_gradient_at_zero_shift(self):
        problem = self.degenerate_problem()
        O = exponential_gauge([.3], 2)
        plain = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2)
        self.assertTrue(np.isfinite(plain.relaxed(O).gradient_O).all())
        for want in ("tied", "split"):
            evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_gradient=want)
            with patch.object(gradients, "_solve_partitions", side_effect=AssertionError("inner solve ran")):
                for method in (evaluator.fixed, evaluator.fixed_z, evaluator.relaxed):
                    with self.subTest(want=want, method=method.__name__), \
                            self.assertRaisesRegex(GaugeRegularityError, "One-body matrix h_sigma"):
                        method(O)
            decomposition = build(problem, build_gauge_state(problem.data, O), None)
            with self.subTest(want=want, public=True), \
                    self.assertRaisesRegex(GaugeRegularityError, "One-body matrix h_sigma"):
                fixed_partition_gradient(problem, decomposition, cholesky_gradient=want)
            # A shift lifts the degeneracy of h_sigma.
            shifted = evaluator.fixed(O, cholesky_shift=shift_of(want, problem))
            self.assertTrue(np.isfinite(shifted.gradient_O).all())

    def test_split_h_reports_the_degenerate_spin(self):
        """With N = (2, 1), mu_up = 0 and mu_down = v: h_up = h + v L, while h_down = h + (N_down - 1) v L = h."""
        problem = self.degenerate_problem((2, 1))
        r = problem.data.nchol
        mu = np.zeros((r, 2))
        mu[:, 1] = shift_of("tied", problem)
        evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=mu)
        O = exponential_gauge([.3], 2)
        eps = evaluator.decomposition(O).one_body.eigenvalues
        self.assertEqual(np.shape(eps), (2, problem.data.nbasis))
        self.assertLess(np.diff(eps[1]).min(), 1e-12)
        self.assertGreater(np.diff(eps[0]).min(), 1e-3)
        with self.assertRaisesRegex(GaugeRegularityError, r"h_sigma \(spin down\)"):
            evaluator.fixed(O)

    def test_balanced_zero_leg_scale_is_rejected_before_inner_work(self):
        problem, gauge = fixture((2, 1))
        ell = np.asarray(gauge.eigenvalues)
        r = problem.data.nchol
        mu = np.zeros((r, 2))
        mu[0, 0] = ell[0, 1]                             # lambda_up[0, 1] = 0 exactly
        mu[:, 1] = .07
        balanced = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=mu)
        with patch.object(gradients, "_solve_partitions", side_effect=AssertionError("inner solve ran")), \
                self.assertRaisesRegex(GaugeRegularityError, r"zero/near-zero eigenvalue \(leg scale\)"):
            balanced.relaxed(gauge.O)
        anchored = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition="anchored_log", cholesky_shift=mu)
        self.assertTrue(np.isfinite(anchored.relaxed(gauge.O).gradient_O).all())


class ValidationTests(unittest.TestCase):
    def test_requests_and_particle_numbers(self):
        problem, gauge = fixture((2, 1))
        with self.assertRaisesRegex(ValueError, "cholesky_gradient"):
            GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_gradient="both")
        decomposition = build(problem, gauge, shift_of("split", problem))
        with self.assertRaisesRegex(ValueError, "cholesky_gradient"):
            fixed_partition_gradient(problem, decomposition, cholesky_gradient=True)
        h, eri, chol = example_tensors(n=3, r=2)
        fock = prepare_problem(h, eri, chol)
        with self.assertRaisesRegex(ValueError, "particle numbers"):
            GaugeEvaluator(fock, tau1=TAU1, tau2=TAU2, cholesky_gradient="tied")
        plain = build_fixed_decomposition(fock, build_gauge_state(fock.data), TAU1, TAU2, .1)
        with self.assertRaisesRegex(ValueError, "particle numbers"):
            fixed_partition_gradient(fock, plain, cholesky_gradient="tied")

    def test_one_body_basis_must_match_the_gauge_and_shift(self):
        problem, gauge = fixture((2, 1))
        mu = shift_of("split", problem)
        other = build_gauge_state(problem.data, exponential_gauge([-.4], 2))
        stale = FixedDecomposition(build(problem, other, mu).one_body, build(problem, gauge, mu).two_body)
        with self.assertRaisesRegex(ValueError, r"do not reconstruct h_sigma \(spin 0\)"):
            fixed_partition_gradient(problem, stale)
        plain = build(problem, gauge, None)
        one = plain.one_body
        wrong = replace(plain, one_body=replace(one, X=np.ascontiguousarray(one.X[:, ::-1])))
        fixed_partition_gradient(problem, wrong)        # no shift gradient: h is not differentiated
        with self.assertRaisesRegex(ValueError, "do not reconstruct h_sigma"):
            fixed_partition_gradient(problem, wrong, cholesky_gradient="tied")

    def test_relaxed_gradient_of_a_supplied_result(self):
        """relaxed_gauge_gradient keeps its consistency checks with a shift."""
        problem, gauge = fixture((2, 1))
        decomposition = build(problem, gauge, shift_of("split", problem), "anchored_log")
        result = optimize_partitions(decomposition, TIGHT)
        derivative = relaxed_gauge_gradient(problem, result)
        self.assertEqual(derivative.mode, "relaxed")
        self.assertEqual(np.shape(derivative.cholesky_gradient), (problem.data.nchol, 2))
        stale = replace(result, loss=result.loss*(1 + 1e-6))
        with self.assertRaisesRegex(ValueError, "Stored inner loss"):
            relaxed_gauge_gradient(problem, stale)


if __name__ == "__main__":
    unittest.main()
