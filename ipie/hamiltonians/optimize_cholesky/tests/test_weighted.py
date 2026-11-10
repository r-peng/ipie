"""Weighted loss L_w = sum_{i<j} a_i a_j ||[u_i,u_j]||^2 with unnormalized a_i."""

import math
import unittest

import numpy as np
from scipy.linalg import expm

from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator,
    OverlapEvaluator,
    build_fixed_decomposition,
    build_gauge_state,
    check_gauge_direction,
    commutator_loss,
    enumerate_sources,
    exponential_gauge,
    objective_loss,
    optimize_partitions,
    pair_scales,
    partition_coefficients,
    prepare_problem,
    projected_partition_gradient,
    raw_loss,
    with_partitions,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import example_tensors


POLICIES = ((2, 1), (1, 1), (2, 0))


def fixture(nelec=(2, 1), *, tau1=.03, tau2=.04, partition="balanced_log", t=None, seed=23):
    h, eri, chol = example_tensors(seed=seed)
    problem = prepare_problem(h, eri, chol, nelec=nelec)
    gauge = build_gauge_state(problem.data, exponential_gauge([0.37], 2))
    count = len(enumerate_sources(problem))
    t = np.linspace(-0.23, 0.31, count) if t is None else t
    return problem, build_fixed_decomposition(problem, gauge, tau1, tau2, t, partition=partition)


def two_orbital_problem(seed=5):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2, 2))
    h = (A+A.T)/2
    chol = rng.normal(size=(3, 2, 2))*.3
    chol = (chol+np.swapaxes(chol, 1, 2))/2
    return prepare_problem(h, np.einsum("gpr,gqs->prqs", chol, chol), chol, nelec=(1, 1))


class WeightedLossTests(unittest.TestCase):
    def test_scaled_overlap_total_matches_dense_explicit_weights(self):
        for nelec in POLICIES:
            for partition in ("balanced_log", "anchored_log"):
                with self.subTest(nelec=nelec, partition=partition):
                    _, decomposition = fixture(nelec, partition=partition)
                    overlap = objective_loss(decomposition, loss_kind="weighted")
                    dense = objective_loss(decomposition, loss_kind="weighted", method="dense")
                    diagnostic = commutator_loss(decomposition, normalize_weights=False).weighted_comm2
                    np.testing.assert_allclose(overlap, dense, rtol=1e-12, atol=0)
                    np.testing.assert_allclose(overlap, diagnostic, rtol=1e-12, atol=0)
                    # Normalized diagnostics differ by exactly 1/denom^2.
                    normalized = commutator_loss(decomposition).weighted_comm2
                    np.testing.assert_allclose(normalized*decomposition.denom**2, overlap, rtol=1e-12)

    def test_raw_kind_is_the_original_objective(self):
        _, decomposition = fixture()
        self.assertEqual(objective_loss(decomposition), raw_loss(decomposition))
        # total() uses math.fsum; the legacy diagnostic sums sequentially.
        np.testing.assert_allclose(OverlapEvaluator(decomposition).total(), raw_loss(decomposition),
                                   rtol=1e-14, atol=0)
        np.testing.assert_array_equal(pair_scales(decomposition), np.ones(decomposition.nterms))
        np.testing.assert_array_equal(pair_scales(decomposition, "weighted"), decomposition.coefficients)
        with self.assertRaises(ValueError):
            pair_scales(decomposition, "normalized")
        with self.assertRaises(ValueError):
            OverlapEvaluator(decomposition, loss_kind="weighted").evaluate()

    def test_weighted_loss_does_not_depend_on_tau2_at_fixed_t(self):
        for partition in ("balanced_log", "anchored_log"):
            with self.subTest(partition=partition):
                _, first = fixture(tau2=.04, partition=partition)
                _, second = fixture(tau2=.02, partition=partition)
                np.testing.assert_allclose(objective_loss(first, loss_kind="weighted"),
                                           objective_loss(second, loss_kind="weighted"), rtol=1e-12)
                self.assertGreater(abs(raw_loss(first)-raw_loss(second)), 1e-6*raw_loss(first))

    def test_weighted_source_profile_is_exact(self):
        _, decomposition = fixture()
        coefficients = partition_coefficients(decomposition, loss_kind="weighted")
        base = objective_loss(decomposition, loss_kind="weighted")
        for i in (0, 3, len(coefficients)-1):
            c = coefficients[i]
            self.assertAlmostEqual(c.value(c.t), base, delta=1e-12*base)
            for delta in (-.2, .15):
                t = np.array(decomposition.two_body.t, dtype=float)
                t[i] += delta
                actual = objective_loss(with_partitions(decomposition, t), loss_kind="weighted")
                np.testing.assert_allclose(c.value(t[i]), actual, rtol=1e-11)

    def test_weighted_inner_solve_is_monotone_and_stationary(self):
        problem, decomposition = fixture()
        result = optimize_partitions(decomposition, loss_kind="weighted", gradient_atol=1e-12,
                                     gradient_rtol=0, max_sweeps=500)
        self.assertTrue(result.converged)
        self.assertEqual(result.loss_kind, "weighted")
        losses = [row.loss for row in result.history]
        self.assertTrue(all(b <= a*(1+1e-14) for a, b in zip(losses, losses[1:])))
        np.testing.assert_allclose(result.loss, objective_loss(result.decomposition, loss_kind="weighted"),
                                   rtol=1e-12)
        fresh = partition_coefficients(result.decomposition, loss_kind="weighted")
        residual = projected_partition_gradient(result.t, [c.gradient for c in fresh], result.bounds)
        self.assertLessEqual(np.max(np.abs(residual)), 1e-11)

    def test_weighted_fixed_and_relaxed_gradients_match_central_differences(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        O = exponential_gauge([0.37], 2)
        K = np.array([[0., 1.], [-1., 0.]])
        for partition in ("balanced_log", "anchored_log"):
            engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, partition=partition, loss_kind="weighted")
            with self.subTest(partition=partition, mode="fixed_t"):
                check = check_gauge_direction(lambda X: engine.fixed(X, t=.1), O, K, steps=(1e-4, 1e-5))
                self.assertLess(check.differences[-1].relative_error, 1e-7)
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        check = check_gauge_direction(
            lambda X: engine.relaxed(X, gradient_atol=1e-13, gradient_rtol=0), O, K, steps=(1e-4, 1e-5))
        self.assertLess(check.differences[-1].relative_error, 1e-6)

    def test_two_orbital_balanced_zero_is_the_weighted_inner_optimum(self):
        problem = two_orbital_problem()
        engine = GaugeEvaluator(problem, tau1=.05, tau2=.05, loss_kind="weighted")
        O = expm(np.array([[0, .3, .1], [-.3, 0, .2], [-.1, -.2, 0]]))
        relaxed = engine.relaxed(O)
        self.assertEqual(relaxed.inner_result.sweeps, 0)
        np.testing.assert_array_equal(relaxed.inner_result.t, np.zeros(len(engine.sources)))
        self.assertTrue(math.isclose(relaxed.loss, engine.fixed(O, t=0.).loss, rel_tol=1e-13))


if __name__ == "__main__":
    unittest.main()
