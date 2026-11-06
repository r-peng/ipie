"""Weak-Wolfe (Lewis-Overton) line search for kinked reduced objectives."""

from dataclasses import asdict
from types import SimpleNamespace
import math
import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky import (
    ExponentialEvaluation,
    GaugeEvaluator,
    GaugeRegularityError,
    LBFGSOptimizer,
    LBFGSOptions,
    OuterOptions,
    build_fixed_decomposition,
    build_gauge_state,
    near_zero_eigenvalues,
    optimize_gauge,
    prepare_problem,
)
from ipie.hamiltonians.optimize_cholesky.optimizers import _minimize_exponential
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_hamiltonian,
    example_tensors,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_optimizers import QuadraticOracle, readonly


WOLFE = dict(line_search="weak_wolfe", validate_final=False)


class FunctionOracle(QuadraticOracle):
    """Exact value/gradient of an arbitrary (possibly kinked) function f(theta)."""

    def __init__(self, function, gradient, size):
        super().__init__(np.eye(size), np.zeros(size))
        self.function, self.gradient = function, gradient

    def evaluate(self, theta):
        theta = np.asarray(theta, dtype=float)
        self.evaluations += 1
        self.visited.append(theta.copy())
        value, gradient = float(self.function(theta)), np.asarray(self.gradient(theta), dtype=float)
        inner = SimpleNamespace(converged=True, sweeps=2, projected_gradient_norm=0.)
        derivative = SimpleNamespace(loss=value, tangent_gradient=readonly(gradient.reshape(1, -1)),
                                     inner_result=inner,
                                     decomposition=SimpleNamespace(theta=readonly(theta)))
        point = ExponentialEvaluation(readonly(theta), readonly(gradient), derivative)
        self.points.append(point)
        self.inner_sweeps += inner.sweeps
        return point


def kinked_oracle():
    # f = 2|x0| + (x1-0.5)^2 + 0.1*x0*x1 has its minimizer f=0 ON the kink x0=0,
    # because the kink coefficient 2 exceeds the smooth slope 0.1*x1 = 0.05 there.
    def f(x):
        return 2*abs(x[0]) + (x[1]-.5)**2 + .1*x[0]*x[1]

    def g(x):
        return np.array([2*np.sign(x[0]) + .1*x[1], 2*(x[1]-.5) + .1*x[0]])
    return FunctionOracle(f, g, 2)


class WeakWolfeOptionTests(unittest.TestCase):
    def test_defaults_validation_and_legacy_round_trip(self):
        self.assertEqual(LBFGSOptions().line_search, "armijo")
        self.assertEqual(LBFGSOptions().wolfe_c2, .9)
        self.assertEqual(LBFGSOptions().wolfe_max_step_norm, math.pi)
        legacy = OuterOptions(4, 80, 3, line_search="weak_wolfe", wolfe_c2=.5, validate_final=False)
        converted = LBFGSOptimizer(QuadraticOracle([[1.]], [2.]), options=legacy).options
        self.assertEqual(asdict(converted), asdict(legacy))
        self.assertEqual((converted.max_iterations, converted.max_evaluations, converted.memory), (4, 80, 3))
        for kwargs in ({"line_search": "wolfe"}, {"wolfe_c2": 1e-4}, {"wolfe_c2": 1.},
                       {"armijo": .5, "wolfe_c2": .4}, {"wolfe_max_step_norm": .5},
                       {"wolfe_max_step_norm": math.inf}):
            for constructor in (LBFGSOptions, OuterOptions):
                with self.subTest(constructor=constructor.__name__, kwargs=kwargs), \
                        self.assertRaises(ValueError):
                    constructor(**kwargs)


class WeakWolfeControlledTests(unittest.TestCase):
    def test_quadratic_converges_and_accepted_steps_satisfy_both_conditions(self):
        oracle = QuadraticOracle([[4., 1.], [1., 3.]], [.3, -.4])
        options = LBFGSOptions(max_iterations=80, memory=4, max_step_norm=.75,
                               gradient_atol=1e-9, gradient_rtol=0, **WOLFE)
        result = _minimize_exponential(oracle, [1.2, .8], options)
        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.theta, oracle.target, atol=1e-8, rtol=0)
        self.assertTrue(all(row.curvature_update for row in result.history[1:]))
        self.assertTrue(all(row.acceptance == "weak_wolfe" for row in result.history[1:]))
        previous, accepted = oracle.points[0], 0
        for trial, point in zip(result.trials, oracle.points[1:]):
            if not trial.accepted:
                continue
            s = point.theta-previous.theta
            slope = float(np.dot(previous.gradient_theta, s))
            self.assertLess(slope, 0)
            self.assertLessEqual(point.loss, previous.loss+options.armijo*slope+1e-14)
            self.assertGreaterEqual(float(np.dot(point.gradient_theta, s)), options.wolfe_c2*slope-1e-14)
            previous, accepted = point, accepted+1
        self.assertEqual(accepted, result.iterations)
        self.assertIs(result.evaluation, previous)

    def test_kinked_minimizer_is_reached_where_gradient_tests_cannot_pass(self):
        oracle = kinked_oracle()
        result = _minimize_exponential(oracle, [.7, -.4], LBFGSOptions(max_iterations=200, **WOLFE))
        self.assertLess(result.loss, 1e-8)
        self.assertLess(abs(result.theta[0]), 1e-8)
        self.assertAlmostEqual(result.theta[1], .5, delta=1e-4)
        # The gradient never vanishes at a kink, so a gradient test cannot certify it.
        self.assertGreater(result.gradient_norm, 1.)
        self.assertIn(result.status, ("line_search_failed", "max_iterations"))
        if result.status == "line_search_failed":
            self.assertIn("nonsmooth", result.message)
        np.testing.assert_array_less(np.diff([row.loss for row in result.history]), 1e-15)

    def test_minimizer_on_a_curved_kink_surface_is_reached(self):
        # f = 2|x0-x1^2| + (x1-0.5)^2 + 0.05*x0: the minimizer lies on the curved
        # kink x0 = x1^2, so iterates must slide along a nonsmooth valley.
        def f(x):
            return 2*abs(x[0]-x[1]**2) + (x[1]-.5)**2 + .05*x[0]

        def g(x):
            sign = np.sign(x[0]-x[1]**2)
            return np.array([2*sign + .05, -4*x[1]*sign + 2*(x[1]-.5)])
        x1 = .5/1.05
        fmin = (x1-.5)**2 + .05*x1**2
        for start in ([.9, -.6], [-.5, 1.2], [1.5, .2]):
            with self.subTest(start=start):
                result = _minimize_exponential(FunctionOracle(f, g, 2), start,
                                               LBFGSOptions(max_iterations=300, **WOLFE))
                self.assertLess(result.loss-fmin, 1e-9)
                np.testing.assert_allclose(result.theta, [x1**2, x1], atol=1e-4)

    def test_curvature_failure_expands_the_step_up_to_the_cap(self):
        # f = 0.001(x-10)^2 from x=0: the curvature test needs alpha >= 5 when c2 = 0.5.
        for cap, alphas, acceptance in ((20., [1., 2., 4., 8.], "weak_wolfe"),
                                        (math.pi, [1., 2., math.pi], "expansion_capped")):
            with self.subTest(cap=cap):
                oracle = QuadraticOracle([[.002]], [10.])
                options = LBFGSOptions(max_iterations=1, wolfe_c2=.5, wolfe_max_step_norm=cap, **WOLFE)
                result = _minimize_exponential(oracle, [0.], options)
                np.testing.assert_allclose([trial.step_size for trial in result.trials], alphas)
                self.assertEqual([trial.accepted for trial in result.trials].count(True), 1)
                self.assertEqual(result.history[1].acceptance, acceptance)
                self.assertAlmostEqual(result.history[1].step_size, alphas[-1])
                np.testing.assert_allclose(result.theta, [alphas[-1]])

    def test_armijo_only_fallback_accepts_the_best_decreasing_trial(self):
        oracle = QuadraticOracle([[.002]], [10.])
        options = LBFGSOptions(max_iterations=1, max_line_search=2, wolfe_c2=.5,
                               wolfe_max_step_norm=20., **WOLFE)
        result = _minimize_exponential(oracle, [0.], options)
        self.assertEqual([trial.reason for trial in result.trials],
                         ["insufficient_curvature", "armijo_only"])
        self.assertEqual(result.history[1].acceptance, "armijo_only")
        np.testing.assert_allclose(result.theta, [2.])

    def test_invalid_trials_set_the_upper_bracket(self):
        def failure(theta, count):
            if theta[0] > .75:
                raise GaugeRegularityError("controlled invalid large trial")
        oracle = QuadraticOracle([[1.]], [.6], failure=failure)
        result = _minimize_exponential(oracle, [0.], LBFGSOptions(max_iterations=40, gradient_atol=1e-10,
                                                                  gradient_rtol=0, **WOLFE))
        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.theta, [.6], atol=1e-9)
        first, second = result.trials[:2]
        self.assertIsNone(first.loss)
        self.assertIn("GaugeRegularityError", first.reason)
        self.assertEqual(second.step_size, .5*first.step_size)
        self.assertGreater(result.failed_evaluations, 0)

    def test_no_decreasing_trial_fails_without_replacing_state(self):
        oracle = QuadraticOracle([[1000.]], [.03])
        result = _minimize_exponential(oracle, [0.], LBFGSOptions(max_iterations=3, max_line_search=1,
                                                                  **WOLFE))
        self.assertEqual(result.status, "line_search_failed")
        self.assertIn("Weak-Wolfe", result.message)
        self.assertEqual(result.iterations, 0)
        self.assertIs(result.evaluation, oracle.points[0])

    def test_budget_during_search_preserves_initial_point(self):
        oracle = QuadraticOracle([[1000.]], [.03])
        result = _minimize_exponential(oracle, [0.], LBFGSOptions(max_iterations=3, max_evaluations=2,
                                                                  max_line_search=10, **WOLFE))
        self.assertEqual(result.status, "max_evaluations")
        self.assertEqual(result.iterations, 0)
        self.assertIs(result.evaluation, oracle.points[0])


class NearZeroEigenvalueTests(unittest.TestCase):
    def test_rank_deficient_factor_is_reported_from_every_container(self):
        h, _, chol = example_tensors(n=2, r=2, seed=37)
        x = np.array([.6, .8])
        chol[0] = .7*np.outer(x, x)
        eri = np.einsum("gpr,gqs->prqs", chol, chol)
        problem = prepare_problem(h, eri, chol, nelec=(1, 1))
        gauge = build_gauge_state(problem.data)
        found = near_zero_eigenvalues(gauge)
        self.assertEqual([(g, a) for g, a, _ in found], [(0, 0)])
        self.assertLess(abs(found[0][2]), 1e-12)
        self.assertIsInstance(found[0][1], int)
        decomposition = build_fixed_decomposition(problem, gauge, .05, .05)
        self.assertEqual(near_zero_eigenvalues(decomposition), found)
        self.assertEqual(near_zero_eigenvalues(SimpleNamespace(decomposition=decomposition)), found)
        self.assertEqual(near_zero_eigenvalues(gauge, rtol=0.), ())
        with self.assertRaises(TypeError):
            near_zero_eigenvalues(object())
        with self.assertRaises(ValueError):
            near_zero_eigenvalues(gauge, rtol=-1.)


class WeakWolfePhysicalTests(unittest.TestCase):
    def test_two_orbital_nested_descent_with_weak_wolfe(self):
        h, _, chol = example_tensors(n=2, r=2, seed=37)
        chol[0] += .8*np.eye(2)
        eri = np.einsum("gpr,gqs->prqs", chol, chol)
        problem = prepare_problem(h, eri, chol, nelec=(1, 1))
        engine = GaugeEvaluator(problem, tau1=.2, tau2=.04)
        armijo = optimize_gauge(engine, options=OuterOptions(max_iterations=4))
        wolfe = optimize_gauge(engine, options=OuterOptions(max_iterations=4, line_search="weak_wolfe"))
        self.assertLess(wolfe.loss, .9*wolfe.initial_loss)
        self.assertEqual(wolfe.initial_loss, armijo.initial_loss)
        self.assertTrue(all(row.acceptance in ("weak_wolfe", "armijo_only", "expansion_capped")
                            for row in wolfe.history[1:]))
        self.assertTrue(all(row.acceptance == "armijo" for row in armijo.history[1:]))
        self.assertEqual(sum(trial.accepted for trial in wolfe.trials), wolfe.iterations)
        self.assertIsNotNone(wolfe.validation_report)
        check_hamiltonian(wolfe.decomposition, h, eri, (1, 1))


if __name__ == "__main__":
    unittest.main()
