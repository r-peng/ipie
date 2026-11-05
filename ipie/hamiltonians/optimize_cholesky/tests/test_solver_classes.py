"""Solver boundaries, lifecycle reuse, and parity with the step-6 public API."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ipie.hamiltonians.optimize_cholesky import (
    ExponentialGaugeObjective, GDMOptimizer, GaugeEvaluator, LBFGSOptimizer,
    LBFGSOptions, OptimizerIteration, OptimizerOptions, OptimizerStep,
    OuterOptimizer, OuterOptions, optimize_gauge, prepare_problem,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_outer_controlled import QuadraticOracle


class MatrixOracle:
    """Analytic matrix quadratic with no theta or coordinate reference."""

    def __init__(self):
        self.target = np.array([[1., .2], [-.3, .7]])
        self.evaluations = self.validations = 0

    def evaluate(self, matrix):
        self.evaluations += 1
        matrix = np.array(matrix, dtype=float, copy=True)
        gradient = matrix-self.target
        matrix.setflags(write=False)
        gradient.setflags(write=False)
        inner = SimpleNamespace(sweeps=0, projected_gradient_norm=0.)
        return SimpleNamespace(
            matrix=matrix, gradient=gradient, loss=float(.5*np.sum(gradient**2)),
            gradient_norm=float(np.linalg.norm(gradient)),
            tangent_gradient_norm=float(np.linalg.norm(gradient)),
            gauge_gradient=SimpleNamespace(inner_result=inner),
        )

    def validate(self, point):
        self.validations += 1
        return bool(np.isfinite(point.matrix).all())


@dataclass(frozen=True)
class MatrixResult:
    evaluation: object
    summary: object
    validation_report: object = None


class MatrixOptimizer(OuterOptimizer):
    """Test-only child with its own point geometry and result type."""

    def _initialize(self, initial):
        if initial is None:
            initial = np.zeros_like(self.objective.target)
        return self.objective.evaluate(initial), {"steps": 0}

    def _step(self, point, run, iteration):
        run.solver_state["steps"] += 1
        candidate = self.objective.evaluate(point.matrix-point.gradient)
        return OptimizerStep(evaluation=candidate, step_size=1.,
                             step_norm=float(np.linalg.norm(point.gradient)))

    def _build_result(self, point, summary):
        return MatrixResult(point, summary)


class RejectingMatrixOptimizer(MatrixOptimizer):
    def _step(self, point, run, iteration):
        return OptimizerStep(status="test_rejected", message="No candidate accepted")


def physical_engine():
    h = np.array([[.2, .13], [.13, -.1]])
    chol = np.array([np.diag([1.3, .7]), np.diag([.7, -.5])])
    eri = np.einsum("gpr,gqs->prqs", chol, chol)
    return GaugeEvaluator(prepare_problem(h, eri, chol, nelec=(1, 1)),
                          tau1=.2, tau2=.04)


class SolverClassTests(unittest.TestCase):
    def test_parent_is_abstract_and_gdm_is_explicit_before_evaluation(self):
        objective = MatrixOracle()
        with self.assertRaises(TypeError):
            OuterOptimizer(objective)
        with self.assertRaisesRegex(NotImplementedError, "GDMOptimizer is not implemented"):
            GDMOptimizer(objective).optimize(np.eye(2))
        self.assertEqual(objective.evaluations, 0)
        self.assertEqual(objective.validations, 0)

    def test_parent_runs_a_matrix_child_without_exponential_attributes(self):
        objective = MatrixOracle()
        records = []
        result = MatrixOptimizer(objective).optimize(callback=records.append)
        self.assertFalse(hasattr(result.evaluation, "theta"))
        self.assertFalse(hasattr(objective, "reference"))
        self.assertTrue(result.summary.converged)
        np.testing.assert_array_equal(result.evaluation.matrix, objective.target)
        self.assertEqual(result.summary.evaluations, 2)
        self.assertEqual(result.summary.failed_evaluations, 0)
        self.assertEqual(tuple(records), result.summary.history)
        self.assertTrue(all(type(row) is OptimizerIteration for row in records))
        self.assertTrue(result.validation_report)
        self.assertEqual(objective.validations, 1)

    def test_common_budget_rejection_and_callback_keep_accepted_matrix(self):
        for solver_type, options, status in (
            (MatrixOptimizer, OptimizerOptions(max_evaluations=1), "max_evaluations"),
            (RejectingMatrixOptimizer, OptimizerOptions(), "test_rejected"),
        ):
            with self.subTest(status=status):
                objective = MatrixOracle()
                result = solver_type(objective, options=options).optimize()
                self.assertEqual(result.summary.status, status)
                self.assertFalse(result.summary.converged)
                self.assertEqual(result.summary.evaluations, 1)
                np.testing.assert_array_equal(result.evaluation.matrix, np.zeros((2, 2)))
        objective = MatrixOracle()
        def stop(record):
            if record.iteration == 1:
                raise StopIteration
        result = MatrixOptimizer(objective).optimize(callback=stop)
        self.assertEqual(result.summary.status, "callback_stopped")
        np.testing.assert_array_equal(result.evaluation.matrix, objective.target)
        self.assertEqual(objective.validations, 1)

    def test_common_and_lbfgs_options_are_separate_and_legacy_layout_survives(self):
        common = OptimizerOptions(max_iterations=3, validate_final=False)
        self.assertFalse(hasattr(common, "memory"))
        objective = QuadraticOracle([[1.]], [2.])
        promoted = LBFGSOptimizer(objective, options=common).options
        self.assertIsInstance(promoted, LBFGSOptions)
        self.assertEqual(promoted.max_iterations, 3)
        self.assertEqual(promoted.memory, 10)
        # Original OuterOptions third positional argument was memory.
        legacy = OuterOptions(4, 80, 3, validate_final=False)
        converted = LBFGSOptimizer(objective, options=legacy).options
        self.assertEqual(asdict(converted), asdict(legacy))
        explicit = LBFGSOptions(memory=7, validate_final=False)
        self.assertIs(LBFGSOptimizer(objective, options=explicit).options, explicit)
        with self.assertRaises(TypeError):
            LBFGSOptimizer(objective, options={})
        for constructor, kwargs in (
            (OptimizerOptions, {"max_iterations": -1}),
            (LBFGSOptions, {"max_evaluations": 0}),
            (LBFGSOptions, {"memory": 0}),
            (LBFGSOptions, {"backtrack": 1.}),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                constructor(**kwargs)

    def test_direct_lbfgs_matches_public_physical_wrapper_and_validates_once(self):
        engine = physical_engine()
        options = LBFGSOptions(max_iterations=10, max_step_norm=.5)
        objective = ExponentialGaugeObjective(engine)
        with patch.object(objective, "validate", wraps=objective.validate) as validate:
            direct = LBFGSOptimizer(objective, options=options).optimize()
        self.assertEqual(validate.call_count, 1)
        self.assertIs(validate.call_args.args[0], direct.evaluation)
        wrapped = optimize_gauge(engine, options=OuterOptions(**asdict(options)))
        self.assertTrue(direct.converged)
        self.assertGreater(direct.iterations, 0)
        np.testing.assert_array_equal(direct.theta, wrapped.theta)
        self.assertEqual(direct.loss, wrapped.loss)
        self.assertEqual(direct.history, wrapped.history)
        self.assertEqual(direct.trials, wrapped.trials)
        self.assertEqual(direct.evaluations, wrapped.evaluations)
        self.assertTrue(direct.validation_report.passed)

    def test_same_solver_starts_fresh_memory_and_per_run_budget(self):
        objective = QuadraticOracle([[4., 1.], [1., 3.]], [.3, -.4])
        solver = LBFGSOptimizer(objective, options=LBFGSOptions(
            max_iterations=2, max_evaluations=4, max_step_norm=.5,
            gradient_atol=0., gradient_rtol=0., validate_final=False))
        first = solver.optimize([1.2, .8])
        second = solver.optimize([1.2, .8])
        self.assertTrue(any(row.memory_size for row in first.history[1:]))
        self.assertEqual(second.history[0].memory_size, 0)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.trials, second.trials)
        self.assertEqual(first.evaluations, second.evaluations)
        self.assertGreater(second.evaluations, 1)
        self.assertLessEqual(second.evaluations, 4)
        self.assertEqual(objective.evaluations, first.evaluations+second.evaluations)
        self.assertEqual(objective.inner_sweeps, first.inner_sweeps+second.inner_sweeps)

    def test_reused_objective_cache_is_counted_per_run(self):
        objective = ExponentialGaugeObjective(physical_engine())
        point = objective.evaluate([0.])
        solver = LBFGSOptimizer(objective, options=LBFGSOptions(
            max_iterations=0, max_evaluations=1, validate_final=False))
        first = solver.optimize([0.])
        second = solver.optimize([0.])
        for result in (first, second):
            self.assertIs(result.evaluation, point)
            self.assertEqual(result.evaluations, 0)
            self.assertEqual(result.cache_hits, 1)
            self.assertEqual(result.inner_sweeps, 0)
        self.assertEqual(objective.evaluations, 1)
        self.assertEqual(objective.cache_hits, 2)

    def test_validation_is_optional_but_required_when_enabled(self):
        objective = QuadraticOracle([[1.]], [0.])
        with self.assertRaisesRegex(TypeError, "objective.validate"):
            LBFGSOptimizer(objective)
        result = LBFGSOptimizer(objective, options=LBFGSOptions(
            validate_final=False)).optimize([0.])
        self.assertTrue(result.converged)
        self.assertIsNone(result.validation_report)


if __name__ == "__main__":
    unittest.main()
