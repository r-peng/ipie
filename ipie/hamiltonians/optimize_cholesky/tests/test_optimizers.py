"""Outer solvers: controlled oracles, physical systems, and generic lifecycle."""

from types import SimpleNamespace
import unittest
import numpy as np
import math
from unittest.mock import patch
from ipie.hamiltonians.optimize_cholesky import (
    ExponentialEvaluation,
    GaugeFeasibilityError,
    GaugeRegularityError,
    InnerConvergenceError,
    OuterOptions,
    ExponentialGaugeObjective,
    GaugeEvaluator,
    InnerOptions,
    exponential_gauge,
    exponential_gauge_vjp,
    optimize_gauge,
    prepare_problem,
    raw_loss,
    GDMOptimizer,
    LBFGSOptimizer,
    LBFGSOptions,
    OptimizerIteration,
    OptimizerOptions,
    OptimizerStep,
    OuterOptimizer,
)
from ipie.hamiltonians.optimize_cholesky.optimizers import _minimize_exponential
from dataclasses import FrozenInstanceError, asdict, dataclass
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_hamiltonian,
    example_tensors,
    make_decomposition,
)


# test_outer_controlled

def readonly(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


class QuadraticOracle:
    """Exact f(theta)=.5*(theta-target).T@A@(theta-target), with call records.

    failure(theta,call_number) can raise an explicitly requested exception.
    Each successful call reports two dummy inner sweeps; this makes driver
    counters auditable without replacing any scientific numerical operation.
    """

    def __init__(self, matrix, target, *, failure=None, tangent_norm=None):
        self.matrix = np.atleast_2d(np.array(matrix, dtype=float))
        self.target = np.atleast_1d(np.array(target, dtype=float))
        self.reference = readonly(np.eye(max(1, self.target.size)))
        self.failure = failure
        self.tangent_norm = tangent_norm
        self.evaluations = 0
        self.failed_evaluations = 0
        self.cache_hits = 0
        self.inner_sweeps = 0
        self.visited = []
        self.points = []

    def evaluate(self, theta):
        theta = np.asarray(theta, dtype=float)
        self.evaluations += 1
        self.visited.append(theta.copy())
        if self.failure is not None:
            try:
                self.failure(theta, self.evaluations)
            except (GaugeFeasibilityError, GaugeRegularityError,
                    InnerConvergenceError, FloatingPointError, np.linalg.LinAlgError):
                self.failed_evaluations += 1
                raise
        delta = theta-self.target
        value = float(.5*delta@self.matrix@delta)
        gradient = self.matrix@delta
        tangent = gradient.reshape(1, -1)
        if self.tangent_norm is not None:
            tangent = np.array([[self.tangent_norm]])
        inner = SimpleNamespace(converged=True, sweeps=2,
                                projected_gradient_norm=0.)
        snapshot = SimpleNamespace(theta=readonly(theta))
        derivative = SimpleNamespace(loss=value,
                                     tangent_gradient=readonly(tangent),
                                     inner_result=inner,
                                     decomposition=snapshot)
        point = ExponentialEvaluation(readonly(theta), readonly(gradient), derivative)
        self.points.append(point)
        self.inner_sweeps += inner.sweeps
        return point


class ControlledOuterDriverTests(unittest.TestCase):
    def test_quadratic_convergence_armijo_and_curvature_memory(self):
        oracle = QuadraticOracle([[4., 1.], [1., 3.]], [.3, -.4])
        options = OuterOptions(max_iterations=80, memory=4, max_step_norm=.75,
                               gradient_atol=1e-9, gradient_rtol=0,
                               validate_final=False)
        result = _minimize_exponential(oracle, [1.2, .8], options)
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.theta, oracle.target, atol=1e-8, rtol=0)
        self.assertLessEqual(result.gradient_norm, result.gradient_threshold)
        self.assertLessEqual(result.tangent_gradient_norm, result.tangent_gradient_threshold)
        self.assertLess(result.loss, result.initial_loss)
        self.assertTrue(np.all(np.diff([row.loss for row in result.history]) <= 0))
        self.assertTrue(any(row.curvature_update for row in result.history[1:]))
        self.assertTrue(all(row.memory_size <= options.memory for row in result.history))
        previous = oracle.points[0]
        accepted_count = 0
        # Every trial here is numerically valid; points[1:] follows trial order.
        for trial, point in zip(result.trials, oracle.points[1:]):
            if trial.accepted:
                displacement = point.theta-previous.theta
                armijo_rhs = previous.loss+options.armijo*np.dot(previous.gradient_theta,
                                                               displacement)
                self.assertLessEqual(point.loss, armijo_rhs+1e-14)
                previous = point
                accepted_count += 1
                self.assertEqual(result.history[accepted_count].loss, point.loss)
        self.assertEqual(accepted_count, result.iterations)
        self.assertIs(result.evaluation, previous)
        self.assertEqual(result.evaluations, 1+len(result.trials))
        self.assertEqual(result.failed_evaluations, 0)
        self.assertEqual(result.inner_sweeps, 2*result.evaluations)
        self.assertFalse(result.theta.flags.writeable)
        self.assertFalse(result.gradient_theta.flags.writeable)

    def test_zero_iteration_and_evaluation_budgets_keep_valid_initial_state(self):
        for options, status in (
            (OuterOptions(max_iterations=0), "max_iterations"),
            (OuterOptions(max_iterations=10, max_evaluations=1), "max_evaluations"),
        ):
            with self.subTest(status=status):
                oracle = QuadraticOracle([[1.]], [3.])
                result = _minimize_exponential(oracle, [0.], options)
                self.assertFalse(result.converged)
                self.assertEqual(result.status, status)
                self.assertEqual(result.iterations, 0)
                self.assertEqual(result.evaluations, 1)
                self.assertEqual(len(result.trials), 0)
                self.assertIs(result.evaluation, oracle.points[0])
                np.testing.assert_array_equal(result.theta, [0.])

    def test_evaluation_budget_after_an_accepted_step_returns_that_step(self):
        oracle = QuadraticOracle([[1.]], [3.])
        options = OuterOptions(max_iterations=10, max_evaluations=2,
                               max_step_norm=1., gradient_atol=1e-12,
                               gradient_rtol=0)
        result = _minimize_exponential(oracle, [0.], options)
        self.assertFalse(result.converged)
        self.assertEqual(result.status, "max_evaluations")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.evaluations, 2)
        self.assertIs(result.evaluation, oracle.points[-1])
        self.assertLess(result.loss, result.initial_loss)
        self.assertEqual(result.loss, result.history[-1].loss)

    def test_armijo_rejections_and_failed_line_search_do_not_replace_state(self):
        # The maximum allowed first displacement overshoots this sharp minimum.
        oracle = QuadraticOracle([[1000.]], [.03])
        options = OuterOptions(max_iterations=3, max_line_search=1,
                               max_step_norm=1., gradient_atol=1e-12,
                               gradient_rtol=0)
        result = _minimize_exponential(oracle, [0.], options)
        self.assertEqual(result.status, "line_search_failed")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(len(result.trials), 1)
        self.assertFalse(result.trials[0].accepted)
        self.assertEqual(result.trials[0].reason, "insufficient_decrease")
        self.assertGreater(oracle.points[-1].loss, oracle.points[0].loss)
        self.assertIs(result.evaluation, oracle.points[0])
        self.assertIsNot(result.evaluation, oracle.points[-1])
        np.testing.assert_array_equal(result.decomposition.theta, [0.])

    def test_budget_during_rejected_line_search_preserves_initial_point(self):
        oracle = QuadraticOracle([[1000.]], [.03])
        options = OuterOptions(max_iterations=3, max_evaluations=2,
                               max_line_search=10, max_step_norm=1.)
        result = _minimize_exponential(oracle, [0.], options)
        self.assertEqual(result.status, "max_evaluations")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.evaluations, 2)
        self.assertIs(result.evaluation, oracle.points[0])
        self.assertTrue(all(not trial.accepted for trial in result.trials))

    def test_recoverable_large_trial_failures_can_backtrack_and_converge(self):
        exceptions = (GaugeFeasibilityError, GaugeRegularityError,
                      InnerConvergenceError, FloatingPointError, np.linalg.LinAlgError)
        for exception in exceptions:
            def failure(theta, count):
                if theta[0] > .75:
                    raise exception("controlled invalid large trial")
            with self.subTest(exception=exception.__name__):
                oracle = QuadraticOracle([[10.]], [.3], failure=failure)
                result = _minimize_exponential(
                    oracle, [0.], OuterOptions(max_iterations=25,
                                              gradient_atol=1e-10, gradient_rtol=0))
                self.assertTrue(result.converged)
                self.assertGreater(result.failed_evaluations, 0)
                np.testing.assert_allclose(result.theta, [.3], atol=1e-9, rtol=0)
                failed = [trial for trial in result.trials if trial.loss is None]
                self.assertTrue(failed)
                self.assertTrue(all(not trial.accepted for trial in failed))
                self.assertTrue(all(exception.__name__ in trial.reason for trial in failed))

    def test_failures_after_one_accepted_step_keep_the_accepted_snapshot(self):
        def failure(theta, count):
            if count > 2:
                raise InnerConvergenceError("controlled later failure")
        oracle = QuadraticOracle([[1.]], [3.], failure=failure)
        result = _minimize_exponential(
            oracle, [0.], OuterOptions(max_iterations=4, max_line_search=3,
                                      gradient_atol=1e-12, gradient_rtol=0))
        self.assertEqual(result.status, "line_search_failed")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.failed_evaluations, 3)
        self.assertEqual(result.evaluations, 5)
        self.assertIs(result.evaluation, oracle.points[-1])
        self.assertEqual(result.loss, result.history[-1].loss)
        self.assertFalse(np.array_equal(result.theta, oracle.visited[-1]))
        self.assertEqual(result.partition_result.sweeps, 2)
        self.assertEqual(result.inner_sweeps, 4)

    def test_callback_can_stop_at_initial_or_accepted_state(self):
        for stop_iteration in (0, 1):
            records = []
            def callback(record):
                records.append(record)
                if record.iteration == stop_iteration:
                    raise StopIteration
            with self.subTest(stop_iteration=stop_iteration):
                oracle = QuadraticOracle([[1.]], [3.])
                result = _minimize_exponential(oracle, [0.], OuterOptions(), callback)
                self.assertEqual(result.status, "callback_stopped")
                self.assertFalse(result.converged)
                self.assertEqual(result.iterations, stop_iteration)
                self.assertEqual([row.iteration for row in records],
                                 list(range(stop_iteration+1)))
                self.assertEqual(result.loss, records[-1].loss)
                self.assertIs(result.evaluation, oracle.points[-1])

    def test_initial_failure_and_programming_errors_propagate(self):
        initial_error = GaugeRegularityError("controlled invalid initial point")
        def initial_failure(theta, count):
            raise initial_error
        oracle = QuadraticOracle([[1.]], [3.], failure=initial_failure)
        with self.assertRaises(GaugeRegularityError) as caught:
            _minimize_exponential(oracle, [0.], OuterOptions())
        self.assertIs(caught.exception, initial_error)
        self.assertEqual(oracle.evaluations, 1)
        for exception in (NotImplementedError, TypeError, AttributeError, ValueError, RuntimeError):
            error = exception("controlled programming/configuration error")
            def failure(theta, count):
                if count == 2:
                    raise error
            with self.subTest(exception=exception.__name__):
                oracle = QuadraticOracle([[1.]], [3.], failure=failure)
                with self.assertRaises(exception) as caught:
                    _minimize_exponential(oracle, [0.], OuterOptions())
                self.assertIs(caught.exception, error)
                self.assertEqual(oracle.evaluations, 2)
        oracle = QuadraticOracle([[1.]], [3.])
        with self.assertRaises(TypeError):
            _minimize_exponential(oracle, [0.], OuterOptions(), callback=1)
        self.assertEqual(oracle.evaluations, 0)

    def test_negative_curvature_is_not_retained(self):
        oracle = QuadraticOracle([[-1.]], [0.])
        result = _minimize_exponential(
            oracle, [.2], OuterOptions(max_iterations=3, gradient_atol=0,
                                      gradient_rtol=0, max_step_norm=.3))
        self.assertEqual(result.status, "max_iterations")
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 3)
        self.assertTrue(all(not row.curvature_update for row in result.history))
        self.assertTrue(all(row.memory_size == 0 for row in result.history))
        self.assertTrue(np.all(np.diff([row.loss for row in result.history]) < 0))

    def test_bad_quasi_newton_direction_falls_back_to_descent(self):
        oracle = QuadraticOracle([[1.]], [3.])
        # This replaces only the optimization helper with a deliberately bad
        # direction. No Hamiltonian, linear-algebra or derivative kernel is mocked.
        with patch("ipie.hamiltonians.optimize_cholesky.optimizers._lbfgs_direction",
                   side_effect=lambda gradient, pairs: gradient.copy()):
            result = _minimize_exponential(
                oracle, [0.], OuterOptions(max_iterations=10, gradient_atol=1e-9,
                                          gradient_rtol=0))
        self.assertTrue(result.converged)
        self.assertLess(result.loss, result.initial_loss)
        np.testing.assert_allclose(result.theta, [3.], atol=1e-8, rtol=0)

    def test_coordinate_gradient_alone_does_not_certify_stationarity(self):
        oracle = QuadraticOracle([[1.]], [0.], tangent_norm=1.)
        result = _minimize_exponential(
            oracle, [0.], OuterOptions(gradient_atol=1e-9, gradient_rtol=0))
        self.assertEqual(result.gradient_norm, 0.)
        self.assertEqual(result.tangent_gradient_norm, 1.)
        self.assertEqual(result.status, "coordinate_stalled")
        self.assertFalse(result.converged)
        self.assertEqual(result.evaluations, 1)

    def test_zero_dimensional_coordinate_problem_is_evaluated_once(self):
        oracle = QuadraticOracle(np.empty((0, 0)), np.empty(0))
        result = _minimize_exponential(oracle, np.empty(0), OuterOptions())
        self.assertTrue(result.converged)
        self.assertEqual(result.status, "converged")
        self.assertEqual(result.evaluations, 1)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.theta.shape, (0,))


# test_outer

def engine_fixture(nelec=(2,0),partition="balanced_log"):
    problem,_ = make_decomposition(nelec,partition=partition)
    return problem,GaugeEvaluator(problem,tau1=.2,tau2=.04,partition=partition)


class NestedOracleTests(unittest.TestCase):
    def test_joint_cache_copies_seed_and_keeps_reference_and_configuration_fixed(self):
        problem,engine = engine_fixture()
        reference=np.diag([-1.,1.])@exponential_gauge([.1],2)
        seed=np.linspace(-.1,.1,len(engine.sources))
        saved=seed.copy()
        options=InnerOptions(gradient_atol=1e-12,gradient_rtol=0)
        objective=ExponentialGaugeObjective(engine,reference=reference,
                                            inner_options=options,inner_t0=seed)
        seed[:]=10
        np.testing.assert_array_equal(objective.inner_t0,saved)
        for name,value in (("inner_options",InnerOptions()),("reference",np.eye(2)),
                           ("inner_t0",None),("evaluator",engine),("nparams",99)):
            with self.subTest(name=name),self.assertRaises(AttributeError):
                setattr(objective,name,value)
        with self.assertRaises(FrozenInstanceError):
            options.max_sweeps=1
        theta=np.array([.03])
        point=objective.evaluate(theta)
        np.testing.assert_allclose(point.O,exponential_gauge(theta,2,reference),atol=1e-15)
        expected=exponential_gauge_vjp(theta,2,point.gauge_gradient.gradient_O,reference)
        np.testing.assert_array_equal(point.gradient_theta,expected)
        self.assertAlmostEqual(point.loss,raw_loss(point.decomposition,method="dense"),places=14)
        theta[0]=1.
        self.assertEqual(point.theta[0],.03)
        value,gradient=objective.value_and_grad([.03])
        self.assertEqual(objective.evaluations,1)
        self.assertEqual(objective.cache_hits,1)
        self.assertEqual(value,point.loss)
        gradient[:]=123
        np.testing.assert_array_equal(point.gradient_theta,expected)
        self.assertIs(objective.evaluate([.03]),point)
        self.assertFalse(objective.reference.flags.writeable)
        self.assertFalse(point.theta.flags.writeable)
        self.assertFalse(point.gradient_theta.flags.writeable)

    def test_relaxed_theta_gradient_rechecks_partitions_and_moving_bounds(self):
        _,engine=engine_fixture()
        reference=exponential_gauge([.37],2)
        state=engine.decomposition(reference)
        ell=state.two_body.gauge.eigenvalues
        rho=1.025*max(np.sqrt(abs(ell[s.bix,s.modes[0]]*ell[s.bix,s.modes[1]])*.04)
                       for s in engine.sources)
        objective=ExponentialGaugeObjective(
            engine,reference=reference,
            inner_options=InnerOptions(amplitude_limit=rho,gradient_atol=1e-12,gradient_rtol=0))
        calls=[]
        original=engine.relaxed
        def record(O,**kwargs):
            result=original(O,**kwargs)
            calls.append((O.copy(),result.decomposition.two_body.t.copy()))
            return result
        with patch.object(engine,"relaxed",side_effect=record):
            center=objective.evaluate([0.])
            plus=objective.evaluate([1e-6])
            minus=objective.evaluate([-1e-6])
        self.assertEqual(len(calls),3)
        self.assertTrue(all(not np.array_equal(calls[0][0],item[0]) for item in calls[1:]))
        self.assertGreater(np.max(np.abs(calls[1][1]-calls[2][1])),1e-9)
        finite=(plus.loss-minus.loss)/2e-6
        np.testing.assert_allclose(center.gradient_theta[0],finite,atol=2e-10,rtol=3e-7)
        wrong=exponential_gauge_vjp([0.],2,center.gauge_gradient.fixed_gradient_O,reference)[0]
        self.assertGreater(abs(wrong-finite),1e-6)

    def test_post_inner_regularity_failure_retains_inner_work_diagnostics(self):
        _,engine=engine_fixture()
        objective=ExponentialGaugeObjective(engine)
        with patch("ipie.hamiltonians.optimize_cholesky.gradients.normalized_partition_gradient",
                   side_effect=GaugeRegularityError("injected post-inner failure")):
            with self.assertRaises(GaugeRegularityError) as context:
                objective.evaluate([0.])
        inner=context.exception.inner_result
        self.assertTrue(inner.converged)
        self.assertGreater(inner.sweeps,0)
        self.assertEqual(objective.inner_sweeps,inner.sweeps)
        self.assertEqual(objective.failed_evaluations,1)
        self.assertIsNone(objective._cached)

    def test_options_and_failed_initial_evaluations_are_not_hidden(self):
        _,engine=engine_fixture()
        for kwargs in ({"max_iterations":-1},{"max_evaluations":0},{"memory":0},
                       {"backtrack":1.},{"armijo":0.},{"gradient_atol":np.nan},
                       {"initial_step":0.},{"curvature_rtol":1.},{"validate_final":"yes"}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                OuterOptions(**kwargs)
        for kwargs in ({"max_sweeps":True},{"gradient_atol":-1},{"amplitude_limit":1.},
                       {"max_abs_t":0.}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                InnerOptions(**kwargs)
        with self.assertRaises(TypeError):
            optimize_gauge(engine,options={})
        with self.assertRaises(TypeError):
            optimize_gauge(engine,inner_options={})
        with self.assertRaises(ValueError):
            optimize_gauge(engine,theta0=[0.,0.])
        with self.assertRaises(ValueError):
            optimize_gauge(engine,reference=np.zeros((2,2)))
        with self.assertRaises(InnerConvergenceError):
            optimize_gauge(engine,inner_options=InnerOptions(max_sweeps=0))
        with self.assertRaises(GaugeFeasibilityError):
            optimize_gauge(engine,inner_options=InnerOptions(amplitude_limit=.001))


class PhysicalOuterTests(unittest.TestCase):
    def test_nontrivial_physical_descent_reaches_a_smooth_stationary_region(self):
        # Commuting factors but a noncommuting h. A rotation from the mixed
        # definite/indefinite region enters a smooth constant-minimum region.
        h=np.array([[.2,.13],[.13,-.1]])
        chol=np.array([np.diag([1.3,.7]),np.diag([.7,-.5])])
        eri=np.einsum("gpr,gqs->prqs",chol,chol)
        problem=prepare_problem(h,eri,chol,nelec=(1,1))
        engine=GaugeEvaluator(problem,tau1=.2,tau2=.04)
        result=optimize_gauge(engine,options=OuterOptions(max_iterations=10,max_step_norm=.5))
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations,0)
        self.assertLess(result.loss,.9*result.initial_loss)
        self.assertLessEqual(result.gradient_norm,result.gradient_threshold)
        self.assertLessEqual(result.tangent_gradient_norm,result.tangent_gradient_threshold)
        self.assertTrue(result.validation_report.passed)
        check_hamiltonian(result.decomposition,h,eri,(1,1))

    def test_nested_descent_reconstruction_and_final_snapshot_all_policies_and_maps(self):
        for nelec in ((2,0),(2,1),(1,1)):
            for partition in ("balanced_log","anchored_log"):
                with self.subTest(nelec=nelec,partition=partition):
                    problem,engine=engine_fixture(nelec,partition)
                    callback_rows=[]
                    result=optimize_gauge(
                        engine,options=OuterOptions(max_iterations=3,max_step_norm=.3),
                        inner_options=InnerOptions(gradient_atol=1e-12,gradient_rtol=0),
                        callback=callback_rows.append)
                    self.assertLess(result.loss,result.initial_loss)
                    self.assertTrue(np.all(np.diff([row.loss for row in result.history])<=0))
                    self.assertEqual(tuple(callback_rows),result.history)
                    self.assertTrue(result.validation_report.passed)
                    self.assertTrue(result.partition_result.converged)
                    np.testing.assert_allclose(result.loss,raw_loss(result.decomposition,method="dense"),
                                               atol=3e-17,rtol=3e-11)
                    np.testing.assert_allclose(result.O,exponential_gauge(result.theta,2,result.reference),atol=1e-15)
                    expected=exponential_gauge_vjp(result.theta,2,result.evaluation.gauge_gradient.gradient_O,
                                                   result.reference)
                    np.testing.assert_array_equal(result.gradient_theta,expected)
                    check_hamiltonian(result.decomposition,problem.data.h1,problem.data.eri,nelec)
                    self.assertEqual(len(result.history),result.iterations+1)
                    self.assertFalse(result.theta.flags.writeable)
                    self.assertFalse(result.reference.flags.writeable)
                    if result.converged:
                        self.assertLessEqual(result.gradient_norm,result.gradient_threshold)
                        self.assertLessEqual(result.tangent_gradient_norm,result.tangent_gradient_threshold)

    def test_recoverable_trials_backtrack_and_preserve_the_frozen_seed(self):
        _,engine=engine_fixture()
        original=engine.relaxed
        calls=[]
        seed=np.linspace(-.15,.15,len(engine.sources))
        def reject_large(O,**kwargs):
            calls.append((O.copy(),kwargs["t0"].copy()))
            if np.linalg.norm(O-np.eye(2))>.11:
                raise InnerConvergenceError("injected large-step inner failure")
            return original(O,**kwargs)
        with patch.object(engine,"relaxed",side_effect=reject_large):
            result=optimize_gauge(engine,inner_t0=seed,
                                   options=OuterOptions(max_iterations=1,max_step_norm=.4))
        self.assertGreater(result.failed_evaluations,0)
        self.assertTrue(any("InnerConvergenceError" in row.reason for row in result.trials))
        self.assertEqual(result.iterations,1)
        self.assertLess(result.loss,result.initial_loss)
        self.assertLessEqual(np.linalg.norm(result.O-np.eye(2)),.11)
        for _,supplied_seed in calls:
            np.testing.assert_array_equal(supplied_seed,seed)

    def test_failed_line_search_returns_accepted_state_not_last_trial(self):
        _,engine=engine_fixture()
        original=engine.relaxed
        count=0
        def reject_all_trials(O,**kwargs):
            nonlocal count
            count+=1
            if count>1:
                raise GaugeRegularityError("injected rejected trial")
            return original(O,**kwargs)
        with patch.object(engine,"relaxed",side_effect=reject_all_trials):
            result=optimize_gauge(engine,options=OuterOptions(max_iterations=2,max_line_search=3))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"line_search_failed")
        self.assertEqual(result.iterations,0)
        self.assertEqual(result.evaluations,4)
        self.assertEqual(result.failed_evaluations,3)
        self.assertEqual(result.loss,result.initial_loss)
        np.testing.assert_array_equal(result.theta,[0.])
        np.testing.assert_array_equal(result.O,np.eye(2))
        self.assertTrue(result.validation_report.passed)

    def test_two_orbital_shortcut_has_zero_inner_sweeps_during_nontrivial_descent(self):
        h,_,chol=example_tensors(n=2,r=2,seed=37)
        chol[0]+=.8*np.eye(2)
        eri=np.einsum("gpr,gqs->prqs",chol,chol)
        problem=prepare_problem(h,eri,chol,nelec=(1,1))
        engine=GaugeEvaluator(problem,tau1=.2,tau2=.04)
        result=optimize_gauge(engine,options=OuterOptions(max_iterations=3))
        self.assertLess(result.loss,.9*result.initial_loss)
        self.assertTrue(all(row.inner_sweeps==0 for row in result.history))
        check_hamiltonian(result.decomposition,h,eri,(1,1))

    def test_physical_stationarity_is_checked_at_a_singular_exponential_chart(self):
        h,eri,chol=example_tensors(n=3,r=3,seed=29)
        problem=prepare_problem(h,eri,chol,nelec=(2,1))
        engine=GaugeEvaluator(problem,tau1=.2,tau2=.04)
        objective=ExponentialGaugeObjective(engine)
        center=objective.evaluate(np.zeros(3))
        axis=np.cross(center.gradient_theta,[1.,0.,0.])
        if np.linalg.norm(axis)<1e-14:
            axis=np.cross(center.gradient_theta,[0.,1.,0.])
        theta=2*math.pi*axis/np.linalg.norm(axis)
        result=optimize_gauge(engine,theta0=theta,
                               options=OuterOptions(max_iterations=3,gradient_atol=1e-9,gradient_rtol=0))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"coordinate_stalled")
        self.assertEqual(result.iterations,0)
        self.assertLess(result.gradient_norm,1e-9)
        self.assertGreater(result.tangent_gradient_norm,1e-6)
        np.testing.assert_array_equal(result.theta,theta)
        np.testing.assert_array_equal(result.reference,np.eye(3))

    def test_single_factor_has_no_continuous_gauge_parameters(self):
        h=np.array([[.2,.1],[.1,-.1]])
        chol=np.array([np.diag([-.3,.7])])
        eri=np.einsum("gpr,gqs->prqs",chol,chol)
        problem=prepare_problem(h,eri,chol,nelec=(1,1))
        engine=GaugeEvaluator(problem,tau1=.1,tau2=.04)
        result=optimize_gauge(engine,reference=[[-1.]])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations,0)
        self.assertEqual(result.evaluations,1)
        self.assertEqual(result.theta.size,0)
        np.testing.assert_array_equal(result.O,[[-1.]])
        self.assertGreater(result.loss,0.)


# test_solver_classes

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


# Public solver-selection entry point

import ipie.hamiltonians.optimize_cholesky as optimization

class SolverDispatchTests(unittest.TestCase):
    def test_default_and_explicit_lbfgs_are_identical(self):
        engine = physical_engine()
        options = optimization.LBFGSOptions(max_iterations=0, validate_final=False)
        default = optimization.optimize_gauge(engine, options=options)
        explicit = optimization.optimize_gauge(engine, solver="lbfgs", options=options)
        np.testing.assert_array_equal(default.theta, explicit.theta)
        self.assertEqual(default.loss, explicit.loss)
        self.assertEqual(default.status, explicit.status)
        self.assertEqual(default.evaluations, explicit.evaluations)
        self.assertEqual(default.history, explicit.history)
        self.assertEqual(default.trials, explicit.trials)

    def test_gdm_is_an_explicit_placeholder_without_evaluation(self):
        engine = physical_engine()
        with patch.object(optimization.ExponentialGaugeObjective, "evaluate",
                          side_effect=AssertionError("objective must not be evaluated")) as evaluate:
            with self.assertRaises(NotImplementedError):
                optimization.optimize_gauge(engine, solver="gdm")
        evaluate.assert_not_called()

    def test_bad_solver_is_rejected_before_engine_use(self):
        # None is deliberately not an evaluator: errors must identify the
        # solver argument before attempting any engine-specific work.
        with self.assertRaisesRegex(ValueError, "solver"):
            optimization.optimize_gauge(None, solver="unknown_solver")
        for value in (None, 12, ["lbfgs"]):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "solver"):
                optimization.optimize_gauge(None, solver=value)


if __name__ == "__main__":
    unittest.main()
