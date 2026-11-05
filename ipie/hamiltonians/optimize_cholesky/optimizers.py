"""Outer objective, solver-independent lifecycle, and explicit solver subclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace, asdict
import math
import numpy as np
from typing import Any, Tuple
from .models import (
    GaugeFeasibilityError,
    GaugeRegularityError,
    InnerConvergenceError,
    OptimizerIteration,
    OptimizerOptions,
    OuterTrial,
    ExponentialEvaluation,
    InnerOptions,
    LBFGSOptions,
    OuterIteration,
    OuterOptions,
    OuterOptimizationResult,
)
from .gradients import exponential_gauge_vjp, GaugeEvaluator
from .partitions import _partition_limits, _readonly
from .problem import exponential_gauge, _real_array, validate_gauge_state


# Solver-independent lifecycle and subclass contracts

RECOVERABLE_TRIAL_ERRORS = (
    GaugeFeasibilityError, GaugeRegularityError, InnerConvergenceError,
    FloatingPointError, np.linalg.LinAlgError,
)


@dataclass
class OptimizerRun:
    """Per-call opaque solver state plus counter offsets; never reused."""

    solver_state: Any
    counter_offsets: dict


@dataclass(frozen=True)
class OptimizerStep:
    """A child's accepted step or explicit failure, with scalar diagnostics.

    evaluation=None means that the parent must retain the previous point.
    diagnostics belongs to the child; the parent does not interpret its keys.
    """

    evaluation: Any = None
    status: str = ""
    message: str = ""
    trials: Tuple[OuterTrial, ...] = ()
    step_size: float = 0.0
    step_norm: float = 0.0
    line_search_trials: int = 0
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerSummary:
    """Common run metadata passed to the child's result builder."""

    initial_evaluation: Any
    gradient_threshold: float
    tangent_gradient_threshold: float
    history: tuple
    trials: tuple
    status: str
    message: str
    evaluations: int
    failed_evaluations: int
    cache_hits: int
    inner_sweeps: int

    @property
    def converged(self):
        return self.status == "converged"


class OuterOptimizer(ABC):
    """Common outer driver; concrete children implement exactly three hooks.

    Objective contract: evaluate(parameters) returns an immutable point with
    loss, gradient_norm, tangent_gradient_norm and gauge_gradient.inner_result.
    Objective.evaluations counts fresh work. Extra counters are optional.
    If validate_final=True, objective.validate(point) must return a report and
    the child's result must be a dataclass with a validation_report field.

    Coordinate geometry belongs to the point/objective and child. No packed
    angle vector or orthogonal-matrix update is assumed by this parent.
    """

    _COUNTERS = ("evaluations", "failed_evaluations", "cache_hits", "inner_sweeps")

    def __init__(self, objective, *, options=None):
        options = OptimizerOptions() if options is None else options
        if not isinstance(options, OptimizerOptions):
            raise TypeError("options must derive from OptimizerOptions")
        if not callable(getattr(objective, "evaluate", None)) or not hasattr(objective, "evaluations"):
            raise TypeError("objective must provide evaluate(parameters) and an evaluations counter")
        if options.validate_final and not callable(getattr(objective, "validate", None)):
            raise TypeError("validate_final=True requires objective.validate(point)")
        self._objective = objective
        self._options = options

    @property
    def objective(self):
        return self._objective

    @property
    def options(self):
        return self._options

    @abstractmethod
    def _initialize(self, initial):
        """Return (initial_point, fresh_solver_state); initial failures raise."""

    @abstractmethod
    def _step(self, point, run, iteration):
        """Return OptimizerStep; the child owns proposals and acceptance."""

    @abstractmethod
    def _build_result(self, point, summary):
        """Package the last accepted point and common metadata in a result."""

    def _counter_snapshot(self):
        return {name: int(getattr(self.objective, name, 0)) for name in self._COUNTERS}

    def _budget_exhausted(self, run):
        used = self.objective.evaluations-run.counter_offsets["evaluations"]
        return used >= self.options.max_evaluations

    def _iteration_record(self, point, index, *, previous=None, step=None):
        """Common scalar record; children may extend it with their own fields."""
        inner = point.gauge_gradient.inner_result
        return OptimizerIteration(
            index, point.loss, point.gradient_norm, point.tangent_gradient_norm,
            0. if step is None else step.step_size,
            0. if step is None else step.step_norm,
            0. if previous is None else previous.loss-point.loss,
            inner.sweeps, inner.projected_gradient_norm,
            0 if step is None else step.line_search_trials,
        )

    def _stationarity(self, point, threshold, tangent_threshold):
        if point.gradient_norm <= threshold:
            if point.tangent_gradient_norm <= tangent_threshold:
                return "converged", "Solver and physical tangent gradient tolerances met"
            return ("coordinate_stalled",
                    "Solver gradient met its tolerance but the physical tangent gradient did not")
        return None

    def optimize(self, initial=None, *, callback=None):
        """Run a fresh solver state; reuse objective caches but report per-run counters.

        The callback receives initial/accepted records. StopIteration requests
        a clean stop; other callback or programming exceptions propagate.
        Only a child's accepted point can replace the current result.
        """
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        offsets = self._counter_snapshot()
        point, state = self._initialize(initial)
        run = OptimizerRun(state, offsets)
        initial_point = point
        threshold = self.options.gradient_atol+self.options.gradient_rtol*point.gradient_norm
        tangent_threshold = (self.options.gradient_atol+
                             self.options.gradient_rtol*point.tangent_gradient_norm)
        if not math.isfinite(threshold+tangent_threshold):
            raise ValueError("Outer gradient stopping thresholds are not representable")
        history = [self._iteration_record(point, 0)]
        trials = []
        status, message = "max_iterations", "Outer iteration limit reached"

        def notify():
            row = history[-1]

            if self.options.verbosity >= 1:
                print(
                    f"[outer {row.iteration:3d}] "
                    f"loss={row.loss:.6e} "
                    f"grad={row.gradient_norm:.4e} "
                    f"grad_tol={threshold:.4e} "
                    f"tangent={row.tangent_gradient_norm:.4e} "
                    f"tangent_tol={tangent_threshold:.4e} "
                    f"step_norm={row.step_norm:.3e} "
                    f"inner_sweeps={row.inner_sweeps} "
                    f"inner_grad={row.inner_projected_gradient_norm:.3e}",
                    flush=True,
                )

            if callback is not None:
                callback(row)

        try:
            notify()
        except StopIteration:
            status, message = "callback_stopped", "Callback stopped at the accepted initial state"
        else:
            stopped = self._stationarity(point, threshold, tangent_threshold)
            if stopped is not None:
                status, message = stopped
            else:
                for iteration in range(1, self.options.max_iterations+1):
                    if self._budget_exhausted(run):
                        status, message = "max_evaluations", "Outer evaluation budget exhausted"
                        break
                    step = self._step(point, run, iteration)
                    if not isinstance(step, OptimizerStep):
                        raise TypeError("_step must return OptimizerStep")
                    trials.extend(step.trials)
                    if step.evaluation is None:
                        if not step.status or step.status == "converged":
                            raise ValueError("A rejected step must report a nonconverged termination status")
                        status, message = step.status, step.message
                        break
                    previous = point
                    point = step.evaluation
                    history.append(self._iteration_record(point, iteration, previous=previous, step=step))
                    try:
                        notify()
                    except StopIteration:
                        status, message = "callback_stopped", "Callback stopped at an accepted outer state"
                        break
                    stopped = self._stationarity(point, threshold, tangent_threshold)
                    if stopped is not None:
                        status, message = stopped
                        break
        current = self._counter_snapshot()
        counts = {name: current[name]-offsets[name] for name in self._COUNTERS}
        summary = OptimizerSummary(
            initial_point, float(threshold), float(tangent_threshold), tuple(history),
            tuple(trials), status, message, **counts,
        )
        result = self._build_result(point, summary)

        if self.options.validate_final:
            result = replace(
                result,
                validation_report=self.objective.validate(point),
            )

        if self.options.verbosity >= 1:
            print(
                f"[outer finished] status={status} "
                f"accepted_iterations={history[-1].iteration} "
                f"evaluations={counts['evaluations']} "
                f"failed_evaluations={counts['failed_evaluations']} "
                f"loss={point.loss:.17e}",
                flush=True,
            )
            print(f"[outer finished] {message}", flush=True)

        return result


# Fixed exponential objective and cached nested evaluations

_TRIAL_ERRORS = RECOVERABLE_TRIAL_ERRORS


class ExponentialGaugeObjective:
    """Reduced value/gradient oracle with one immutable coordinate reference.

    The inner configuration and supplied seed are frozen for a run. With no
    seed, every trial starts at its own interval midpoints. A rejected trial
    never becomes the seed for another evaluation. The one-entry cache holds
    theta, value, gradient and inner solution together; no unbounded snapshot
    history is retained. General inner solutions are tracked local solutions,
    not certified global minima.
    """

    def __init__(self,evaluator,*,reference=None,inner_options=None,inner_t0=None):
        if not isinstance(evaluator,GaugeEvaluator):
            raise TypeError("evaluator must be a GaugeEvaluator")
        self._evaluator = evaluator
        self._nchol = evaluator.problem.data.nchol
        self._nparams = self.nchol*(self.nchol-1)//2
        if inner_options is not None and not isinstance(inner_options,InnerOptions):
            raise TypeError("inner_options must be InnerOptions")
        self._inner_options = InnerOptions() if inner_options is None else inner_options
        _partition_limits(evaluator.amplitude_bound,self.inner_options.amplitude_limit,
                          self.inner_options.max_abs_t)
        self._reference = _readonly(exponential_gauge(np.zeros(self.nparams),self.nchol,reference))
        if inner_t0 is None:
            self._inner_t0 = None
        else:
            seed = _real_array(inner_t0,"inner_t0")
            if seed.ndim != 0 and seed.shape != (len(evaluator.sources),):
                raise ValueError("inner_t0 must be a scalar or have one entry per source")
            self._inner_t0 = _readonly(seed)
        self.evaluations = self.failed_evaluations = self.cache_hits = self.inner_sweeps = 0
        self._cached = None

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def nchol(self):
        return self._nchol

    @property
    def nparams(self):
        return self._nparams

    @property
    def inner_options(self):
        return self._inner_options

    @property
    def reference(self):
        return self._reference

    @property
    def inner_t0(self):
        return self._inner_t0

    def evaluate(self,theta):
        """Return one matching value/coordinate-gradient/inner snapshot."""
        theta = _real_array(theta,"theta")
        if theta.shape != (self.nparams,):
            raise ValueError(f"theta must be a length-{self.nparams} vector")
        if self._cached is not None and np.array_equal(theta,self._cached.theta):
            self.cache_hits += 1
            return self._cached
        self.evaluations += 1
        inner = None
        try:
            O = exponential_gauge(theta,self.nchol,self.reference)
            derivative = self.evaluator.relaxed(O,t0=self.inner_t0,**asdict(self.inner_options))
            inner = derivative.inner_result
            if inner is None or not inner.converged:
                raise InnerConvergenceError("Outer objective requires a converged inner snapshot",inner)
            gradient = exponential_gauge_vjp(theta,self.nchol,derivative.gradient_O,
                                              self.reference,backend=self.evaluator.backend)
            if (not np.isfinite(derivative.loss) or not np.isfinite(gradient).all()
                    or not np.isfinite(derivative.tangent_gradient).all()):
                raise FloatingPointError("Nonfinite outer value or coordinate gradient")
            point = ExponentialEvaluation(_readonly(theta),_readonly(gradient),derivative)
            if not math.isfinite(point.tangent_gradient_norm):
                raise FloatingPointError("Nonfinite physical tangent-gradient norm")
        except _TRIAL_ERRORS as exc:
            self.failed_evaluations += 1
            if inner is None:
                inner = getattr(exc,"inner_result",None)
            if inner is None and isinstance(exc,InnerConvergenceError):
                inner = exc.result
            if inner is not None:
                self.inner_sweeps += inner.sweeps
            raise
        self.inner_sweeps += inner.sweeps
        self._cached = point
        return point

    def value_and_grad(self,theta):
        """Joint callable for external optimizers, e.g. a SciPy jac=True call."""
        point = self.evaluate(theta)
        return point.loss,point.gradient_theta.copy()

    def validate(self,point):
        """Full final gauge/ERI check, called outside the optimizer loops."""
        return validate_gauge_state(
            self.evaluator.problem.data,point.decomposition.two_body.gauge,
            self.evaluator.problem.report.options)






# Explicit L-BFGS: directions, Armijo line search, curvature updates

def _coerce_lbfgs_options(options):
    if options is None:
        return LBFGSOptions()
    if isinstance(options, LBFGSOptions):
        return options
    if isinstance(options, (OptimizerOptions, OuterOptions)):
        return LBFGSOptions(**asdict(options))
    raise TypeError("options must be LBFGSOptions, OptimizerOptions or legacy OuterOptions")


def _lbfgs_direction(gradient, pairs):
    """Standard two-loop inverse-Hessian product for positive-curvature pairs."""
    q = gradient.copy()
    alphas = []
    for s, y, rho in reversed(pairs):
        alpha = rho*np.dot(s, q)
        alphas.append(alpha)
        q -= alpha*y
    scale = 1.0
    if pairs:
        s, y, _ = pairs[-1]
        scale = np.dot(s, y)/np.dot(y, y)
    value = scale*q
    for (s, y, rho), alpha in zip(pairs, reversed(alphas)):
        value += s*(alpha-rho*np.dot(y, value))
    return -value


class LBFGSOptimizer(OuterOptimizer):
    """L-BFGS on an ExponentialGaugeObjective, retaining the step-6 behavior.

    Construct the objective separately to keep coordinate geometry out of the
    parent. Each optimize(theta0) call creates fresh curvature memory. Common
    lifecycle, callbacks, stopping and validation are inherited from the parent.
    """

    def __init__(self, objective, *, options=None):
        super().__init__(objective, options=_coerce_lbfgs_options(options))

    def _initialize(self, initial):
        initial = np.zeros(self.objective.nparams) if initial is None else initial
        return self.objective.evaluate(initial), []

    def _step(self, point, run, iteration):
        options = self.options
        pairs = run.solver_state
        gradient = point.gradient_theta
        # 1. Two-loop direction; discard bad history and fall back to descent.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            direction = _lbfgs_direction(gradient, pairs)
        slope = float(np.dot(gradient, direction))
        if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
            pairs.clear()
            direction = -gradient.copy()
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm == 0:
            return OptimizerStep(status="stalled", message="No finite nonzero descent direction is representable")
        cap = options.max_step_norm/options.initial_step
        # 2. Normalize the history-free direction and cap the trial length.
        if not pairs:
            direction /= norm
            norm = 1.0
        if norm > cap:
            direction *= cap/norm
        slope = float(np.dot(gradient, direction))
        if not np.isfinite(slope) or slope >= 0:
            return OptimizerStep(status="stalled", message="A negative coordinate directional derivative is not representable")
        alpha = options.initial_step
        candidate = None
        exhausted = False
        trials = []
        # 3. Armijo backtracking; every new gauge re-relaxes its partitions.
        for attempt in range(1, options.max_line_search+1):
            if alpha < options.min_step:
                break
            if self._budget_exhausted(run):
                exhausted = True
                break
            theta = point.theta+alpha*direction
            if np.array_equal(theta, point.theta):
                break
            try:
                trial = self.objective.evaluate(theta)
            except RECOVERABLE_TRIAL_ERRORS as exc:
                reason = f"{type(exc).__name__}: {exc}"
                trials.append(OuterTrial(iteration, attempt, alpha, False, None, reason))
                if options.verbosity >= 2:
                    print(
                        f"\t[line search] "
                        f"attempt={attempt} alpha={alpha:.4e} "
                        f"rejected: {reason}",
                        flush=True,
                    )
            else:
                # This assignment replaces the previous inline expression.
                rhs = point.loss + options.armijo * alpha * slope
                accepted = trial.loss <= rhs
                if options.verbosity >= 2:
                    print(
                        f"\t[line search] "
                        f"attempt={attempt} alpha={alpha:.4e} "
                        f"Ftrial={trial.loss:.6e} "
                        f"dF={trial.loss-point.loss:+.4e} "
                        f"required_dF={options.armijo*alpha*slope:+.4e} "
                        f"accepted={accepted}",
                        flush=True,
                    )
                if accepted:
                    candidate = trial
                    trials.append(OuterTrial(iteration, attempt, alpha, True, trial.loss, "armijo"))
                    break
                trials.append(OuterTrial(iteration, attempt, alpha, False, trial.loss, "insufficient_decrease"))
            alpha *= options.backtrack
        if candidate is None:
            if exhausted:
                status, message = "max_evaluations", "Outer evaluation budget exhausted during line search"
            else:
                reason = trials[-1].reason if trials else "step is below a numerical or configured limit"
                status, message = ("line_search_failed",
                                   "No admissible Armijo-decreasing trial was accepted; "+reason)
            return OptimizerStep(status=status, message=message, trials=tuple(trials))
        # 4. Store only acceptable positive-curvature pairs in this fixed chart.
        s = candidate.theta-point.theta
        y = candidate.gradient_theta-point.gradient_theta
        sy = float(np.dot(s, y))
        curvature_scale = float(np.linalg.norm(s)*np.linalg.norm(y))
        update = (np.isfinite(sy) and sy > 0 and np.isfinite(curvature_scale)
                  and sy > options.curvature_rtol*curvature_scale and np.isfinite(1.0/sy))
        if update:
            pairs.append((s.copy(), y.copy(), 1.0/sy))
            del pairs[:-options.memory]
        # 5. Only this accepted snapshot is offered to the common lifecycle.
        return OptimizerStep(
            evaluation=candidate, trials=tuple(trials), step_size=alpha,
            step_norm=float(np.linalg.norm(s)), line_search_trials=len(trials),
            diagnostics={"curvature_update": bool(update), "memory_size": len(pairs)},
        )

    def _iteration_record(self, point, index, *, previous=None, step=None):
        common = super()._iteration_record(point, index, previous=previous, step=step)
        details = {} if step is None else step.diagnostics
        return OuterIteration(**asdict(common), curvature_update=details.get("curvature_update", False),
                              memory_size=details.get("memory_size", 0))

    def _build_result(self, point, summary):
        return OuterOptimizationResult(
            point, _readonly(self.objective.reference), _readonly(summary.initial_evaluation.theta),
            float(summary.initial_evaluation.loss), summary.gradient_threshold,
            summary.tangent_gradient_threshold, summary.history, summary.trials,
            summary.converged, summary.status, summary.message, summary.evaluations,
            summary.failed_evaluations, summary.cache_hits, summary.inner_sweeps,
        )


# Reserved GDM subclass

class GDMOptimizer(OuterOptimizer):
    """Extension point for GDM; geometric steps are not implemented yet.

    A future implementation supplies a manifold objective/point, its own
    retraction or exponential path, direction/history transport, acceptance
    procedure and result builder. None of those choices belong in the parent.
    """

    @staticmethod
    def _unimplemented():
        raise NotImplementedError(
            "GDMOptimizer is not implemented yet; its geometric update and solver state "
            "must be supplied in this subclass."
        )

    def _initialize(self, initial):
        self._unimplemented()

    def _step(self, point, run, iteration):
        self._unimplemented()

    def _build_result(self, point, summary):
        self._unimplemented()


# Solver selection and compatibility entry points

def _minimize_exponential(objective,theta0,options,callback=None):
    """Legacy private driver; final validation remains the caller's responsibility.

    Kept for the step-6 controlled tests and existing private callers. New
    code should construct LBFGSOptimizer and use optimize() directly.
    """
    options = replace(_coerce_lbfgs_options(options),validate_final=False)
    return LBFGSOptimizer(objective,options=options).optimize(theta0,callback=callback)



def optimize_gauge(evaluator,*,theta0=None,reference=None,inner_t0=None,
                    inner_options=None,options=None,callback=None,solver="lbfgs"):
    """Select an outer solver; L-BFGS uses fixed-exponential nested relaxation.

    New code can construct ExponentialGaugeObjective and LBFGSOptimizer
    explicitly. options accepts LBFGSOptions, common OptimizerOptions or the
    legacy OuterOptions. solver='gdm' is reserved and raises NotImplementedError
    before objective construction. Solver selection lives outside the parent.

    callback(record) receives initial/accepted scalar diagnostics. Raising
    StopIteration returns that accepted state with status='callback_stopped'.
    Final full gauge/ERI validation is run once by default, outside the loops.
    """
    if not isinstance(solver,str):
        raise TypeError("solver must be a string: 'lbfgs' or 'gdm'")
    if solver not in ("lbfgs","gdm"):
        raise ValueError(f"Unknown solver {solver!r}; expected 'lbfgs' or 'gdm'")
    if solver == "gdm":
        GDMOptimizer._unimplemented()
    options = _coerce_lbfgs_options(options)
    objective = ExponentialGaugeObjective(evaluator,reference=reference,
                                          inner_options=inner_options,inner_t0=inner_t0)
    return LBFGSOptimizer(objective,options=options).optimize(theta0,callback=callback)

