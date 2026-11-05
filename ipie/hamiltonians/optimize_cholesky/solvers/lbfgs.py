"""Fixed-exponential L-BFGS child: direction, line search and curvature only."""

from dataclasses import asdict

import numpy as np

from ..models import LBFGSOptions, OptimizerOptions, OuterIteration, OuterOptions, OuterOptimizationResult, OuterTrial
from ..partitions import _readonly
from .base import OuterOptimizer, OptimizerStep, RECOVERABLE_TRIAL_ERRORS


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
                trials.append(OuterTrial(iteration, attempt, alpha, False, None,
                                         f"{type(exc).__name__}: {exc}"))
            else:
                if trial.loss <= point.loss+options.armijo*alpha*slope:
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
        s = candidate.theta-point.theta
        y = candidate.gradient_theta-point.gradient_theta
        sy = float(np.dot(s, y))
        curvature_scale = float(np.linalg.norm(s)*np.linalg.norm(y))
        update = (np.isfinite(sy) and sy > 0 and np.isfinite(curvature_scale)
                  and sy > options.curvature_rtol*curvature_scale and np.isfinite(1.0/sy))
        if update:
            pairs.append((s.copy(), y.copy(), 1.0/sy))
            del pairs[:-options.memory]
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
