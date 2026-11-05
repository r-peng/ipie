"""Solver-independent outer lifecycle and protected subclass contracts.

The parent does not select a search direction, coordinate representation,
trial path, line search, curvature model or concrete result record.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
import math
from typing import Any, Tuple

import numpy as np

from ..models import (
    GaugeFeasibilityError, GaugeRegularityError, InnerConvergenceError,
    OptimizerIteration, OptimizerOptions, OuterTrial,
)


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
            if callback is not None:
                callback(history[-1])

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
            result = replace(result, validation_report=self.objective.validate(point))
        return result
