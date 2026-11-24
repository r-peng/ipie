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
    OuterOptimizationResult,
)
from .channels import check_cholesky_shift, cholesky_shift_array
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
        # Relaxed shifts are reported; fixed or absent shifts are not.
        shift = (tuple(float(mu) for mu in inner.shift)
                 if getattr(inner, "optimize_shift", False) else None)
        return OptimizerIteration(
            index, point.loss, point.gradient_norm, point.tangent_gradient_norm,
            0. if step is None else step.step_size,
            0. if step is None else step.step_norm,
            0. if previous is None else previous.loss-point.loss,
            inner.sweeps, inner.projected_gradient_norm,
            0 if step is None else step.line_search_trials,
            shift,
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
                    f"inner_grad={row.inner_projected_gradient_norm:.3e}"
                    + ("" if getattr(row, "shift", None) is None else
                       f" mu=({row.shift[0]:+.8e},{row.shift[1]:+.8e})"),
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

def factor_eigenvalues(problem,O=None):
    """(r, n) ascending eigenvalues of the rotated factors L_g(O) (O=None: the input factors)."""
    data = problem.data
    O = np.eye(data.nchol) if O is None else _real_array(O,"O")
    if O.shape != (data.nchol,data.nchol):
        raise ValueError("O must have shape (nchol, nchol)")
    chol = (O@data.chol.reshape(data.nchol,-1)).reshape(data.chol.shape)
    return np.linalg.eigh(chol)[0]


def median_cholesky_shift(problem,O=None):
    """Each rotated factor's spectral median, as the midpoint of its middle gap: shape (r,).

    For an even number of orbitals this is np.median of the eigenvalues, as
    in sor_base.compute_cholesky_shift. For an odd number np.median is the
    middle eigenvalue itself (lambda = 0 exactly: balanced_log raises there
    and the relaxed loss has a kink), so the midpoint of the wider of the two
    gaps next to it is used instead. A single orbital gives 0.
    """
    ell = factor_eigenvalues(problem,O)
    n = ell.shape[1]
    if n == 1:
        return np.zeros(ell.shape[0])
    m = n//2
    if n % 2 == 0:
        return .5*(ell[:,m-1]+ell[:,m])
    below,above = ell[:,m]-ell[:,m-1],ell[:,m+1]-ell[:,m]
    return np.where(above > below,.5*(ell[:,m]+ell[:,m+1]),.5*(ell[:,m-1]+ell[:,m]))


def cholesky_shift_scale(problem,O=None):
    """s_g = max - min eigenvalue of each rotated factor (1 for a flat factor): shape (r,).

    The outer coordinates of a Cholesky shift are mu/s, so that one unit
    moves mu across the whole spectrum of its factor.
    """
    ell = factor_eigenvalues(problem,O)
    spread = ell[:,-1]-ell[:,0]
    return np.where(np.isfinite(spread) & (spread > 0),spread,1.)


def resolve_cholesky_shift(problem,start,O=None):
    """A Cholesky shift (r, 2) from 'median', 'zero', or an (r,) / (r, 2) array.

    'median' is median_cholesky_shift at the gauge O for both spins (spin up
    only for aa_only, whose mu_down is 0); an (r,) array is tied the same way.
    """
    r = problem.data.nchol
    if isinstance(start,str):
        if start == "median":
            start = median_cholesky_shift(problem,O)
        elif start == "zero":
            return np.zeros((r,2))
        else:
            raise ValueError("A Cholesky shift start must be 'median', 'zero' or an array")
    mu = cholesky_shift_array(start,r,problem.policy)
    check_cholesky_shift(mu,problem.policy,problem.nelec)
    return np.zeros((r,2)) if mu is None else np.array(mu)


def _shift_mode(value):
    """False (or None) for no optimized Cholesky shift, else 'tied' or 'split'."""
    if value is None or value is False:
        return None
    if isinstance(value,str) and value in ("tied","split"):
        return value
    raise ValueError("optimize_cholesky_shift must be False, 'tied' or 'split'")


class ExponentialGaugeObjective:
    """Reduced value/gradient oracle with one immutable coordinate reference.

    The inner configuration and supplied seed are frozen for a run. With no
    seed, every trial starts at its own interval midpoints. A rejected trial
    never becomes the seed for another evaluation. The one-entry cache holds
    the coordinates, value, gradient and inner solution together; no
    unbounded snapshot history is retained. General inner solutions are
    tracked local solutions, not certified global minima.

    Coordinates. By default they are the packed gauge angles theta of
    O = expm(S(theta)) @ reference, and the evaluator's own Cholesky shift
    (if any) stays fixed. With optimize_cholesky_shift='tied' or 'split' they
    are x = (theta, mu/s) (pack and unpack convert):
      * 'tied': one mu_g for both spins (spin up only for aa_only), r entries;
      * 'split': mu[:, up] then mu[:, down], 2r entries (r for aa_only, which
        has no down-spin channels and is then the same as 'tied');
      * s = cholesky_scale (r,), fixed for the run; None takes
        cholesky_shift_scale at the reference gauge. optimize_gauge uses the
        starting gauge instead.
    frozen_theta holds the gauge angles fixed (mu only); x is then mu/s.
    The Cholesky shift is an outer variable: every evaluation solves the
    inner problem at that shift and uses its envelope gradient.
    """

    def __init__(self,evaluator,*,reference=None,inner_options=None,inner_t0=None,
                 optimize_cholesky_shift=False,cholesky_scale=None,frozen_theta=None):
        if not isinstance(evaluator,GaugeEvaluator):
            raise TypeError("evaluator must be a GaugeEvaluator")
        self._evaluator = evaluator
        self._nchol = evaluator.problem.data.nchol
        if inner_options is not None and not isinstance(inner_options,InnerOptions):
            raise TypeError("inner_options must be InnerOptions")
        self._inner_options = InnerOptions() if inner_options is None else inner_options
        _partition_limits(evaluator.amplitude_bound,self.inner_options.amplitude_limit,
                          self.inner_options.max_abs_t)
        self._reference = _readonly(exponential_gauge(np.zeros(self.nchol*(self.nchol-1)//2),
                                                      self.nchol,reference))
        if inner_t0 is None:
            self._inner_t0 = None
        else:
            seed = _real_array(inner_t0,"inner_t0")
            if seed.ndim != 0 and seed.shape != (len(evaluator.sources),):
                raise ValueError("inner_t0 must be a scalar or have one entry per source")
            self._inner_t0 = _readonly(seed)
        self._shift_mode = _shift_mode(optimize_cholesky_shift)
        self._engine = evaluator
        self._scale = None
        self._frozen = None
        self._nshift = 0
        if self._shift_mode is None:
            if cholesky_scale is not None or frozen_theta is not None:
                raise ValueError("cholesky_scale and frozen_theta need optimize_cholesky_shift='tied' or 'split'")
        else:
            problem = evaluator.problem
            if problem.nelec is None:
                raise ValueError("Optimizing a Cholesky shift requires particle numbers: pass nelec to prepare_problem")
            tied = self._shift_mode == "tied" or problem.policy == "aa_only"
            self._nshift = self.nchol if tied else 2*self.nchol
            self._engine = evaluator.with_cholesky_gradient("tied" if tied else "split")
            scale = cholesky_shift_scale(problem,self._reference) if cholesky_scale is None else \
                _real_array(cholesky_scale,"cholesky_scale")
            if scale.shape != (self.nchol,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"cholesky_scale must hold {self.nchol} finite positive numbers")
            self._scale = _readonly(scale)
            if frozen_theta is not None:
                frozen = _real_array(frozen_theta,"frozen_theta")
                if frozen.shape != (self.nchol*(self.nchol-1)//2,):
                    raise ValueError(f"frozen_theta must be a length-{self.nchol*(self.nchol-1)//2} vector")
                self._frozen = _readonly(frozen)
        self._ntheta = 0 if self._frozen is not None else self.nchol*(self.nchol-1)//2
        self._nparams = self._ntheta+self._nshift
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
    def ntheta(self):
        """Gauge angles among the coordinates: r(r-1)/2, or 0 with frozen_theta."""
        return self._ntheta

    @property
    def nshift(self):
        """Scaled Cholesky-shift entries among the coordinates (0 when the shift is not optimized)."""
        return self._nshift

    @property
    def optimize_cholesky_shift(self):
        """None, 'tied' or 'split' (as requested; aa_only is laid out as 'tied')."""
        return self._shift_mode

    @property
    def cholesky_scale(self):
        """The fixed scale s (r,) of the shift coordinates mu/s, or None."""
        return self._scale

    @property
    def frozen_theta(self):
        return self._frozen

    @property
    def inner_options(self):
        return self._inner_options

    @property
    def reference(self):
        return self._reference

    @property
    def inner_t0(self):
        return self._inner_t0

    def unpack(self,x):
        """(theta, mu) for coordinates x; mu is the (r, 2) Cholesky shift, or None if it is not optimized."""
        x = _real_array(x,"x")
        if x.shape != (self.nparams,):
            raise ValueError(f"x must be a length-{self.nparams} vector")
        theta = self._frozen if self._frozen is not None else x[:self._ntheta]
        if self._shift_mode is None:
            return theta,None
        y = x[self._ntheta:]*np.tile(self._scale,self._nshift//self.nchol)
        mu = np.zeros((self.nchol,2))
        if self._nshift == self.nchol:
            mu[:,0] = y
            if self._evaluator.problem.policy != "aa_only":
                mu[:,1] = y
        else:
            mu[:,0],mu[:,1] = y[:self.nchol],y[self.nchol:]
        return theta,mu

    def pack(self,theta=None,cholesky_shift=None):
        """Coordinates x for gauge angles theta (zeros if None) and a Cholesky shift ((r,) or (r, 2)).

        A 'tied' layout needs a tied shift (equal columns; mu_down = 0 for aa_only).
        """
        r = self.nchol
        parts = []
        if self._frozen is None:
            theta = np.zeros(r*(r-1)//2) if theta is None else _real_array(theta,"theta")
            if theta.shape != (r*(r-1)//2,):
                raise ValueError(f"theta must be a length-{r*(r-1)//2} vector")
            parts.append(theta)
        if self._shift_mode is not None:
            mu = resolve_cholesky_shift(self._evaluator.problem,
                                        np.zeros(r) if cholesky_shift is None else cholesky_shift)
            if self._nshift == r:
                other = np.zeros(r) if self._evaluator.problem.policy == "aa_only" else mu[:,0]
                if not np.array_equal(mu[:,1],other):
                    raise ValueError("A tied layout needs a tied Cholesky shift (equal spin columns)")
                parts.append(mu[:,0]/self._scale)
            else:
                parts.extend((mu[:,0]/self._scale,mu[:,1]/self._scale))
        elif cholesky_shift is not None:
            raise ValueError("This objective does not optimize a Cholesky shift")
        return np.concatenate(parts) if parts else np.zeros(0)

    def _shift_coordinates_gradient(self,derivative):
        """d loss/d (mu/s) from the evaluator's cholesky_gradient, in the coordinate layout."""
        g = np.asarray(derivative.cholesky_gradient,dtype=np.float64)
        if self._nshift == self.nchol:
            return g*self._scale
        return np.concatenate([g[:,0]*self._scale,g[:,1]*self._scale])

    def evaluate(self,theta):
        """Return one matching value/coordinate-gradient/inner snapshot.

        theta: the coordinates (the gauge angles, followed by mu/s when the
        Cholesky shift is optimized; see the class docstring).
        """
        theta = _real_array(theta,"theta")
        if theta.shape != (self.nparams,):
            raise ValueError(f"theta must be a length-{self.nparams} vector")
        if self._cached is not None and np.array_equal(theta,self._cached.coordinates):
            self.cache_hits += 1
            return self._cached
        self.evaluations += 1
        inner = None
        try:
            if self._shift_mode is None:
                O = exponential_gauge(theta,self.nchol,self.reference)
                derivative = self.evaluator.relaxed(O,t0=self.inner_t0,options=self.inner_options)
            else:
                angles,mu = self.unpack(theta)
                O = exponential_gauge(angles,self.nchol,self.reference)
                derivative = self._engine.relaxed(O,t0=self.inner_t0,options=self.inner_options,
                                                  cholesky_shift=mu)
            inner = derivative.inner_result
            if inner is None or not inner.converged:
                raise InnerConvergenceError("Outer objective requires a converged inner snapshot",inner)
            if self._shift_mode is None:
                gradient = exponential_gauge_vjp(theta,self.nchol,derivative.gradient_O,self.reference)
                if (not np.isfinite(derivative.loss) or not np.isfinite(gradient).all()
                        or not np.isfinite(derivative.tangent_gradient).all()):
                    raise FloatingPointError("Nonfinite outer value or coordinate gradient")
                point = ExponentialEvaluation(_readonly(theta),_readonly(gradient),derivative)
            else:
                gradient = exponential_gauge_vjp(angles,self.nchol,derivative.gradient_O,self.reference)
                shift_gradient = self._shift_coordinates_gradient(derivative)
                if (not np.isfinite(derivative.loss) or not np.isfinite(gradient).all()
                        or not np.isfinite(shift_gradient).all()
                        or not np.isfinite(derivative.tangent_gradient).all()):
                    raise FloatingPointError("Nonfinite outer value or coordinate gradient")
                point = ExponentialEvaluation(
                    _readonly(angles),_readonly(gradient),derivative,
                    scaled_shift=_readonly(theta[self._ntheta:]),
                    gradient_scaled_shift=_readonly(shift_gradient),
                    theta_frozen=self._frozen is not None)
            if not math.isfinite(point.tangent_gradient_norm):
                raise FloatingPointError("Nonfinite physical tangent-gradient norm")
        except RECOVERABLE_TRIAL_ERRORS as exc:
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
        return point.loss,point.gradient.copy()

    def validate(self,point):
        """Full final gauge/ERI check, called outside the optimizer loops."""
        problem = self.evaluator.problem
        return validate_gauge_state(problem.data,point.decomposition.two_body.gauge,
                                    problem.report.options,input_report=problem.report)





# Explicit L-BFGS: directions, Armijo line search, curvature updates

@dataclass(frozen=True)
class _SearchOutcome:
    """Line-search result: an accepted evaluation, or a failure status."""

    candidate: Any
    step_size: float
    trials: Tuple[OuterTrial, ...]
    acceptance: str
    status: str = ""
    message: str = ""


def _coerce_lbfgs_options(options):
    if options is None:
        return LBFGSOptions()
    if isinstance(options, LBFGSOptions):
        return options
    if isinstance(options, OptimizerOptions):
        return LBFGSOptions(**asdict(options))
    raise TypeError("options must be LBFGSOptions or OptimizerOptions")


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
        gradient = point.gradient
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
        # 3. Line search; every new gauge re-relaxes its partitions.
        search = (self._weak_wolfe_search if options.line_search == "weak_wolfe"
                  else self._armijo_search)
        outcome = search(point, direction, slope, run, iteration)
        if outcome.candidate is None:
            return OptimizerStep(status=outcome.status, message=outcome.message,
                                 trials=outcome.trials)
        candidate, alpha = outcome.candidate, outcome.step_size
        # 4. Store only acceptable positive-curvature pairs in this fixed chart.
        s = candidate.coordinates-point.coordinates
        y = candidate.gradient-point.gradient
        sy = float(np.dot(s, y))
        curvature_scale = float(np.linalg.norm(s)*np.linalg.norm(y))
        update = (np.isfinite(sy) and sy > 0 and np.isfinite(curvature_scale)
                  and sy > options.curvature_rtol*curvature_scale and np.isfinite(1.0/sy))
        if update:
            pairs.append((s.copy(), y.copy(), 1.0/sy))
            del pairs[:-options.memory]
        # 5. Only this accepted snapshot is offered to the common lifecycle.
        return OptimizerStep(
            evaluation=candidate, trials=outcome.trials, step_size=alpha,
            step_norm=float(np.linalg.norm(s)), line_search_trials=len(outcome.trials),
            diagnostics={"curvature_update": bool(update), "memory_size": len(pairs),
                         "acceptance": outcome.acceptance},
        )

    def _print_rejected(self, attempt, alpha, reason):
        if self.options.verbosity >= 2:
            print(
                f"\t[line search] "
                f"attempt={attempt} alpha={alpha:.4e} "
                f"rejected: {reason}",
                flush=True,
            )

    def _armijo_search(self, point, direction, slope, run, iteration):
        """Original sufficient-decrease backtracking, unchanged in behavior."""
        options = self.options
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
            theta = point.coordinates+alpha*direction
            if np.array_equal(theta, point.coordinates):
                break
            try:
                trial = self.objective.evaluate(theta)
            except RECOVERABLE_TRIAL_ERRORS as exc:
                reason = f"{type(exc).__name__}: {exc}"
                trials.append(OuterTrial(iteration, attempt, alpha, False, None, reason))
                self._print_rejected(attempt, alpha, reason)
            else:
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
        if candidate is not None:
            return _SearchOutcome(candidate, alpha, tuple(trials), "armijo")
        if exhausted:
            status, message = "max_evaluations", "Outer evaluation budget exhausted during line search"
        else:
            reason = trials[-1].reason if trials else "step is below a numerical or configured limit"
            status, message = ("line_search_failed",
                               "No admissible Armijo-decreasing trial was accepted; "+reason)
        return _SearchOutcome(None, 0.0, tuple(trials), "", status, message)

    def _weak_wolfe_search(self, point, direction, slope, run, iteration):
        """Lewis-Overton weak-Wolfe bracketing for objectives that may have kinks.

        Accept alpha when both
          F(theta+alpha*p) <= F(theta) + armijo*alpha*slope        (decrease)
          g(theta+alpha*p).p >= wolfe_c2*slope                      (weak curvature)
        hold. A decrease failure or a recoverable invalid trial sets the upper
        bracket; a curvature failure sets the lower bracket. The next trial
        bisects a finite bracket, otherwise doubles alpha up to the expansion
        cap wolfe_max_step_norm/||p||. The weak (one-sided) curvature test is
        essential: at a kink the directional derivative jumps and never becomes
        small, so strong Wolfe has no solution there, whereas any point just
        beyond the kink satisfies the weak test and yields s.y > 0.

        If no point satisfies both conditions within max_line_search trials,
        the lowest-loss trial that satisfied the decrease condition is accepted
        (acceptance "armijo_only", or "expansion_capped" if expansion stopped
        at the cap). Only when no trial decreased F does the step fail; at a
        nonsmooth minimizer that is the expected termination.
        """
        options = self.options
        c1, c2 = options.armijo, options.wolfe_c2
        alpha_max = options.wolfe_max_step_norm/float(np.linalg.norm(direction))
        lower, upper = 0.0, math.inf
        alpha = min(options.initial_step, alpha_max)
        trials = []
        best = None  # (trial index, evaluation, alpha) with lowest decreasing loss
        exhausted = False
        stop = "max_line_search trials used"
        capped = False
        for attempt in range(1, options.max_line_search+1):
            if alpha < options.min_step:
                stop = "alpha below min_step"
                break
            if upper < math.inf and upper-lower <= options.min_step*max(1.0, upper):
                stop = "bracket collapsed below min_step"
                break
            if self._budget_exhausted(run):
                exhausted = True
                break
            theta = point.coordinates+alpha*direction
            if np.array_equal(theta, point.coordinates) or (
                    best is not None and np.array_equal(theta, best[1].coordinates)):
                stop = "trial parameters unchanged in floating point"
                break
            try:
                trial = self.objective.evaluate(theta)
            except RECOVERABLE_TRIAL_ERRORS as exc:
                reason = f"{type(exc).__name__}: {exc}"
                trials.append(OuterTrial(iteration, attempt, alpha, False, None, reason))
                self._print_rejected(attempt, alpha, reason)
                upper = alpha
            else:
                trial_slope = float(np.dot(trial.gradient, direction))
                decrease = trial.loss <= point.loss+c1*alpha*slope
                curvature = math.isfinite(trial_slope) and trial_slope >= c2*slope
                if decrease and curvature:
                    reason = "weak_wolfe"
                elif not decrease:
                    reason = "insufficient_decrease"
                else:
                    reason = "insufficient_curvature"
                if options.verbosity >= 2:
                    print(
                        f"\t[line search] "
                        f"attempt={attempt} alpha={alpha:.4e} "
                        f"Ftrial={trial.loss:.6e} "
                        f"dF={trial.loss-point.loss:+.4e} "
                        f"required_dF={c1*alpha*slope:+.4e} "
                        f"slope_ratio={trial_slope/slope:+.4e} "
                        f"required_ratio<={c2:.3g} "
                        f"result={reason}",
                        flush=True,
                    )
                if reason == "weak_wolfe":
                    trials.append(OuterTrial(iteration, attempt, alpha, True, trial.loss, reason))
                    return _SearchOutcome(trial, alpha, tuple(trials), reason)
                trials.append(OuterTrial(iteration, attempt, alpha, False, trial.loss, reason))
                if not decrease:
                    upper = alpha
                else:
                    lower = alpha
                    if best is None or trial.loss < best[1].loss:
                        best = (len(trials)-1, trial, alpha)
                    if upper == math.inf and alpha >= alpha_max:
                        capped = True
                        stop = "step expansion reached wolfe_max_step_norm"
                        break
            alpha = 0.5*(lower+upper) if upper < math.inf else min(2.0*lower, alpha_max)
        if best is not None:
            index, candidate, alpha = best
            acceptance = "expansion_capped" if capped else "armijo_only"
            trials[index] = replace(trials[index], accepted=True, reason=acceptance)
            return _SearchOutcome(candidate, alpha, tuple(trials), acceptance)
        if exhausted:
            status, message = "max_evaluations", "Outer evaluation budget exhausted during line search"
        else:
            last = trials[-1].reason if trials else "no trial was evaluated"
            status, message = (
                "line_search_failed",
                f"Weak-Wolfe search found no decreasing trial ({stop}; last: {last}). "
                "At a nonsmooth (kink) minimizer this is the expected termination; "
                "see near_zero_eigenvalues(result).")
        return _SearchOutcome(None, 0.0, tuple(trials), "", status, message)

    def _iteration_record(self, point, index, *, previous=None, step=None):
        common = super()._iteration_record(point, index, previous=previous, step=step)
        details = {} if step is None else step.diagnostics
        return OuterIteration(**asdict(common), curvature_update=details.get("curvature_update", False),
                              memory_size=details.get("memory_size", 0),
                              acceptance=details.get("acceptance", ""))

    def _build_result(self, point, summary):
        initial = summary.initial_evaluation
        if getattr(self.objective, "nshift", 0):
            shift = self.objective.unpack(initial.coordinates)[1]       # zeros stay zeros
        else:
            shift = initial.cholesky_shift
        scale = getattr(self.objective, "cholesky_scale", None)
        return OuterOptimizationResult(
            point, _readonly(self.objective.reference), _readonly(initial.theta),
            float(initial.loss), summary.gradient_threshold,
            summary.tangent_gradient_threshold, summary.history, summary.trials,
            summary.converged, summary.status, summary.message, summary.evaluations,
            summary.failed_evaluations, summary.cache_hits, summary.inner_sweeps,
            initial_cholesky_shift=None if shift is None else _readonly(shift),
            cholesky_scale=None if scale is None else _readonly(scale),
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

def optimize_gauge(evaluator,*,theta0=None,reference=None,inner_t0=None,
                    inner_options=None,options=None,callback=None,solver="lbfgs",
                    optimize_cholesky_shift=False,cholesky_shift0=None,optimize_theta=True):
    """Select an outer solver; L-BFGS uses fixed-exponential nested relaxation.

    New code can construct ExponentialGaugeObjective and LBFGSOptimizer
    explicitly. options accepts LBFGSOptions or common OptimizerOptions. 
    solver='gdm' is reserved and raises NotImplementedError
    before objective construction. Solver selection lives outside the parent.

    callback(record) receives initial/accepted scalar diagnostics. Raising
    StopIteration returns that accepted state with status='callback_stopped'.
    Final full gauge/ERI validation is run once by default, outside the loops.

    Cholesky shifts (all need nelec):
      * optimize_cholesky_shift=False (default) optimizes theta. The shift is
        the evaluator's own, or cholesky_shift0 if given (then held fixed).
      * optimize_cholesky_shift='tied' or 'split' optimizes x = (theta, mu/s),
        with s = cholesky_shift_scale at the starting gauge O(theta0), fixed
        for the run. cholesky_shift0 is the start: 'median' (the default),
        'zero', or an (r,) / (r, 2) array (resolve_cholesky_shift, at O(theta0)).
      * optimize_theta=False holds the gauge at theta0 and optimizes mu only.
    The result reports initial_cholesky_shift, cholesky_scale and cholesky_shift.
    """
    if not isinstance(solver,str):
        raise TypeError("solver must be a string: 'lbfgs' or 'gdm'")
    if solver not in ("lbfgs","gdm"):
        raise ValueError(f"Unknown solver {solver!r}; expected 'lbfgs' or 'gdm'")
    if solver == "gdm":
        GDMOptimizer._unimplemented()
    options = _coerce_lbfgs_options(options)
    mode = _shift_mode(optimize_cholesky_shift)
    if not isinstance(optimize_theta,(bool,np.bool_)):
        raise ValueError("optimize_theta must be boolean")
    if mode is None and not optimize_theta:
        raise ValueError("Nothing to optimize: optimize_theta=False needs optimize_cholesky_shift='tied' or 'split'")
    if mode is None and cholesky_shift0 is None:
        objective = ExponentialGaugeObjective(evaluator,reference=reference,
                                              inner_options=inner_options,inner_t0=inner_t0)
        return LBFGSOptimizer(objective,options=options).optimize(theta0,callback=callback)
    if not isinstance(evaluator,GaugeEvaluator):
        raise TypeError("evaluator must be a GaugeEvaluator")
    r = evaluator.problem.data.nchol
    angles = np.zeros(r*(r-1)//2) if theta0 is None else _real_array(theta0,"theta0")
    if angles.shape != (r*(r-1)//2,):
        raise ValueError(f"theta0 must be a length-{r*(r-1)//2} vector")
    start = exponential_gauge(angles,r,reference)
    mu0 = resolve_cholesky_shift(evaluator.problem,"median" if cholesky_shift0 is None else cholesky_shift0,start)
    if mode is None:
        # theta only, at a fixed shift
        objective = ExponentialGaugeObjective(evaluator.with_cholesky_shift(mu0),reference=reference,
                                              inner_options=inner_options,inner_t0=inner_t0)
        return LBFGSOptimizer(objective,options=options).optimize(theta0,callback=callback)
    objective = ExponentialGaugeObjective(
        evaluator,reference=reference,inner_options=inner_options,inner_t0=inner_t0,
        optimize_cholesky_shift=mode,cholesky_scale=cholesky_shift_scale(evaluator.problem,start),
        frozen_theta=None if optimize_theta else angles)
    return LBFGSOptimizer(objective,options=options).optimize(objective.pack(angles,mu0),callback=callback)
