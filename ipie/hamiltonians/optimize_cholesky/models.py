"""Data contracts for real, zero-shift Cholesky gauge optimization."""

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ValidationOptions:
    """Acceptance: max|candidate-reference| <= atol + rtol*max|reference|.

    Symmetry and eigenpair checks use a separate scale for each factor.
    ``block_size`` is the number of flattened (p,r) rows per ERI check.
    """

    atol: float = 1e-12
    rtol: float = 1e-10
    block_size: int = 16

    def __post_init__(self):
        for name in ("atol", "rtol"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative real number")
        if isinstance(self.block_size, bool) or not isinstance(self.block_size, Integral):
            raise ValueError("block_size must be a positive integer")
        if self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")


@dataclass(frozen=True)
class ErrorMetric:
    max_abs_error: float
    reference_max_abs: float
    frobenius_error: float
    reference_frobenius: float
    relative_frobenius: Optional[float]
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ValidationReport:
    checks: dict
    options: ValidationOptions

    @property
    def passed(self):
        return all(metric.passed for metric in self.checks.values())

    def as_dict(self):
        """JSON-compatible diagnostics, including null for undefined relatives."""
        return {"passed": self.passed, "options": asdict(self.options),
                "checks": {key: asdict(value) for key, value in self.checks.items()}}


class ValidationError(ValueError):
    """A numerical identity failed; ``report`` retains all computed metrics."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class TensorData:
    """h1[p,q] is original h; eri[p,r,q,s]; chol[gamma,p,r].

    This is a lightweight container. Call validate_inputs or prepare_problem
    before using a directly constructed container. No casting or repair occurs
    in its constructor, so invalid input cannot be silently hidden.
    """

    h1: np.ndarray
    eri: np.ndarray
    chol: np.ndarray

    @property
    def nbasis(self):
        return self.h1.shape[0]

    @property
    def nchol(self):
        return self.chol.shape[0]


@dataclass(frozen=True)
class PreparedProblem:
    """Validated float64 copies; particle counts are optional in step 1."""

    data: TensorData
    report: ValidationReport
    nelec: Optional[Tuple[int, int]]
    policy: str


@dataclass(frozen=True)
class GaugeState:
    """One evaluated gauge, not a claim that an optimization has converged.

    X[gamma,p,alpha] are eigenvectors in columns, in ascending eigenvalue
    order. All factors/modes, including zero and degenerate ones, are retained.
    """

    O: np.ndarray
    chol: np.ndarray
    eigenvalues: np.ndarray
    X: np.ndarray
    diagnostics: Optional[ValidationReport] = None


@dataclass(frozen=True)
class Channel:
    """Two-body channel schema, using the requested sor_base fields.

    bix indexes the FINAL Cholesky eigenbasis; spin 0=up, 1=down.
    a_i is an UNNORMALIZED positive coefficient in Lambda*I-H, not an H
    coefficient or sampling probability. d contains the actual branch signs.
    Basis and mode upper bounds are checked by the channel builders.
    """

    bix: int
    spins: Tuple[int, int]
    a_i: float
    modes: Tuple[int, int]
    d: Tuple[float, float]

    def __post_init__(self):
        def integer(value):
            return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))

        if not integer(self.bix) or self.bix < 0:
            raise ValueError("bix must be a nonnegative integer")
        for name in ("spins", "modes", "d"):
            value = getattr(self, name)
            try:
                values = tuple(value)
            except TypeError as exc:
                raise ValueError(f"{name} must contain exactly two entries") from exc
            if len(values) != 2:
                raise ValueError(f"{name} must contain exactly two entries")
            object.__setattr__(self, name, values)
        if any(not integer(s) or s not in (0, 1) for s in self.spins):
            raise ValueError("spins must contain 0 (up) or 1 (down)")
        if any(not integer(p) or p < 0 for p in self.modes):
            raise ValueError("modes must contain nonnegative integers")
        if not isinstance(self.a_i, Real) or not np.isfinite(self.a_i) or self.a_i <= 0:
            raise ValueError("a_i must be finite and strictly positive")
        if any(not isinstance(d, Real) or not np.isfinite(d) for d in self.d):
            raise ValueError("d must contain two finite real amplitudes")


@dataclass(frozen=True)
class DensitySource:
    """One partition coordinate: an "two_branch" density product or "spin_quartet".

    The source index is its position in TwoBodyDecomposition.sources. No
    coupling is stored here: W changes with the gauge, whereas topology does
    not. Quartet sources have spins=None; two-branch sources have two spins.
    """

    bix: int
    modes: Tuple[int, int]
    kind: str
    spins: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class OneBodyChannel:
    """A single leg in the joint h eigenbasis, kept separate from Channel.

    h is the original physical one-body matrix. Its basis has no Cholesky bix.
    As for two-body channels, a_i is a positive unnormalized Green weight.
    """

    spin: int
    a_i: float
    mode: int
    d: float


@dataclass(frozen=True)
class TwoBodyDecomposition:
    """Fixed-gauge, fixed-partition two-body channels and their source map.

    source_slices[i]=(start,stop) refers to channels[start:stop]. This map
    preserves the branch sharing required by each source identity. Arrays
    produced by builders are owned, read-only snapshots.
    """

    gauge: GaugeState
    policy: str
    nelec: Optional[Tuple[int, int]]
    tau2: float
    partition: str
    amplitude_bound: float
    sources: Tuple[DensitySource, ...]
    t: np.ndarray
    amplitudes: np.ndarray
    source_slices: Tuple[Tuple[int, int], ...]
    channels: Tuple[Channel, ...]

    @property
    def weight_sum(self):
        return sum((channel.a_i for channel in self.channels), 0.0)


@dataclass(frozen=True)
class OneBodyDecomposition:
    """One-body channels and the jointly diagonalized original h.

    shift=(mu_up, mu_down) gives d[k,sigma]=(mu_sigma-epsilon_k)*tau1, i.e. the
    channels represent sum_k (epsilon_k-mu_sigma) n_{k,sigma}. The remainder
    sum_sigma mu_sigma N_sigma is a constant only on a fixed (N_up, N_down)
    sector; see FixedDecomposition.shift_energy.
    """

    X: np.ndarray
    eigenvalues: np.ndarray
    tau1: float
    channels: Tuple[OneBodyChannel, ...]
    shift: Tuple[float, float] = (0.0, 0.0)

    @property
    def spins(self):
        """Spins that carry one-body channels: (0,) for aa_only, else (0, 1)."""
        return tuple(sorted({channel.spin for channel in self.channels}))

    @property
    def weight_sum(self):
        return sum((channel.a_i for channel in self.channels), 0.0)


@dataclass(frozen=True)
class FixedDecomposition:
    """Reference step-2 representation, with one-body and two-body lists.

    Iteration order is one-body channels followed by two-body channels.
    coefficients are UNNORMALIZED; denom=sum(coefficients). For E0=0,
    H=sum_i a_i*(I-U_i), G=denom*I-H=sum_i a_i*U_i on the stated sector.
    This is not yet an optimization result or a sor_base adapter.
    """

    one_body: OneBodyDecomposition
    two_body: TwoBodyDecomposition

    @property
    def nterms(self):
        return len(self.one_body.channels) + len(self.two_body.channels)

    @property
    def denom(self):
        return self.one_body.weight_sum + self.two_body.weight_sum

    @property
    def coefficients(self):
        return np.array([c.a_i for c in self.one_body.channels]
                        + [c.a_i for c in self.two_body.channels], dtype=np.float64)

    @property
    def shift(self):
        return tuple(float(mu) for mu in self.one_body.shift)

    @property
    def shift_energy(self):
        """sum_sigma mu_sigma*N_sigma; requires particle numbers if any mu != 0."""
        shift = self.shift
        if not any(shift):
            return 0.0
        nelec = self.two_body.nelec
        if nelec is None:
            raise ValueError("A nonzero one-body shift requires particle numbers (nelec)")
        return float(shift[0]*nelec[0] + shift[1]*nelec[1])

    @property
    def lambda_electronic(self):
        """Lambda without E0: denom + sum_sigma mu_sigma N_sigma.

        On the (N_up, N_down) sector, (E0+lambda_electronic)*I - H_tot equals
        sum_i a_i U_i. With zero shift this is denom.
        """
        return float(self.denom + self.shift_energy)


@dataclass(frozen=True)
class CommutatorLoss:
    """Squared single-particle losses, matching sor_base's printed diagnostics.

    comm2 is the unweighted optimization objective. Weighted fields are only
    diagnostics. Their weights are normalized by the combined coefficient
    sum when weights_normalized=True. weighted_mean=None means weight=0,
    e.g. a one-channel decomposition with no unordered pairs.
    """

    comm2: float
    weighted_comm2: float
    weight: float
    weighted_mean: Optional[float]
    pair_count: int
    weights_normalized: bool

    def as_dict(self):
        return asdict(self)


def _scaled_exponential(value, exponent):
    """Scale a nonnegative coefficient without forming an overflowing exp."""
    if value == 0:
        return 0.0
    try:
        result = math.exp(math.log(value) + exponent)
    except OverflowError as exc:
        raise FloatingPointError("Partition coefficient/profile overflow") from exc
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite partition coefficient/profile")
    return result


@dataclass(frozen=True)
class PartitionCoefficients:
    """One exact coordinate profile, with all OTHER partitions held fixed.

    L(s)=C+A*exp(2*s)+B*exp(-2*s). For numerical stability the stored
    nonnegative terms are left_term=A*exp(2*t), right_term=B*exp(-2*t)
    at the reference coordinate t. A and B are properties, not fitted data.
    C includes every unaffected pair and can be negative. The centered
    terms allow optimization even if A or B itself is not representable.
    """

    source_index: int
    t: float
    C: float
    left_term: float
    right_term: float

    def __post_init__(self):
        if isinstance(self.source_index, (bool, np.bool_)) or not isinstance(self.source_index, Integral) or self.source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        for name in ("t", "C", "left_term", "right_term"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar")
        if self.left_term < 0 or self.right_term < 0:
            raise ValueError("left_term and right_term must be nonnegative")

    @property
    def A(self):
        return _scaled_exponential(self.left_term, -2.0*self.t)

    @property
    def B(self):
        return _scaled_exponential(self.right_term, 2.0*self.t)

    @property
    def gradient(self):
        return 2.0*(self.left_term-self.right_term)

    def terms(self, t):
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, Real) or not np.isfinite(t):
            raise ValueError("t must be a finite real scalar")
        delta = float(t)-self.t
        return (_scaled_exponential(self.left_term, 2.0*delta),
                _scaled_exponential(self.right_term, -2.0*delta))

    def value(self, t):
        left, right = self.terms(t)
        return math.fsum((self.C, left, right))

    def derivative(self, t):
        left, right = self.terms(t)
        return 2.0*(left-right)

    def change(self, t):
        """L(t)-L(reference), retaining tiny quadratic coordinate decreases."""
        left,right = self.terms(t)
        exponent = 2.0*(float(t)-self.t)
        if abs(exponent) < 0.5:
            return math.fsum((self.left_term*math.expm1(exponent),
                              self.right_term*math.expm1(-exponent)))
        return math.fsum((left,right,-self.left_term,-self.right_term))


@dataclass(frozen=True)
class ShiftCoefficients:
    """Exact quadratic profile of the loss in one spin's shift mu_sigma.

    With every other variable fixed,
        L(m) = const + tau1**2 * (weight*m**2 - 2*moment*m),
    where weight = sum_k w_k and moment = sum_k w_k*epsilon_k over this spin's
    one-body channels, w_k = s_k * sum_j ||[x_k x_k^T, A_j]||_F^2 with unit
    one-body legs and loss-scaled two-body blocks A_j. lower is the closed
    feasibility bound mu >= max_k epsilon_k - one_body_limit/tau1.
    """

    spin: int
    mu: float
    lower: float
    weight: float
    moment: float
    tau1: float

    def __post_init__(self):
        for name in ("mu", "lower", "weight", "moment", "tau1"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real scalar")
        if self.weight < 0 or self.tau1 <= 0:
            raise ValueError("weight must be nonnegative and tau1 positive")

    @property
    def optimum(self):
        """Unconstrained minimizer, a weighted mean of epsilon; mu if flat."""
        return self.mu if self.weight == 0 else self.moment/self.weight

    @property
    def update(self):
        """Exact minimizer on [lower, inf)."""
        return float(max(self.optimum, self.lower))

    def gradient(self, m=None):
        m = self.mu if m is None else float(m)
        return 2.0*self.tau1**2*(self.weight*m-self.moment)

    @property
    def projected_gradient(self):
        """Wall-filtered: at the lower wall keep only min(g, 0)."""
        g = self.gradient()
        return min(g, 0.0) if self.mu <= self.lower else g

    def change(self, m):
        """L(m) - L(mu)."""
        m = float(m)
        return self.tau1**2*(m-self.mu)*(self.weight*(m+self.mu)-2.0*self.moment)


@dataclass(frozen=True)
class PartitionSweep:
    """Diagnostics after a full sequential sweep; sweep=0 is the input."""

    sweep: int
    loss: float
    projected_gradient_norm: float
    loss_decrease: float
    max_step: float
    updates: int
    shift_change: float = 0.0


@dataclass(frozen=True)
class PartitionOptimizationResult:
    """Fixed-gauge inner solve; arrays are owned read-only snapshots.

    Convergence means the infinity norm of the projected t-gradient meets
    gradient_threshold on the stored closed bounds and, when the shifts are
    optimized, that the last sweep moved every mu_sigma by at most
    shift_tolerance. loss and initial_loss use loss_kind; initial_loss is
    taken after mu is initialized to mu*(t0), input_loss before. It does not certify a
    global minimum of the joint partition problem. The final decomposition
    can be passed straight to the step-3 loss functions.
    """

    decomposition: FixedDecomposition
    initial_loss: float
    loss: float
    bounds: np.ndarray
    coefficients: Tuple[PartitionCoefficients, ...]
    gradient: np.ndarray
    projected_gradient: np.ndarray
    gradient_threshold: float
    history: Tuple[PartitionSweep, ...]
    converged: bool
    status: str
    amplitude_limit: Optional[float] = None
    max_abs_t: float = 30.0
    loss_kind: str = "raw"
    optimize_shift: bool = False
    shift_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    shift_gradient: Optional[Tuple[float, float]] = None
    shift_tolerance: Optional[float] = None
    input_loss: Optional[float] = None
    one_body_limit: Optional[float] = None
    shift_atol: float = 0.0
    shift_rtol: float = 1e-8

    @property
    def t(self):
        return self.decomposition.two_body.t

    @property
    def shift(self):
        return self.decomposition.shift

    @property
    def sweeps(self):
        return self.history[-1].sweep

    @property
    def projected_gradient_norm(self):
        return self.history[-1].projected_gradient_norm


class GaugeRegularityError(ValueError):
    """A requested smooth gauge derivative is undefined or ill-conditioned."""


class GaugeFeasibilityError(ValueError):
    """A trial gauge admits no partition within the configured amplitude box."""


class InnerConvergenceError(ValueError):
    """A relaxed gauge gradient was requested without a converged inner solve."""

    def __init__(self, message, result=None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class GradientOptions:
    """Smooth-branch guards, applied separately to each Cholesky factor.

    No eigengap is shifted or clipped. Balanced amplitudes require eigenvalues
    away from zero; anchored amplitudes can be differentiated at a simple zero.
    bound_switch_atol detects ties between an amplitude bound and the t cap.
    """

    gap_atol: float = 1e-12
    gap_rtol: float = 1e-10
    zero_atol: float = 1e-12
    zero_rtol: float = 1e-10
    bound_switch_atol: float = 1e-10

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite, real, and nonnegative")


@dataclass(frozen=True)
class GaugeGradient:
    """Raw loss and gauge derivatives at exactly the stored decomposition.

    gradient_O is the ambient derivative, tangent_gradient its Frobenius-metric
    projection, generator_gradient=skew(gradient_O @ O.T). A left plane-angle
    derivative is TWICE the corresponding upper generator entry.
    fixed_gradient_O holds t fixed; bound_correction_O supplies the fixed-z
    chain rule in fixed_z/relaxed modes. Inner optimizer steps are not unrolled.
    """

    loss: float
    decomposition: FixedDecomposition
    mode: str
    gradient_O: np.ndarray
    tangent_gradient: np.ndarray
    generator_gradient: np.ndarray
    fixed_gradient_O: np.ndarray
    bound_correction_O: np.ndarray
    partition_gradient: np.ndarray
    eigenvalue_gradient: np.ndarray
    eigenvector_gradient: np.ndarray
    rotated_cholesky_gradient: np.ndarray
    normalized_partitions: Optional[np.ndarray] = None
    inner_result: Optional[PartitionOptimizationResult] = None


@dataclass(frozen=True)
class GaugeDifference:
    step: float
    finite_difference: float
    absolute_error: float
    relative_error: float


@dataclass(frozen=True)
class GaugeGradientCheck:
    analytic: float
    differences: Tuple[GaugeDifference, ...]


def _solver_real(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def _solver_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class InnerOptions:
    """One fixed partition-solver configuration for the entire outer run.

    optimize_shift=True relaxes (mu_up, mu_down) together with t: mu starts at
    mu*(t0) and each sweep updates all t's, then mu. one_body_limit (default
    1-1e-8) is the closed bound 1+d >= 1-one_body_limit on one-body channels.
    The shift test requires the last sweep to move each mu by at most
    shift_atol + shift_rtol*(max(epsilon)-min(epsilon)).
    """

    max_sweeps: int = 500
    gradient_atol: float = 1e-11
    gradient_rtol: float = 1e-9
    amplitude_limit: Optional[float] = None
    max_abs_t: float = 30.0
    optimize_shift: bool = False
    one_body_limit: Optional[float] = None
    shift_atol: float = 0.0
    shift_rtol: float = 1e-8

    def __post_init__(self):
        _solver_integer(self.max_sweeps, "max_sweeps", 0)
        _solver_real(self.gradient_atol, "inner gradient_atol")
        _solver_real(self.gradient_rtol, "inner gradient_rtol")
        _solver_real(self.max_abs_t, "max_abs_t", positive=True)
        if self.amplitude_limit is not None:
            _solver_real(self.amplitude_limit, "amplitude_limit", positive=True)
            if self.amplitude_limit >= 1:
                raise ValueError("amplitude_limit must be < 1")
        if not isinstance(self.optimize_shift, (bool, np.bool_)):
            raise ValueError("optimize_shift must be boolean")
        if self.one_body_limit is not None:
            _solver_real(self.one_body_limit, "one_body_limit", positive=True)
            if self.one_body_limit >= 1:
                raise ValueError("one_body_limit must be < 1")
        _solver_real(self.shift_atol, "shift_atol")
        _solver_real(self.shift_rtol, "shift_rtol")


@dataclass(frozen=True)
class OptimizerOptions:
    """Solver-independent outer stopping, evaluation budget and validation."""

    max_iterations: int = 100
    max_evaluations: int = 1000
    gradient_atol: float = 1e-8
    gradient_rtol: float = 1e-6
    validate_final: bool = True
    verbosity: int = 0

    def __post_init__(self):
        _validate_optimizer_options(self)


def _validate_optimizer_options(options):
    _solver_integer(options.max_iterations, "max_iterations", 0)
    _solver_integer(options.max_evaluations, "max_evaluations", 1)
    _solver_integer(options.verbosity, "verbosity", 0)
    _solver_real(options.gradient_atol, "gradient_atol")
    _solver_real(options.gradient_rtol, "gradient_rtol")
    if not isinstance(options.validate_final, (bool,np.bool_)):
        raise ValueError("validate_final must be boolean")


LOSS_KINDS = ("raw", "weighted")
"""raw: sum_{i<j} ||[u_i,u_j]||_F^2.
weighted: sum_{i<j} a_i*a_j*||[u_i,u_j]||_F^2 with the stored (unnormalized) a_i."""


LINE_SEARCHES = ("armijo", "weak_wolfe")
"""armijo: sufficient-decrease backtracking (original behavior, default).
weak_wolfe: Lewis-Overton bracketing on sufficient decrease plus the weak
curvature condition g(theta+alpha*p).p >= wolfe_c2*g(theta).p; see README."""


def _validate_lbfgs_options(options):
    _solver_integer(options.memory, "memory", 1)
    _solver_integer(options.max_line_search, "max_line_search", 1)
    _solver_real(options.curvature_rtol, "curvature_rtol")
    for name in ("initial_step", "backtrack", "armijo", "min_step", "max_step_norm"):
        _solver_real(getattr(options,name), name, positive=True)
    if not 0 < options.backtrack < 1 or not 0 < options.armijo < 1:
        raise ValueError("backtrack and armijo must lie strictly between 0 and 1")
    if options.curvature_rtol >= 1:
        raise ValueError("curvature_rtol must be < 1")
    if options.min_step > options.initial_step:
        raise ValueError("min_step cannot exceed initial_step")
    if options.line_search not in LINE_SEARCHES:
        raise ValueError(f"line_search must be one of {LINE_SEARCHES}; got {options.line_search!r}")
    _solver_real(options.wolfe_c2, "wolfe_c2", positive=True)
    if not options.armijo < options.wolfe_c2 < 1:
        raise ValueError("wolfe_c2 must satisfy armijo < wolfe_c2 < 1")
    _solver_real(options.wolfe_max_step_norm, "wolfe_max_step_norm", positive=True)
    if options.wolfe_max_step_norm < options.max_step_norm:
        raise ValueError("wolfe_max_step_norm cannot be smaller than max_step_norm")


@dataclass(frozen=True)
class LBFGSOptions(OptimizerOptions):
    """L-BFGS curvature and line-search settings, extending common outer options.

    line_search="weak_wolfe" selects Lewis-Overton weak-Wolfe bracketing,
    intended for reduced objectives with kinks (see README). wolfe_c2 is the
    curvature constant and wolfe_max_step_norm caps the parameter displacement
    reached by step expansion. Both are ignored by the default "armijo" search.
    """

    memory: int = 10
    initial_step: float = 1.0
    backtrack: float = 0.5
    armijo: float = 1e-4
    max_line_search: int = 25
    min_step: float = 1e-12
    max_step_norm: float = 1.0
    curvature_rtol: float = 1e-10
    line_search: str = "armijo"
    wolfe_c2: float = 0.9
    wolfe_max_step_norm: float = math.pi

    def __post_init__(self):
        super().__post_init__()
        _validate_lbfgs_options(self)


@dataclass(frozen=True)
class OuterOptions:
    """Legacy L-BFGS configuration; new code can use LBFGSOptions.

    The original field order and constructor remain available. This legacy
    record is accepted by LBFGSOptimizer and optimize_gauge, and converted
    into LBFGSOptions. The solver-independent parent uses OptimizerOptions.
    """

    max_iterations: int = 100
    max_evaluations: int = 1000
    memory: int = 10
    gradient_atol: float = 1e-8
    gradient_rtol: float = 1e-6
    initial_step: float = 1.0
    backtrack: float = 0.5
    armijo: float = 1e-4
    max_line_search: int = 25
    min_step: float = 1e-12
    max_step_norm: float = 1.0
    curvature_rtol: float = 1e-10
    validate_final: bool = True
    verbosity: int = 0
    line_search: str = "armijo"
    wolfe_c2: float = 0.9
    wolfe_max_step_norm: float = math.pi

    def __post_init__(self):
        _validate_optimizer_options(self)
        _validate_lbfgs_options(self)


@dataclass(frozen=True)
class ExponentialEvaluation:
    """Matching parameters, reduced loss, gradient and converged inner snapshot."""

    theta: np.ndarray
    gradient_theta: np.ndarray
    gauge_gradient: GaugeGradient

    @property
    def loss(self):
        return self.gauge_gradient.loss

    @property
    def decomposition(self):
        return self.gauge_gradient.decomposition

    @property
    def O(self):
        return self.decomposition.two_body.gauge.O

    @property
    def gradient_norm(self):
        return float(np.max(np.abs(self.gradient_theta),initial=0.))

    @property
    def tangent_gradient_norm(self):
        return float(np.linalg.norm(self.gauge_gradient.tangent_gradient,"fro"))


@dataclass(frozen=True)
class OptimizerIteration:
    """Common accepted-state diagnostics with no solver-specific fields."""

    iteration: int
    loss: float
    gradient_norm: float
    tangent_gradient_norm: float
    step_size: float
    step_norm: float
    loss_decrease: float
    inner_sweeps: int
    inner_projected_gradient_norm: float
    line_search_trials: int
    shift: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class OuterIteration:
    """L-BFGS accepted-state diagnostics, retaining the step-6 record layout."""

    iteration: int
    loss: float
    gradient_norm: float
    tangent_gradient_norm: float
    step_size: float
    step_norm: float
    loss_decrease: float
    inner_sweeps: int
    inner_projected_gradient_norm: float
    line_search_trials: int
    curvature_update: bool
    memory_size: int
    acceptance: str = ""
    shift: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class OuterTrial:
    iteration: int
    attempt: int
    step_size: float
    accepted: bool
    loss: Optional[float]
    reason: str


@dataclass(frozen=True)
class OuterOptimizationResult:
    """Last ACCEPTED state, never the last rejected line-search evaluation."""

    evaluation: ExponentialEvaluation
    reference: np.ndarray
    initial_theta: np.ndarray
    initial_loss: float
    gradient_threshold: float
    tangent_gradient_threshold: float
    history: Tuple[OuterIteration, ...]
    trials: Tuple[OuterTrial, ...]
    converged: bool
    status: str
    message: str
    evaluations: int
    failed_evaluations: int
    cache_hits: int
    inner_sweeps: int
    validation_report: Optional[ValidationReport] = None

    @property
    def theta(self):
        return self.evaluation.theta

    @property
    def O(self):
        return self.evaluation.O

    @property
    def loss(self):
        return self.evaluation.loss

    @property
    def gradient_theta(self):
        return self.evaluation.gradient_theta

    @property
    def gradient_norm(self):
        return self.evaluation.gradient_norm

    @property
    def tangent_gradient_norm(self):
        return self.evaluation.tangent_gradient_norm

    @property
    def decomposition(self):
        return self.evaluation.decomposition

    @property
    def partition_result(self):
        return self.evaluation.gauge_gradient.inner_result

    @property
    def iterations(self):
        return self.history[-1].iteration
