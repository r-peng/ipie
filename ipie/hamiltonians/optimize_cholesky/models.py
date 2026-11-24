"""Data contracts for real Cholesky gauge optimization.

Decompositions store their channels as read-only arrays (SourceTable,
ChannelTable, OneBodyTable); the small value records Channel, OneBodyChannel,
DensitySource and PartitionCoefficients are built from those arrays only when
an element is requested. Validation happens where data enters (problem
preparation, builders, channels.with_channels, public functions that receive a
caller-built decomposition), never in the record views.
"""

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


SOURCE_KINDS = ("two_branch", "spin_quartet")
"""Source kinds; SourceTable.kind stores the index into this tuple."""


def _frozen(value, dtype, shape=None):
    """An owned, read-only array copy."""
    array = np.array(value, dtype=dtype, copy=True)
    if shape is not None:
        array = array.reshape(shape)
    array.setflags(write=False)
    return array


class _RecordSequence:
    """Read-only sequence of value records built on demand from array rows.

    Integer indexing returns one record, slicing a tuple of records, iteration
    yields records. Equality with another sequence compares the records.
    """

    def _record(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self._record(i) for i in range(*index.indices(len(self))))
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
            raise TypeError("index must be an integer or a slice")
        position = int(index)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("index out of range")
        return self._record(position)

    def __iter__(self):
        return (self._record(i) for i in range(len(self)))

    def __eq__(self, other):
        try:
            if len(other) != len(self):
                return False
        except TypeError:
            return NotImplemented
        return all(a == b for a, b in zip(self, other))

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None


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


@dataclass(frozen=True, eq=False)
class SourceTable(_RecordSequence):
    """The fixed source topology of a problem, with its canonical channel layout.

    Row i is source i (one partition coordinate t_i): factor bix, modes
    (alpha, beta), kind (index into SOURCE_KINDS) and spins ((-1, -1) for a
    spin quartet, whose DensitySource.spins is None). The canonical layout
    lists the channels every source emits, in order: source slices[i] =
    [start, stop) and, per channel, its source, branch sign, spins and branch
    weight factor (Green weight = factor/tau2). Built by
    channels.enumerate_sources; element access returns DensitySource.
    """

    bix: np.ndarray
    modes: np.ndarray
    kind: np.ndarray
    spins: np.ndarray
    slices: np.ndarray
    channel_source: np.ndarray
    channel_sign: np.ndarray
    channel_spins: np.ndarray
    channel_weight: np.ndarray

    def __len__(self):
        return int(self.bix.shape[0])

    @property
    def nchannels(self):
        return int(self.channel_source.shape[0])

    def _record(self, i):
        spins = None if self.spins[i, 0] < 0 else (int(self.spins[i, 0]), int(self.spins[i, 1]))
        return DensitySource(int(self.bix[i]), (int(self.modes[i, 0]), int(self.modes[i, 1])),
                             SOURCE_KINDS[int(self.kind[i])], spins)

    def __eq__(self, other):
        if isinstance(other, SourceTable):
            return all(np.array_equal(getattr(self, name), getattr(other, name))
                       for name in ("bix", "modes", "kind", "spins"))
        return super().__eq__(other)


@dataclass(frozen=True, eq=False)
class ChannelTable(_RecordSequence):
    """Two-body channels as arrays: bix (T,), spins (T,2), modes (T,2), d (T,2), a (T,).

    a holds the UNNORMALIZED positive Green weights. Element access returns a
    Channel. Tables are built and checked by channels.py; this class does no
    validation of its own.
    """

    bix: np.ndarray
    spins: np.ndarray
    modes: np.ndarray
    d: np.ndarray
    a: np.ndarray

    def __len__(self):
        return int(self.bix.shape[0])

    def _record(self, i):
        return Channel(int(self.bix[i]), (int(self.spins[i, 0]), int(self.spins[i, 1])), float(self.a[i]),
                       (int(self.modes[i, 0]), int(self.modes[i, 1])), (float(self.d[i, 0]), float(self.d[i, 1])))

    def __eq__(self, other):
        if isinstance(other, ChannelTable):
            return all(np.array_equal(getattr(self, name), getattr(other, name))
                       for name in ("bix", "spins", "modes", "d", "a"))
        return super().__eq__(other)


@dataclass(frozen=True, eq=False)
class OneBodyTable(_RecordSequence):
    """One-body channels as arrays: spin (K,), mode (K,), d (K,), a (K,).

    Element access returns a OneBodyChannel.
    """

    spin: np.ndarray
    mode: np.ndarray
    d: np.ndarray
    a: np.ndarray

    def __len__(self):
        return int(self.spin.shape[0])

    def _record(self, i):
        return OneBodyChannel(int(self.spin[i]), float(self.a[i]), int(self.mode[i]), float(self.d[i]))

    def __eq__(self, other):
        if isinstance(other, OneBodyTable):
            return all(np.array_equal(getattr(self, name), getattr(other, name))
                       for name in ("spin", "mode", "d", "a"))
        return super().__eq__(other)


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

    table holds the channels. canonical=True means table is exactly the
    canonical layout of sources at amplitudes (what the builders produce);
    the partition solver and the gauge gradients require it.
    channels.with_channels makes non-canonical decompositions (hand-made or
    filtered channel lists) for the loss kernels, rotation matrices and tests.
    source_slices[i] = (start, stop) refers to channels[start:stop] of a
    canonical table. All arrays are owned, read-only snapshots.

    cholesky_shift: None, or (r, 2) shifts mu[gamma, sigma] of the rotated
    factors, so the channels use lambda[sigma, gamma, alpha] =
    ell[gamma, alpha] - mu[gamma, sigma] (see FixedDecomposition.cholesky_energy).
    amplitudes holds (d_left, d_right) per source, shape (count, 2), unless the
    shift differs between spins; then it is (count, 2, 2) = [source, leg, spin]
    and split is True. A tied or absent shift uses today's (count, 2) layout.
    """

    gauge: GaugeState
    policy: str
    nelec: Optional[Tuple[int, int]]
    tau2: float
    partition: str
    amplitude_bound: float
    sources: SourceTable
    t: np.ndarray
    amplitudes: np.ndarray
    table: ChannelTable
    canonical: bool = True
    cholesky_shift: Optional[np.ndarray] = None

    @property
    def split(self):
        """True when the amplitudes are stored per spin, shape (count, 2, 2)."""
        return np.ndim(self.amplitudes) == 3

    @property
    def spin_eigenvalues(self):
        """(2, r, n) shifted factor eigenvalues lambda[sigma] = ell - mu[:, sigma]."""
        ell = np.asarray(self.gauge.eigenvalues, dtype=np.float64)
        if self.cholesky_shift is None:
            return np.stack([ell, ell])
        return ell[None, :, :] - np.asarray(self.cholesky_shift, dtype=np.float64).T[:, :, None]

    @property
    def map_eigenvalues(self):
        """The eigenvalues the partition map uses: (r, n) unless split, then (2, r, n)."""
        if self.split:
            return self.spin_eigenvalues
        if self.cholesky_shift is None:
            return self.gauge.eigenvalues
        return np.asarray(self.gauge.eigenvalues, dtype=np.float64) \
            - np.asarray(self.cholesky_shift, dtype=np.float64)[:, 0][:, None]

    @property
    def leg_amplitudes(self):
        """(count, 2, 2) amplitudes [source, leg, spin]; equal spins unless split."""
        if self.split:
            return self.amplitudes
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        return np.repeat(amplitudes[:, :, None], 2, axis=2)

    @property
    def channels(self):
        """The channels as a read-only sequence of Channel records."""
        return self.table

    @property
    def source_slices(self):
        """(number_of_sources, 2) channel ranges of the canonical layout."""
        return self.sources.slices

    @property
    def weight_sum(self):
        return sum(self.table.a.tolist(), 0.0)


@dataclass(frozen=True)
class OneBodyDecomposition:
    """One-body channels and the jointly diagonalized original h.

    shift=(mu_up, mu_down) gives d[k,sigma]=(mu_sigma-epsilon_k)*tau1, i.e. the
    channels represent sum_k (epsilon_k-mu_sigma) n_{k,sigma}. The remainder
    sum_sigma mu_sigma N_sigma is a constant only on a fixed (N_up, N_down)
    sector; see FixedDecomposition.shift_energy.

    Without a Cholesky shift the matrix is the bare h, X is (n, n) and
    eigenvalues (n,). With one it is h_sigma = h + sum_g (m_g - mu[g,sigma]) L_g
    (channels.one_body_matrices), and cholesky_shift records the (r, 2) shift.
    When h_up and h_down differ, X is (2, n, n) and eigenvalues (2, n), one
    eigensystem per spin (split is True).
    """

    X: np.ndarray
    eigenvalues: np.ndarray
    tau1: float
    table: OneBodyTable
    shift: Tuple[float, float] = (0.0, 0.0)
    cholesky_shift: Optional[np.ndarray] = None

    @property
    def split(self):
        """True when each spin has its own eigensystem, X (2, n, n)."""
        return np.ndim(self.X) == 3

    def spin_X(self, spin):
        """Eigenvectors (columns) of spin's one-body matrix."""
        return self.X[spin] if self.split else self.X

    def spin_eigenvalues(self, spin):
        """Eigenvalues of spin's one-body matrix, ascending."""
        return self.eigenvalues[spin] if self.split else self.eigenvalues

    @property
    def mode_eigenvalues(self):
        """(2, n) eigenvalues eps[sigma, k], equal rows unless split."""
        eps = np.asarray(self.eigenvalues, dtype=np.float64)
        return eps if self.split else np.stack([eps, eps])

    @property
    def channels(self):
        """The channels as a read-only sequence of OneBodyChannel records."""
        return self.table

    @property
    def spins(self):
        """Spins that carry one-body channels: (0,) for aa_only, else (0, 1)."""
        return tuple(sorted(set(self.table.spin.tolist())))

    @property
    def weight_sum(self):
        return sum(self.table.a.tolist(), 0.0)


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

    def __post_init__(self):
        a, b = self.one_body.cholesky_shift, self.two_body.cholesky_shift
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            raise ValueError("The one-body and two-body channels were built for different Cholesky shifts")

    @property
    def cholesky_shift(self):
        """None, or the (r, 2) shifts mu[gamma, sigma] of the rotated factors."""
        return self.two_body.cholesky_shift

    @property
    def cholesky_energy(self):
        """E_c = 1/2 sum_g (sum_s N_s mu[g,s]^2 - m_g^2), m_g = sum_s N_s mu[g,s].

        On the (N_up, N_down) sector H = H_channels + E_c, where H_channels uses
        the shifted factors L_g - mu[g,s] I and h_s = h + sum_g (m_g - mu[g,s]) L_g.
        A tied shift gives -1/2 N(N-1) sum_g mu_g^2.
        """
        mu = self.cholesky_shift
        if mu is None:
            return 0.0
        nelec = self.two_body.nelec
        if nelec is None:
            raise ValueError("A Cholesky shift requires particle numbers (nelec)")
        counts = np.asarray(nelec, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        m = mu @ counts
        return float(0.5*np.sum((mu*mu) @ counts - m*m))

    @property
    def nterms(self):
        return len(self.one_body.table) + len(self.two_body.table)

    @property
    def denom(self):
        return self.one_body.weight_sum + self.two_body.weight_sum

    @property
    def coefficients(self):
        return np.concatenate([self.one_body.table.a, self.two_body.table.a]).astype(np.float64)

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
        """Lambda without E0: denom + sum_sigma mu_sigma N_sigma + cholesky_energy.

        On the (N_up, N_down) sector, (E0+lambda_electronic)*I - H_tot equals
        sum_i a_i U_i. With zero shifts this is denom.
        """
        if self.cholesky_shift is None:
            return float(self.denom + self.shift_energy)
        return float(self.denom + self.shift_energy + self.cholesky_energy)


LOSS_KINDS = ("raw", "weighted")
"""raw: sum_{i<j} ||[u_i,u_j]||_F^2.
weighted: sum_{i<j} a_i*a_j*||[u_i,u_j]||_F^2 with the stored (unnormalized) a_i."""


@dataclass(frozen=True)
class CommutatorLoss:
    """Squared single-particle losses, matching sor_base's printed diagnostics.

    Field                          Value
    comm2                          L_raw = sum_{i<j} ||[u_i,u_j]||_F^2
    weighted_comm2                 sum_{i<j} w_i w_j ||[u_i,u_j]||_F^2, w_i = a_i/denom
                                   if weights_normalized else a_i (sor_base convention)
    weight                         sum_{i<j} w_i w_j over ALL pairs (same w)
    weighted_mean                  weighted_comm2/weight; None if weight=0
    weighted_comm2_unnormalized    L_w = sum_{i<j} a_i a_j ||[u_i,u_j]||_F^2, always with
                                   the stored a_i, independent of weights_normalized

    The normalized weighted_comm2 equals L_w/denom**2; weighted_mean is the
    same in both normalizations. weighted_mean=None means weight=0, e.g. a
    one-channel decomposition with no unordered pairs. The optimized
    objective alone comes from loss.objective_loss.
    """

    comm2: float
    weighted_comm2: float
    weight: float
    weighted_mean: Optional[float]
    pair_count: int
    weights_normalized: bool
    weighted_comm2_unnormalized: Optional[float] = None

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


@dataclass(frozen=True, eq=False)
class PartitionProfiles(_RecordSequence):
    """The coordinate profiles of all sources as arrays (t, C, left_term, right_term).

    Element i is the PartitionCoefficients of source i, built on access.
    """

    t: np.ndarray
    C: np.ndarray
    left_term: np.ndarray
    right_term: np.ndarray

    def __len__(self):
        return int(self.t.shape[0])

    def _record(self, i):
        return PartitionCoefficients(i, float(self.t[i]), float(self.C[i]),
                                     float(self.left_term[i]), float(self.right_term[i]))

    @property
    def gradient(self):
        return 2.0*(self.left_term-self.right_term)


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
    coefficients: PartitionProfiles
    gradient: np.ndarray
    projected_gradient: np.ndarray
    gradient_threshold: float
    history: Tuple[PartitionSweep, ...]
    converged: bool
    status: str
    amplitude_limit: Optional[float] = None
    max_abs_t: float = 30.0
    loss_kind: str = "weighted"
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

    cholesky_gradient: None, or dL/d of the Cholesky shift mu[gamma, sigma] at
    the same point, with the same t (fixed_t), z (fixed_z) or envelope
    (relaxed) convention: (r,) for a tied shift (one mu_gamma for every spin
    with channels; for aa_only that is mu_up), (r, 2) per spin. With a shift,
    gradient_O includes the gauge dependence of the one-body matrices h_sigma.
    (PartitionOptimizationResult.shift_gradient is the one-body shift's.)
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
    inner_result: Optional[PartitionOptimizationResult] = None
    cholesky_gradient: Optional[np.ndarray] = None


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
class ExponentialEvaluation:
    """Matching parameters, reduced loss, gradient and converged inner snapshot.

    theta and gradient_theta are the gauge angles and their gradient. When
    the outer objective also optimizes the Cholesky shift, scaled_shift holds
    its coordinates mu/s and gradient_scaled_shift their gradient; the solver
    then works on coordinates = (theta, mu/s), or mu/s alone when
    theta_frozen. Without them coordinates is theta itself.
    """

    theta: np.ndarray
    gradient_theta: np.ndarray
    gauge_gradient: GaugeGradient
    scaled_shift: Optional[np.ndarray] = None
    gradient_scaled_shift: Optional[np.ndarray] = None
    theta_frozen: bool = False

    @property
    def coordinates(self):
        """The solver's coordinate vector."""
        if self.scaled_shift is None:
            return self.theta
        if self.theta_frozen:
            return self.scaled_shift
        return np.concatenate([self.theta, self.scaled_shift])

    @property
    def gradient(self):
        """The gradient with respect to coordinates."""
        if self.scaled_shift is None:
            return self.gradient_theta
        if self.theta_frozen:
            return self.gradient_scaled_shift
        return np.concatenate([self.gradient_theta, self.gradient_scaled_shift])

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
    def cholesky_shift(self):
        """The (r, 2) Cholesky shift of this point, or None (also for a decomposition without one)."""
        return getattr(self.decomposition, "cholesky_shift", None)

    @property
    def gradient_norm(self):
        """max |gradient| over the solver coordinates (gauge angles and scaled shift)."""
        return float(np.max(np.abs(self.gradient),initial=0.))

    @property
    def tangent_gradient_norm(self):
        """Frobenius norm of the physical tangent gradient, joined (2-norm) with the scaled shift gradient.

        The tangent part is left out when the gauge is frozen.
        """
        tangent = float(np.linalg.norm(self.gauge_gradient.tangent_gradient,"fro"))
        if self.scaled_shift is None:
            return tangent
        shift = float(np.linalg.norm(self.gradient_scaled_shift))
        return shift if self.theta_frozen else math.hypot(tangent, shift)


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
class OuterIteration(OptimizerIteration):
    """L-BFGS accepted-state diagnostics: the common record plus L-BFGS fields."""

    curvature_update: bool = False
    memory_size: int = 0
    acceptance: str = ""


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
    """Last ACCEPTED state, never the last rejected line-search evaluation.

    theta and initial_theta are gauge angles. When the Cholesky shift was
    optimized, initial_cholesky_shift is the start, cholesky_scale the fixed
    s of its coordinates mu/s, and cholesky_shift the final (r, 2) shift.
    """

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
    initial_cholesky_shift: Optional[np.ndarray] = None
    cholesky_scale: Optional[np.ndarray] = None

    @property
    def theta(self):
        return self.evaluation.theta

    @property
    def cholesky_shift(self):
        """The final (r, 2) Cholesky shift, or None."""
        return self.evaluation.cholesky_shift

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
