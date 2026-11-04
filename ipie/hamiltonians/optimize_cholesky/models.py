"""Data contracts for real, zero-shift Cholesky gauge optimization."""

from dataclasses import asdict, dataclass
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
    """One-body channels and the jointly diagonalized original h."""

    X: np.ndarray
    eigenvalues: np.ndarray
    tau1: float
    channels: Tuple[OneBodyChannel, ...]

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


