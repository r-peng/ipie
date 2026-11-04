"""Data contracts for real, zero-shift Cholesky gauge optimization (step 1)."""

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
    """Future two-body channel schema, using the requested sor_base fields.

    bix indexes the FINAL Cholesky eigenbasis; spin 0=up, 1=down.
    a_i is an UNNORMALIZED positive coefficient in Lambda*I-H, not an H
    coefficient or sampling probability. d contains the actual branch signs.
    Step 1 defines this contract but does not generate channel lists.
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

