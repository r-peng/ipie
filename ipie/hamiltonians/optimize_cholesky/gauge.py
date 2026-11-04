"""ERI-preserving orthogonal gauges and Cholesky eigensystems (NumPy reference)."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import expm

from ipie.hamiltonians.optimize_cholesky.models import GaugeState
from ipie.hamiltonians.optimize_cholesky.validation import (
    _factor_array, _gauge_array, _options, _real_array, validate_gauge_state,
)


def _dimension(nchol):
    if isinstance(nchol, (bool, np.bool_)) or not isinstance(nchol, Integral) or nchol < 1:
        raise ValueError("nchol must be a positive integer")
    return int(nchol)


def skew_from_parameters(theta, nchol):
    """Pack theta into upper triangle: (0,1),(0,2),...,(1,2),... .

    S[p,q]=theta[k] for p<q, S[q,p]=-theta[k], and diagonal zero.
    """
    r = _dimension(nchol)
    theta = _real_array(theta, "theta")
    count = r * (r - 1) // 2
    if theta.shape != (count,):
        raise ValueError(f"theta must be a length-{count} vector for {r} factors")
    S = np.zeros((r, r), dtype=np.float64)
    rows, cols = np.triu_indices(r, 1)
    S[rows, cols] = theta
    S[cols, rows] = -theta
    return S


def exponential_gauge(theta, nchol, reference=None):
    """Return O=expm(S(theta)) @ reference (reference defaults to identity).

    Exponential coordinates stay in the determinant component of reference.
    A reflection reference is permitted, including for r=1.
    """
    r = _dimension(nchol)
    reference = np.eye(r) if reference is None else _gauge_array(reference, r, _options(None))
    result = expm(skew_from_parameters(theta, r)) @ reference
    return _gauge_array(result, r, _options(None))


def givens_gauge(nchol, rotations, reference=None):
    """Left-multiply reference by successive (p,q,angle) plane rotations.

    Require 0<=p<q<r. The active block is [[cos,sin],[-sin,cos]].
    In list order, O <- G(p,q,angle) @ O; the last rotation is leftmost.
    Only the two affected rows are updated; no dense G is constructed.
    """
    r = _dimension(nchol)
    result = np.eye(r) if reference is None else _gauge_array(reference, r, _options(None)).copy()
    for rotation in rotations:
        try:
            p, q, angle = rotation
        except (TypeError, ValueError) as exc:
            raise ValueError("Each Givens rotation must be (p,q,angle)") from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (p,q)):
            raise ValueError("Givens indices must be integers")
        if not 0 <= p < q < r:
            raise ValueError(f"Require 0 <= p < q < {r}; got {(p,q)}")
        if not isinstance(angle, Real) or not np.isfinite(angle):
            raise ValueError("Givens angle must be finite and real (radians)")
        cosine, sine = np.cos(angle), np.sin(angle)
        row_p, row_q = result[p].copy(), result[q].copy()
        result[p] = cosine * row_p + sine * row_q
        result[q] = -sine * row_p + cosine * row_q
    return _gauge_array(result, r, _options(None))


def rotate_cholesky(chol, O, options=None):
    """Return rotated[eta,p,r] = sum_gamma O[eta,gamma]*chol[gamma,p,r]."""
    options = _options(options)
    chol = _factor_array(chol, options)
    O = _gauge_array(O, chol.shape[0], options)
    return (O @ chol.reshape(chol.shape[0], -1)).reshape(chol.shape)


def diagonalize_factors(chol, options=None):
    """Return (ell[gamma,alpha], X[gamma,p,alpha]) with ascending ell.

    All modes are retained. No clipping, low-rank truncation, symmetrization,
    or perturbation of degeneracies is performed. Sign/basis choices within
    degenerate eigenspaces are those of numpy.linalg.eigh.
    """
    chol = _factor_array(chol, _options(options))
    return np.linalg.eigh(chol)


def build_gauge_state(data, O=None, options=None):
    """Build and fully validate a step-1 state against supplied exact ERIs.

    This full-validation function is intended for initial/reference checks,
    not for every future optimizer loss evaluation. Low-level rotation/eigh
    functions above do not reconstruct the ERI tensor.
    """
    options = _options(options)
    chol = _factor_array(data.chol, options)
    O = np.eye(chol.shape[0]) if O is None else O
    O = _gauge_array(O, chol.shape[0], options)
    rotated = rotate_cholesky(chol, O, options)
    ell, X = diagonalize_factors(rotated, options)
    state = GaugeState(O.copy(), rotated, ell, X)
    report = validate_gauge_state(data, state, options)
    # Own immutable snapshots; never mutate the caller's supplied arrays.
    for array in (state.O, state.chol, state.eigenvalues, state.X):
        array.setflags(write=False)
    return GaugeState(state.O, state.chol, state.eigenvalues, state.X, report)

