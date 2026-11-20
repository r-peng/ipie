"""Pair-based references for the tests (moved out of production in step 3).

These are the 3680f29 single-pair kernel, its derivatives and the literal
dense-matrix loss. Production computes losses in mode space (loss.py); the
tests compare it against these independent formulas, against the frozen
oracle (tests/reference) and against sor_base.compute_total_commutativity.
"""

import numpy as np

from ipie.hamiltonians.optimize_cholesky.channels import get_rotation_matrix
from ipie.hamiltonians.optimize_cholesky.loss import _weights, loss_kind_name
from ipie.hamiltonians.optimize_cholesky.models import CommutatorLoss
from ipie.hamiltonians.optimize_cholesky.problem import _real_array


def _readonly(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def commutator_norm2(d, e, overlap):
    """Squared Frobenius norm of [V diag(d) V.T, Z diag(e) Z.T].

    V and Z must each have orthonormal columns; overlap=V.T@Z. The formula is
    2*Tr(D^2 C E^2 C.T) - 2*Tr((D C E C.T) @ (D C E C.T)).
    The matrix square in the second term is NOT an elementwise square or a
    squared Frobenius norm. Actual channel blocks have at most two columns.
    Empty blocks give zero. Only negative roundoff relative to the two terms
    is clamped; materially negative values raise FloatingPointError.
    """
    d = _real_array(d, "d")
    e = _real_array(e, "e")
    C = _real_array(overlap, "overlap")
    if d.ndim != 1 or e.ndim != 1 or C.shape != (d.size, e.size):
        raise ValueError("Expected vectors d,e and overlap.shape=(len(d),len(e))")
    if not d.size or not e.size:
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        first = 2.0 * np.sum((d[:,None] * C * e[None,:])**2)
        product = ((d[:,None] * C) * e[None,:]) @ C.T
        second = 2.0 * np.sum(product * product.T)
        value = float(first - second)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite overlap commutator norm")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(float(first)), abs(float(second)))
    if value < -tolerance:
        raise FloatingPointError(
            f"Squared norm {value:.3e} is materially negative (roundoff bound {tolerance:.3e}); "
            "check basis orthogonality and overlap construction"
        )
    return max(0.0, value)


def commutator_norm2_derivatives(d, e, overlap):
    """Return (norm2, d_norm2/dd, d_norm2/de, d_norm2/dC).

    Differentiate the polynomial norm formula, not its roundoff clamp. The
    overlap pullback must include BOTH bases. Blocks may contain zero legs.
    """
    d, e, C = (_real_array(x, n) for x, n in ((d,"d"),(e,"e"),(overlap,"overlap")))
    value = commutator_norm2(d, e, C)
    R = (C*e[None,:]) @ C.T
    S = C.T @ (d[:,None]*C)
    with np.errstate(over="raise", invalid="raise"):
        gd = 4*d*((C*C) @ (e*e))-4*((R*R) @ d)
        ge = 4*e*((C*C).T @ (d*d))-4*((S*S) @ e)
        gC = 4*(d*d)[:,None]*C*(e*e)[None,:]
        gC -= 8*((d[:,None]*R)*d[None,:]) @ (C*e[None,:])
    if not all(np.isfinite(x).all() for x in (gd,ge,gC)):
        raise FloatingPointError("Nonfinite overlap-norm derivative")
    return value, _readonly(gd), _readonly(ge), _readonly(gC)


def _accumulate(pair_norm2, weights, normalize_weights, coefficients):
    """One pass over all i<j pairs.

    weights are the (possibly normalized) diagnostic weights; coefficients are
    the stored unnormalized a_i used for L_w. Sums are sequential, as in
    sor_base; with normalize_weights=False the weighted and unnormalized sums
    are therefore bitwise identical.
    """
    if coefficients.shape != weights.shape:
        raise ValueError("coefficients and weights must have one entry per channel")
    raw = weighted = weight = unnormalized = 0.0
    count = weights.size
    for i in range(count):
        for j in range(i+1, count):
            pair = pair_norm2(i, j)
            wij = float(weights[i] * weights[j])
            raw += pair
            weighted += wij * pair
            unnormalized += float(coefficients[i] * coefficients[j]) * pair
            # Include ALL pairs, even when their commutator is structurally zero.
            weight += wij
    if not all(np.isfinite(value) for value in (raw, weighted, weight, unnormalized)):
        raise FloatingPointError("Overflow in total commutator diagnostics")
    return CommutatorLoss(raw, weighted, weight, weighted/weight if weight > 0 else None,
                          count*(count-1)//2, bool(normalize_weights), unnormalized)


def dense_commutator_loss(decomposition, *, normalize_weights=True):
    """Literal reference matching sor_base, using full n*n spin matrices.

    normalize_weights=True uses a_i/denom, as after parse_decomposition().
    False reproduces the same formulas with raw coefficients. Neither option
    alters the channels, comm2 or weighted_comm2_unnormalized.
    """
    weights = _weights(decomposition, normalize_weights)
    coefficients = _weights(decomposition, False)
    matrices = [get_rotation_matrix(decomposition, i) for i in range(weights.size)]

    def pair_norm2(i, j):
        result = 0.0
        for left, right in zip(matrices[i], matrices[j]):
            if left is None or right is None:
                continue
            comm = np.dot(left, right) - np.dot(right, left)
            result += float(np.linalg.norm(comm, "fro")**2)
        return result

    return _accumulate(pair_norm2, weights, normalize_weights, coefficients)


def dense_pair(decomposition, i, j):
    """||[u_i, u_j]||_F^2 of two channels (combined one-body-first index), from dense matrices."""
    result = 0.0
    for left, right in zip(get_rotation_matrix(decomposition, i), get_rotation_matrix(decomposition, j)):
        if left is not None and right is not None:
            result += np.linalg.norm(left @ right - right @ left, "fro")**2
    return float(result)


def dense_objective_loss(decomposition, loss_kind="weighted"):
    """objective_loss from literal dense matrices (raw: comm2; weighted: unnormalized L_w)."""
    result = dense_commutator_loss(decomposition, normalize_weights=False)
    return result.comm2 if loss_kind_name(loss_kind) == "raw" else result.weighted_comm2_unnormalized


def dense_raw_loss(decomposition):
    return dense_objective_loss(decomposition, "raw")
