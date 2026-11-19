"""Reference squared commutator loss: dense matrices versus small overlaps.

For u=diag(I+A_up,I+A_down), squared spin norms are ADDED, never square
rooted. Two objectives are supported (loss_kind):

  raw       L_raw = sum_{i<j} ||[u_i,u_j]||_F^2
  weighted  L_w   = sum_{i<j} a_i a_j ||[u_i,u_j]||_F^2   (unnormalized a_i)

Because [u_i,u_j]=[A_i,A_j] is bilinear, a_i a_j ||[A_i,A_j]||^2 equals
||[sqrt(a_i) A_i, sqrt(a_j) A_j]||^2: L_w is L_raw evaluated on blocks whose
merged amplitudes kappa are scaled by sqrt(a_i). Every kernel is therefore
shared; the optimizer layers use those scaled blocks.

The loss kind is chosen in one place: objective_loss(..., loss_kind=) and
OverlapEvaluator(..., loss_kind=).total() return the objective, both kinds
through the same scaled-block sum. commutator_loss is the separate diagnostic
record: it evaluates unscaled pair norms once and reports L_raw, the
sor_base-compatible weighted fields (normalized by denom by default) and the
unnormalized L_w together. Its weighted diagnostics match
sor_base.compute_total_commutativity when supplied the same actual channels and
the same normalized coefficients. This module is a real float64 NumPy
reference, not an autodiff implementation.
"""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .channels import get_rotation_matrix
from .models import LOSS_KINDS, CommutatorLoss, FixedDecomposition
from .problem import _real_array


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


def _weights(decomposition, normalize_weights):
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise TypeError("normalize_weights must be boolean")
    weights = decomposition.coefficients
    if weights.shape != (decomposition.nterms,):
        raise ValueError("Coefficient count does not match channel count")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("All channel coefficients must be finite and strictly positive")
    if normalize_weights and weights.size:
        total = decomposition.denom
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Combined coefficient sum must be finite and positive")
        weights = weights / total
    return weights


def loss_kind_name(loss_kind):
    if not isinstance(loss_kind, str) or loss_kind not in LOSS_KINDS:
        raise ValueError(f"loss_kind must be one of {LOSS_KINDS}; got {loss_kind!r}")
    return loss_kind


def pair_scales(decomposition, loss_kind="weighted"):
    """Per-channel s_i with pair weight s_i*s_j (combined one-body-first order).

    raw: s_i=1. weighted: s_i=a_i, the stored UNNORMALIZED Green weights.
    """
    loss_kind = loss_kind_name(loss_kind)
    weights = _weights(decomposition, False)
    if loss_kind == "raw":
        return np.ones_like(weights)
    return weights


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


@dataclass(frozen=True)
class _SpinBlock:
    basis: int
    modes: tuple
    kappa: tuple


class OverlapEvaluator:
    """Low-rank evaluator and lazy basis-overlap cache for one FIXED state.

    Basis -1 is the jointly diagonalized h basis; nonnegative basis ids are
    Cholesky bix values. Spin blocks use at most two orthonormal columns.
    Channel identities/multiplicities are preserved, even for identity terms.
    Build a new evaluator after changing bases, channels, or amplitudes.
    """

    def __init__(self, decomposition, *, loss_kind="weighted", check=True):
        """check=False skips the basis and channel checks for a decomposition
        that GaugeEvaluator built itself; numbers are identical either way."""
        self.decomposition = decomposition
        self.loss_kind = loss_kind_name(loss_kind)
        self.scales = pair_scales(decomposition, self.loss_kind)
        self.root_scales = np.sqrt(self.scales)
        self.root_scales.setflags(write=False)
        if check:
            X = _real_array(decomposition.two_body.gauge.X, "Cholesky bases")
            if X.ndim != 3 or X.shape[0] < 1 or X.shape[1] != X.shape[2] or X.shape[1] < 1:
                raise ValueError("Cholesky bases must have shape (r,n,n)")
        else:
            X = decomposition.two_body.gauge.X
        self.nbasis = X.shape[1]
        basis = self._basis if check else (lambda array, name: array)
        self._bases = {-1: basis(decomposition.one_body.X, "one-body basis")}
        for bix, factor in enumerate(X):
            self._bases[bix] = basis(factor, f"Cholesky basis {bix}")
        self._overlaps = {}
        blocks = []
        for index, channel in enumerate(decomposition.one_body.channels):
            if check:
                self._leg(channel.spin, channel.mode, channel.d)
            spin_blocks = [None, None]
            spin_blocks[channel.spin] = self._block(-1, ((channel.mode,channel.d),),
                                                    self.root_scales[index])
            blocks.append(tuple(spin_blocks))
        offset = len(decomposition.one_body.channels)
        for index, channel in enumerate(decomposition.two_body.channels):
            if check and (channel.bix not in self._bases or channel.bix < 0):
                raise ValueError("Channel Cholesky basis index is out of range")
            legs = [[], []]
            for spin, mode, d in zip(channel.spins, channel.modes, channel.d):
                if check:
                    self._leg(spin, mode, d)
                legs[spin].append((mode,d))
            blocks.append(tuple(self._block(channel.bix, spin_legs, self.root_scales[offset+index])
                                for spin_legs in legs))
        self.blocks = tuple(blocks)

    def _basis(self, array, name):
        basis = _real_array(array, name)
        if basis.shape != (self.nbasis,self.nbasis):
            raise ValueError(f"{name} must have shape {(self.nbasis,self.nbasis)}")
        if not np.allclose(basis.T @ basis, np.eye(self.nbasis), atol=1e-12, rtol=1e-10):
            raise ValueError(f"{name} must have orthonormal columns for the overlap formula")
        basis = basis.copy()
        basis.setflags(write=False)
        return basis

    def _leg(self, spin, mode, d):
        for value, name in ((spin,"spin"), (mode,"mode")):
            if isinstance(value, (bool,np.bool_)) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if spin not in (0,1) or not 0 <= mode < self.nbasis:
            raise ValueError("Channel spin/mode index is out of range")
        if not isinstance(d, Real) or not np.isfinite(d):
            raise ValueError("Channel amplitudes must be finite and real")

    @staticmethod
    def _block(basis, legs, scale=1.0):
        coefficients = {}
        for mode, d in legs:
            old = coefficients.get(mode, 0.0)
            coefficients[mode] = old + d + old*d
        # Distinct modes are orthogonal. Repeated mode: (1+old P)(1+d P).
        # The loss scale multiplies the MERGED amplitude (the matrix A), never d.
        active = [(mode,float(kappa)) for mode,kappa in coefficients.items() if kappa != 0]
        if not active:
            return None
        scale = float(scale)
        return _SpinBlock(basis, tuple(p for p,_ in active),
                          tuple(d if scale == 1.0 else scale*d for _,d in active))

    def _overlap(self, left, right):
        key = (min(left,right), max(left,right))
        if key not in self._overlaps:
            self._overlaps[key] = self._bases[key[0]].T @ self._bases[key[1]]
        overlap = self._overlaps[key]
        return overlap if left <= right else overlap.T

    @property
    def overlap_cache_size(self):
        return len(self._overlaps)

    def pair_norm2(self, i, j):
        """Squared full-spin norm for one pair, including arbitrary spin order."""
        for index in (i,j):
            if isinstance(index, (bool,np.bool_)) or not isinstance(index, Integral):
                raise TypeError("Channel index must be an integer")
            if not 0 <= index < len(self.blocks):
                raise IndexError("Channel index is out of range")
        result = 0.0
        for left, right in zip(self.blocks[i], self.blocks[j]):
            if left is None or right is None or left.basis == right.basis:
                continue
            C = self._overlap(left.basis, right.basis)[np.ix_(left.modes, right.modes)]
            result += commutator_norm2(left.kappa, right.kappa, C)
        return result

    def total(self):
        """The loss_kind objective: fsum of all (scaled) pair norms."""
        value = math.fsum(self.pair_norm2(i, j) for i in range(len(self.blocks))
                          for j in range(i+1, len(self.blocks)))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite commutator loss")
        return value


def overlap_commutator_loss(decomposition, *, normalize_weights=True):
    """Cached-overlap diagnostics; no dense channel or commutator matrices."""
    evaluator = OverlapEvaluator(decomposition, loss_kind="raw")
    return _accumulate(evaluator.pair_norm2, _weights(decomposition, normalize_weights),
                       normalize_weights, _weights(decomposition, False))


def commutator_loss(decomposition, *, method="overlap", normalize_weights=True):
    """Every loss at once: L_raw, unnormalized L_w and sor_base-compatible diagnostics.

    normalize_weights affects only weighted_comm2, weight and weighted_mean;
    comm2 and weighted_comm2_unnormalized do not depend on it. For the
    optimized objective alone, use objective_loss.
    """
    if method == "overlap":
        return overlap_commutator_loss(decomposition, normalize_weights=normalize_weights)
    if method == "dense":
        return dense_commutator_loss(decomposition, normalize_weights=normalize_weights)
    raise ValueError("method must be 'overlap' or 'dense'")


def objective_loss(decomposition, *, loss_kind="weighted", method="overlap"):
    """The optimized objective for loss_kind, and nothing else.

    raw: L_raw. weighted: sum_{i<j} a_i a_j ||[u_i,u_j]||_F^2 with the stored
    unnormalized a_i. method='overlap' uses scaled blocks and math.fsum for
    both kinds, exactly as the partition solver and gradients do; 'dense'
    uses literal matrices and explicit pair weights.
    """
    loss_kind = loss_kind_name(loss_kind)
    if method == "overlap":
        return OverlapEvaluator(decomposition, loss_kind=loss_kind).total()
    if method == "dense":
        result = dense_commutator_loss(decomposition, normalize_weights=False)
        return result.comm2 if loss_kind == "raw" else result.weighted_comm2_unnormalized
    raise ValueError("method must be 'overlap' or 'dense'")


def raw_loss(decomposition, *, method="overlap"):
    """Only L_raw: objective_loss with loss_kind='raw'."""
    return objective_loss(decomposition, loss_kind="raw", method=method)
