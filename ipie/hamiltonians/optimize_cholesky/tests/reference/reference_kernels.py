"""FROZEN reference oracle for optimize_cholesky (ipie lafqmc_thc @ 3680f29).

Do not edit, optimize or "clean up" this file. It exists so that the refactored
package can be compared with the algorithms that produced the published
numbers. It is a self-contained copy of the 3680f29 code paths behind

    GaugeEvaluator.fixed / .fixed_z / .relaxed   (gradients.py)
    loss.objective_loss                          (loss.py, overlap method)
    partitions._solve_partitions + shift.py      (inner Gauss-Seidel solve)
    problem.exponential_gauge, gradients.exponential_gauge_vjp

with the same arithmetic and operation order, but with the input validation
and the dataclass object model removed. It imports nothing from the package,
only NumPy and SciPy, and works on plain arrays:

    oracle = ReferenceOracle(h1, chol, nelec, tau1=..., tau2=...)
    oracle.fixed(O, t=..., shift=...)          -> dict
    oracle.fixed_z(O, z=..., shift=...)        -> dict
    oracle.relaxed(O, t0=None, inner=InnerSettings(...)) -> dict
    oracle.objective_loss(O, t=..., shift=..., loss_kind=...) -> float

Conventions are those of the package: h1 is the bare one-body matrix,
chol[g,p,r] with eri[p,r,q,s] = sum_g chol[g,p,r] chol[g,q,s], spin 0 = up.
The cost is that of the original code (Python loops over channel pairs).
"""

from collections import namedtuple
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
from scipy.linalg import expm, expm_frechet

FROZEN_COMMIT = "3680f2920370dd114c05503045c178ca54b632d0"


# --------------------------------------------------------------------------- errors

class GaugeRegularityError(ValueError):
    """Same meaning as models.GaugeRegularityError."""


class GaugeFeasibilityError(ValueError):
    """Same meaning as models.GaugeFeasibilityError."""


class InnerConvergenceError(ValueError):
    """Same meaning as models.InnerConvergenceError."""


# --------------------------------------------------------------------------- settings

@dataclass(frozen=True)
class InnerSettings:
    """Mirror of models.InnerOptions at 3680f29 (same defaults)."""

    max_sweeps: int = 500
    gradient_atol: float = 1e-11
    gradient_rtol: float = 1e-9
    amplitude_limit: Optional[float] = None
    max_abs_t: float = 30.0
    optimize_shift: bool = False
    one_body_limit: Optional[float] = None
    shift_atol: float = 0.0
    shift_rtol: float = 1e-8


@dataclass(frozen=True)
class GradientSettings:
    """Mirror of models.GradientOptions at 3680f29 (same defaults)."""

    gap_atol: float = 1e-12
    gap_rtol: float = 1e-10
    zero_atol: float = 1e-12
    zero_rtol: float = 1e-10
    bound_switch_atol: float = 1e-10


# --------------------------------------------------------------------------- records

Source = namedtuple("Source", "bix modes kind spins")
Channel = namedtuple("Channel", "bix spins a_i modes d")
OneBodyChannel = namedtuple("OneBodyChannel", "spin a_i mode d")
Gauge = namedtuple("Gauge", "O chol eigenvalues X")
OneBody = namedtuple("OneBody", "X eigenvalues tau1 channels shift")
TwoBody = namedtuple("TwoBody", "gauge policy nelec tau2 partition amplitude_bound "
                                "sources t amplitudes source_slices channels")
Decomposition = namedtuple("Decomposition", "one_body two_body")
Block = namedtuple("Block", "basis modes kappa")

_BRANCHES = {
    "two_branch": ((None, 1.0), (None, -1.0)),
    "spin_quartet": (((0, 0), 1.0), ((0, 1), -1.0), ((1, 0), -1.0), ((1, 1), 1.0)),
}
_BRANCH_WEIGHT = {"two_branch": 0.5, "spin_quartet": 1.0}
DEFAULT_ONE_BODY_LIMIT = 1.0 - 1e-8


def _readonly(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _one_body_spins(one_body):
    return tuple(sorted({channel.spin for channel in one_body.channels}))


def _weight_sum(channels):
    return sum((channel.a_i for channel in channels), 0.0)


def coefficients(decomposition):
    """Unnormalized Green weights, one-body first (FixedDecomposition.coefficients)."""
    return np.array([c.a_i for c in decomposition.one_body.channels]
                    + [c.a_i for c in decomposition.two_body.channels], dtype=np.float64)


def denom(decomposition):
    return _weight_sum(decomposition.one_body.channels) + _weight_sum(decomposition.two_body.channels)


# --------------------------------------------------------------------------- problem and gauge

def sector_policy(nelec):
    """problem.select_sector_policy: None -> hybrid; N_down=0 -> aa_only; (1,1) -> ab_only."""
    if nelec is None:
        return "hybrid"
    nelec = tuple(int(v) for v in nelec)
    if nelec[1] == 0:
        return "aa_only"
    if nelec == (1, 1):
        return "ab_only"
    return "hybrid"


def skew_from_parameters(theta, nchol):
    S = np.zeros((nchol, nchol), dtype=np.float64)
    rows, cols = np.triu_indices(nchol, 1)
    S[rows, cols] = theta
    S[cols, rows] = -theta
    return S


def exponential_gauge(theta, nchol, reference=None):
    """O = expm(S(theta)) @ reference, theta packed as (0,1),(0,2),...,(1,2),..."""
    reference = np.eye(nchol) if reference is None else np.asarray(reference, dtype=np.float64)
    return expm(skew_from_parameters(np.asarray(theta, dtype=np.float64), nchol)) @ reference


def exponential_gauge_vjp(theta, nchol, gradient_O, reference=None):
    """Gradient in the packed exponential coordinates (exact Frechet adjoint)."""
    theta = np.asarray(theta, dtype=np.float64)
    E = np.asarray(gradient_O, dtype=np.float64)
    ref = np.eye(nchol) if reference is None else np.asarray(reference, dtype=float)
    S = skew_from_parameters(theta, nchol)
    adjoint = expm_frechet(S.T, E @ ref.T, compute_expm=False)
    rows, cols = np.triu_indices(nchol, 1)
    return _readonly(adjoint[rows, cols] - adjoint[cols, rows])


def rotate_and_diagonalize(chol, O):
    """GaugeEvaluator._gauge: rotated factors and their ascending eigensystems."""
    r = chol.shape[0]
    O = np.eye(r) if O is None else np.asarray(O, dtype=np.float64)
    rotated = (O @ chol.reshape(r, -1)).reshape(chol.shape)
    ell, X = np.linalg.eigh(rotated)
    return Gauge(_readonly(O), _readonly(rotated), _readonly(ell), _readonly(X))


# --------------------------------------------------------------------------- sources and channels

def enumerate_sources(policy, nchol, nbasis):
    sources = []
    for g in range(nchol):
        for p in range(nbasis):
            if policy == "ab_only":
                for q in range(nbasis):
                    sources.append(Source(g, (p, q), "two_branch", (0, 1)))
            elif policy == "aa_only":
                for q in range(p+1, nbasis):
                    sources.append(Source(g, (p, q), "two_branch", (0, 0)))
            else:
                for q in range(p, nbasis):
                    sources.append(Source(g, (p, q), "two_branch" if p == q else "spin_quartet",
                                          (0, 1) if p == q else None))
    return tuple(sources)


def fixed_partition_amplitudes(eigenvalues, sources, tau2, t=None, partition="balanced_log",
                               amplitude_bound=1.0):
    ell = np.asarray(eigenvalues, dtype=np.float64)
    count = len(sources)
    if t is None:
        ratios = np.zeros(count, dtype=np.float64)
    else:
        supplied = np.asarray(t, dtype=np.float64)
        ratios = np.full(count, float(supplied)) if supplied.ndim == 0 else supplied.copy()
    amplitudes = np.zeros((count, 2), dtype=np.float64)
    log_tau = np.log(tau2)
    log_bound = np.log(amplitude_bound)
    for i, source in enumerate(sources):
        g, (p, q) = source.bix, source.modes
        left, right = ell[g, p], ell[g, q]
        if partition == "balanced_log":
            if left == 0 or right == 0:
                continue
            base = 0.5 * (np.log(abs(left)) + np.log(abs(right)) + log_tau)
            logs = (base + ratios[i], base - ratios[i])
        else:
            logs = ((-np.inf if left == 0 else np.log(abs(left)) + 0.5*log_tau + ratios[i]),
                    (-np.inf if right == 0 else np.log(abs(right)) + 0.5*log_tau - ratios[i]))
        if max(logs) >= log_bound:
            raise GaugeFeasibilityError(f"Source {i}: partition leaves abs(d)<{amplitude_bound}")
        amplitudes[i] = (np.sign(left)*np.exp(logs[0]), -np.sign(right)*np.exp(logs[1]))
        if any(log != -np.inf and d == 0 for log, d in zip(logs, amplitudes[i])):
            raise GaugeFeasibilityError(f"Source {i}: amplitude underflow")
    ratios.setflags(write=False)
    amplitudes.setflags(write=False)
    return ratios, amplitudes


def _source_channels(source, amplitudes, tau2):
    left, right = amplitudes
    weight = _BRANCH_WEIGHT[source.kind]/tau2
    return tuple(Channel(source.bix, source.spins if spins is None else spins, weight,
                         source.modes, (sign*left, sign*right))
                 for spins, sign in _BRANCHES[source.kind])


def _source_slices(sources):
    endpoints = np.r_[0, np.cumsum([len(_BRANCHES[source.kind]) for source in sources])]
    return tuple((int(a), int(b)) for a, b in zip(endpoints[:-1], endpoints[1:]))


def _two_body(gauge, policy, nelec, tau2, partition, amplitude_bound, sources, t, amplitudes):
    channels = tuple(c for source, d in zip(sources, amplitudes)
                     for c in _source_channels(source, d, tau2))
    return TwoBody(gauge, policy, nelec, tau2, partition, float(amplitude_bound),
                   sources, t, amplitudes, _source_slices(sources), channels)


def _shift_pair(shift):
    if shift is None:
        return (0.0, 0.0)
    return (float(shift[0]), float(shift[1]))


def _one_body_from_eigensystem(X, ell, tau1, shift, spins):
    shift = _shift_pair(shift)
    weight = 1.0 / tau1
    channels = []
    for s in spins:
        ds = (shift[s] - ell) * tau1
        if np.any(1.0 + ds <= 0):
            raise GaugeFeasibilityError("One-body amplitudes require 1-(epsilon-mu)*tau1 > 0")
        channels.extend(OneBodyChannel(s, weight, k, float(d)) for k, d in enumerate(ds))
    return OneBody(X, ell, tau1, tuple(channels), shift)


def build_one_body(h1, tau1, shift, policy):
    ell, X = np.linalg.eigh(np.asarray(h1, dtype=np.float64))
    ell.setflags(write=False)
    X.setflags(write=False)
    spins = (0,) if policy == "aa_only" else (0, 1)
    return _one_body_from_eigensystem(X, ell, tau1, shift, spins)


def one_body_at_shift(one_body, shift):
    return _one_body_from_eigensystem(one_body.X, one_body.eigenvalues, one_body.tau1,
                                      _shift_pair(shift), _one_body_spins(one_body))


def with_partitions(decomposition, t):
    two = decomposition.two_body
    ratios, amplitudes = fixed_partition_amplitudes(two.gauge.eigenvalues, two.sources, two.tau2, t,
                                                    two.partition, two.amplitude_bound)
    channels = tuple(c for source, d in zip(two.sources, amplitudes)
                     for c in _source_channels(source, d, two.tau2))
    return decomposition._replace(two_body=two._replace(t=ratios, amplitudes=amplitudes, channels=channels))


def with_shift(decomposition, shift):
    return decomposition._replace(one_body=one_body_at_shift(decomposition.one_body, shift))


# --------------------------------------------------------------------------- bounds

def partition_limits(amplitude_bound, amplitude_limit, max_abs_t):
    rho = (1.0-1e-8)*amplitude_bound if amplitude_limit is None else float(amplitude_limit)
    return rho, float(max_abs_t)


def partition_intervals(eigenvalues, sources, tau2, partition, amplitude_bound, *,
                        amplitude_limit=None, max_abs_t=30.0):
    rho, cap = partition_limits(amplitude_bound, amplitude_limit, max_abs_t)
    log_rho = math.log(rho)
    log_tau = math.log(tau2)
    bounds = np.empty((len(sources), 2))
    for i, source in enumerate(sources):
        left, right = (eigenvalues[source.bix, p] for p in source.modes)
        if partition == "balanced_log":
            base = (-np.inf if left == 0 or right == 0 else
                    0.5*(math.log(abs(left))+math.log(abs(right))+log_tau))
            log_left = log_right = base
        else:
            log_left = -np.inf if left == 0 else math.log(abs(left))+0.5*log_tau
            log_right = -np.inf if right == 0 else math.log(abs(right))+0.5*log_tau
        lower, upper = max(-cap, log_right-log_rho), min(cap, log_rho-log_left)
        if lower > upper:
            raise GaugeFeasibilityError(f"Source {i} has no feasible partition within the requested bounds")
        bounds[i] = (lower, upper)
    return _readonly(bounds)


def one_body_limit_value(one_body_limit):
    return DEFAULT_ONE_BODY_LIMIT if one_body_limit is None else float(one_body_limit)


def shift_bounds(one_body, one_body_limit=None):
    rho = one_body_limit_value(one_body_limit)
    lower = float(np.max(one_body.eigenvalues) - rho/one_body.tau1)
    spins = _one_body_spins(one_body)
    return tuple(lower if s in spins else None for s in (0, 1))


def shift_tolerance(eigenvalues, shift_atol=0.0, shift_rtol=1e-8):
    eps = np.asarray(eigenvalues, dtype=np.float64)
    width = float(np.ptp(eps)) if eps.size else 0.0
    scale = width if width > 0 else max(1.0, float(np.max(np.abs(eps), initial=0.0)))
    return float(shift_atol + shift_rtol*scale)


# --------------------------------------------------------------------------- loss kernels

def commutator_norm2(d, e, overlap):
    """Squared Frobenius norm of [V diag(d) V.T, Z diag(e) Z.T] (loss.commutator_norm2)."""
    d = np.asarray(d, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    C = np.asarray(overlap, dtype=np.float64)
    if not d.size or not e.size:
        return 0.0
    with np.errstate(over="raise", invalid="raise"):
        first = 2.0 * np.sum((d[:, None] * C * e[None, :])**2)
        product = ((d[:, None] * C) * e[None, :]) @ C.T
        second = 2.0 * np.sum(product * product.T)
        value = float(first - second)
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite overlap commutator norm")
    tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(float(first)), abs(float(second)))
    if value < -tolerance:
        raise FloatingPointError("Squared norm is materially negative")
    return max(0.0, value)


def pair_scales(decomposition, loss_kind):
    weights = coefficients(decomposition)
    return np.ones_like(weights) if loss_kind == "raw" else weights


class OverlapEvaluator:
    """loss.OverlapEvaluator with check=False (trusted decomposition)."""

    def __init__(self, decomposition, *, loss_kind="weighted"):
        self.decomposition = decomposition
        self.loss_kind = loss_kind
        self.scales = pair_scales(decomposition, loss_kind)
        self.root_scales = np.sqrt(self.scales)
        X = decomposition.two_body.gauge.X
        self.nbasis = X.shape[1]
        self._bases = {-1: decomposition.one_body.X}
        for bix, factor in enumerate(X):
            self._bases[bix] = factor
        self._overlaps = {}
        blocks = []
        for index, channel in enumerate(decomposition.one_body.channels):
            spin_blocks = [None, None]
            spin_blocks[channel.spin] = self._block(-1, ((channel.mode, channel.d),), self.root_scales[index])
            blocks.append(tuple(spin_blocks))
        offset = len(decomposition.one_body.channels)
        for index, channel in enumerate(decomposition.two_body.channels):
            legs = [[], []]
            for spin, mode, d in zip(channel.spins, channel.modes, channel.d):
                legs[spin].append((mode, d))
            blocks.append(tuple(self._block(channel.bix, spin_legs, self.root_scales[offset+index])
                                for spin_legs in legs))
        self.blocks = tuple(blocks)

    @staticmethod
    def _block(basis, legs, scale=1.0):
        merged = {}
        for mode, d in legs:
            old = merged.get(mode, 0.0)
            merged[mode] = old + d + old*d
        active = [(mode, float(kappa)) for mode, kappa in merged.items() if kappa != 0]
        if not active:
            return None
        scale = float(scale)
        return Block(basis, tuple(p for p, _ in active),
                     tuple(d if scale == 1.0 else scale*d for _, d in active))

    def _overlap(self, left, right):
        key = (min(left, right), max(left, right))
        if key not in self._overlaps:
            self._overlaps[key] = self._bases[key[0]].T @ self._bases[key[1]]
        overlap = self._overlaps[key]
        return overlap if left <= right else overlap.T

    def pair_norm2(self, i, j):
        result = 0.0
        for left, right in zip(self.blocks[i], self.blocks[j]):
            if left is None or right is None or left.basis == right.basis:
                continue
            C = self._overlap(left.basis, right.basis)[np.ix_(left.modes, right.modes)]
            result += commutator_norm2(left.kappa, right.kappa, C)
        return result

    def total(self):
        value = math.fsum(self.pair_norm2(i, j) for i in range(len(self.blocks))
                          for j in range(i+1, len(self.blocks)))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite commutator loss")
        return value


# --------------------------------------------------------------------------- inner solve

class PartitionProfile:
    """models.PartitionCoefficients: L(s) = C + left*exp(2(s-t)) + right*exp(-2(s-t))."""

    def __init__(self, source_index, t, C, left_term, right_term):
        self.source_index, self.t, self.C = source_index, float(t), float(C)
        self.left_term, self.right_term = float(left_term), float(right_term)

    @property
    def gradient(self):
        return 2.0*(self.left_term-self.right_term)

    def terms(self, t):
        delta = float(t)-self.t
        return (_scaled_exponential(self.left_term, 2.0*delta),
                _scaled_exponential(self.right_term, -2.0*delta))

    def change(self, t):
        left, right = self.terms(t)
        exponent = 2.0*(float(t)-self.t)
        if abs(exponent) < 0.5:
            return math.fsum((self.left_term*math.expm1(exponent),
                              self.right_term*math.expm1(-exponent)))
        return math.fsum((left, right, -self.left_term, -self.right_term))


def _scaled_exponential(value, exponent):
    if value == 0:
        return 0.0
    try:
        result = math.exp(math.log(value) + exponent)
    except OverflowError as exc:
        raise FloatingPointError("Partition coefficient/profile overflow") from exc
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite partition coefficient/profile")
    return result


def analytic_partition_update(profile, bounds):
    left, right = profile.left_term, profile.right_term
    if left == right == 0:
        optimal = profile.t
    elif left == 0:
        optimal = bounds[1]
    elif right == 0:
        optimal = bounds[0]
    else:
        optimal = profile.t+0.25*(math.log(right)-math.log(left))
    return float(np.clip(optimal, bounds[0], bounds[1]))


def projected_partition_gradient(t, gradient, bounds):
    result = np.array(gradient, dtype=np.float64, copy=True)
    result[(t == bounds[:, 0]) & (gradient > 0)] = 0.0
    result[(t == bounds[:, 1]) & (gradient < 0)] = 0.0
    result[bounds[:, 0] == bounds[:, 1]] = 0.0
    return _readonly(result)


ShiftProfile = namedtuple("ShiftProfile", "spin mu lower weight moment tau1")


def _shift_update(profile):
    optimum = profile.mu if profile.weight == 0 else profile.moment/profile.weight
    return float(max(optimum, profile.lower))


def _shift_projected_gradient(profile):
    g = 2.0*profile.tau1**2*(profile.weight*profile.mu-profile.moment)
    return min(g, 0.0) if profile.mu <= profile.lower else g


def shift_profiles(blocks, overlap, one_body, root_scales, lower):
    n_one = len(one_body.channels)
    shift = _shift_pair(one_body.shift)
    weight, moment = {}, {}
    for index, channel in enumerate(one_body.channels):
        terms = []
        for other in blocks[n_one:]:
            block = other[channel.spin]
            if block is None:
                continue
            C = overlap(-1, block.basis)[np.ix_((channel.mode,), block.modes)]
            terms.append(commutator_norm2((1.0,), block.kappa, C))
        w = float(root_scales[index])**2*math.fsum(terms)
        eps = float(one_body.eigenvalues[channel.mode])
        weight.setdefault(channel.spin, []).append(w)
        moment.setdefault(channel.spin, []).append(w*eps)
    return tuple(ShiftProfile(spin, shift[spin], float(lower[spin]), math.fsum(weight[spin]),
                              math.fsum(moment[spin]), float(one_body.tau1))
                 for spin in _one_body_spins(one_body))


class _PartitionWorkspace:
    """partitions._PartitionWorkspace with check=False."""

    def __init__(self, decomposition, loss_kind="weighted"):
        self.decomposition = decomposition
        self.two = decomposition.two_body
        self.one_body = decomposition.one_body
        self.shift = list(_shift_pair(decomposition.one_body.shift))
        self.shift_changed = False
        self.t = np.array(self.two.t, dtype=np.float64, copy=True)
        self.amplitudes = np.array(self.two.amplitudes, dtype=np.float64, copy=True)
        self.evaluator = OverlapEvaluator(decomposition, loss_kind=loss_kind)
        self.loss_kind = loss_kind
        self.root_scales = self.evaluator.root_scales
        self.blocks = list(self.evaluator.blocks)
        self.offset = len(decomposition.one_body.channels)
        self.spin_counts = []
        for start, stop in self.two.source_slices:
            counts = np.zeros((2, 2), dtype=int)
            for channel in self.two.channels[start:stop]:
                for leg, spin in enumerate(channel.spins):
                    counts[leg, spin] += 1
            self.spin_counts.append(counts)
        self.loss = self.total_loss()

    def _block_pair(self, left, right):
        if left is None or right is None or left.basis == right.basis:
            return 0.0
        C = self.evaluator._overlap(left.basis, right.basis)[np.ix_(left.modes, right.modes)]
        return commutator_norm2(left.kappa, right.kappa, C)

    def total_loss(self):
        value = math.fsum(self._block_pair(left, right)
                          for i, blocks in enumerate(self.blocks)
                          for other in self.blocks[i+1:]
                          for left, right in zip(blocks, other))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite raw commutator loss")
        return value

    def coefficients(self, i):
        source = self.two.sources[i]
        terms = []
        scale = float(self.root_scales[self.offset+self.two.source_slices[i][0]])
        for leg, mode in enumerate(source.modes):
            d = self.amplitudes[i, leg] if scale == 1.0 else scale*self.amplitudes[i, leg]
            values = []
            for blocks in self.blocks:
                for spin, multiplicity in enumerate(self.spin_counts[i][leg]):
                    block = blocks[spin]
                    if not multiplicity or d == 0 or block is None or block.basis == source.bix:
                        continue
                    C = self.evaluator._overlap(source.bix, block.basis)[np.ix_((mode,), block.modes)]
                    values.append(multiplicity*commutator_norm2((d,), block.kappa, C))
            terms.append(math.fsum(values))
        C = math.fsum((self.loss, -terms[0], -terms[1]))
        return PartitionProfile(i, float(self.t[i]), C, *terms)

    def update(self, i, t):
        source = self.two.sources[i]
        _, amplitudes = fixed_partition_amplitudes(self.two.gauge.eigenvalues, (source,), self.two.tau2, t,
                                                   self.two.partition, self.two.amplitude_bound)
        self.t[i], self.amplitudes[i] = t, amplitudes[0]
        start, stop = self.two.source_slices[i]
        channels = _source_channels(source, amplitudes[0], self.two.tau2)
        for j, channel in zip(range(self.offset+start, self.offset+stop), channels):
            legs = [[], []]
            for spin, mode, d in zip(channel.spins, channel.modes, channel.d):
                legs[spin].append((mode, d))
            self.blocks[j] = tuple(self.evaluator._block(source.bix, spin_legs, self.root_scales[j])
                                   for spin_legs in legs)

    def shift_coefficients(self, lower):
        one_body = self.one_body._replace(shift=tuple(self.shift))
        return shift_profiles(self.blocks, self.evaluator._overlap, one_body, self.root_scales, lower)

    def update_shift(self, shift):
        one_body = self.one_body
        for j, channel in enumerate(one_body.channels):
            d = (float(shift[channel.spin])-one_body.eigenvalues[channel.mode])*one_body.tau1
            if not math.isfinite(d) or 1.0+d <= 0:
                raise GaugeFeasibilityError("Shift leaves the one-body real-exponential domain 1+d>0")
            blocks = [None, None]
            blocks[channel.spin] = self.evaluator._block(-1, ((channel.mode, float(d)),), self.root_scales[j])
            self.blocks[j] = tuple(blocks)
        self.shift = [float(mu) for mu in shift]
        self.shift_changed = True

    def snapshot(self):
        result = with_partitions(self.decomposition, self.t)
        return with_shift(result, tuple(self.shift)) if self.shift_changed else result


InnerResult = namedtuple("InnerResult", "decomposition initial_loss loss bounds profiles gradient "
                                        "projected_gradient gradient_threshold history converged status "
                                        "amplitude_limit max_abs_t loss_kind optimize_shift shift_gradient "
                                        "input_loss")


def solve_partitions(decomposition, inner, loss_kind):
    """partitions._solve_partitions at 3680f29 (sequential exact coordinate updates)."""
    max_sweeps = inner.max_sweeps
    atol, rtol = float(inner.gradient_atol), float(inner.gradient_rtol)
    amplitude_limit, max_abs_t = inner.amplitude_limit, inner.max_abs_t
    optimize_shift = bool(inner.optimize_shift)
    rho1 = one_body_limit_value(inner.one_body_limit)
    lower = shift_bounds(decomposition.one_body, rho1)
    shift_tol = shift_tolerance(decomposition.one_body.eigenvalues, inner.shift_atol, inner.shift_rtol)
    if optimize_shift and decomposition.two_body.nelec is None:
        raise ValueError("optimize_shift requires particle numbers")
    two = decomposition.two_body
    bounds = partition_intervals(two.gauge.eigenvalues, two.sources, two.tau2, two.partition,
                                 two.amplitude_bound, amplitude_limit=amplitude_limit, max_abs_t=max_abs_t)
    workspace = _PartitionWorkspace(decomposition, loss_kind)
    if np.any(workspace.t < bounds[:, 0]) or np.any(workspace.t > bounds[:, 1]):
        raise ValueError("Initial t lies outside the optimizer bounds")
    input_loss = workspace.loss

    def relax_shift():
        new = list(workspace.shift)
        for profile in workspace.shift_coefficients(lower):
            new[profile.spin] = _shift_update(profile)
        change = max(abs(a-b) for a, b in zip(new, workspace.shift))
        workspace.update_shift(new)
        return change

    def shift_gradient():
        result = [0.0, 0.0]
        for profile in workspace.shift_coefficients(lower):
            result[profile.spin] = float(_shift_projected_gradient(profile))
        return tuple(result)

    if optimize_shift:
        relax_shift()
        workspace.loss = workspace.total_loss()

    def diagnostics():
        profiles = tuple(workspace.coefficients(i) for i in range(len(workspace.t)))
        gradient = _readonly([p.gradient for p in profiles])
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite partition gradient")
        projected = projected_partition_gradient(workspace.t, gradient, bounds)
        norm = float(np.max(np.abs(projected), initial=0.0))
        return profiles, gradient, projected, norm

    profiles, gradient, projected, norm = diagnostics()
    initial_loss = workspace.loss
    threshold = atol+rtol*norm
    history = [dict(sweep=0, loss=workspace.loss, projected_gradient_norm=norm, loss_decrease=0.0,
                    max_step=0.0, updates=0, shift_change=0.0)]
    converged = norm <= threshold
    for sweep in range(1, max_sweeps+1):
        if converged:
            break
        previous_loss = workspace.loss
        updates, max_step, shift_change = 0, 0.0, 0.0
        for i in range(len(workspace.t)):
            current = workspace.coefficients(i)
            optimal = analytic_partition_update(current, bounds[i])
            step = optimal-current.t
            if step == 0:
                continue
            change = current.change(optimal)
            roundoff = 128*np.finfo(float).eps*max(workspace.loss, current.left_term,
                                                   current.right_term, np.finfo(float).tiny)
            if change > roundoff:
                raise FloatingPointError("Analytic coordinate update increased the predicted loss")
            if change > 0:
                continue
            workspace.update(i, optimal)
            workspace.loss = math.fsum((workspace.loss, change))
            max_step = max(max_step, abs(step))
            updates += 1
        if optimize_shift:
            shift_change = relax_shift()
        workspace.loss = workspace.total_loss()
        roundoff = 512*np.finfo(float).eps*max(previous_loss, workspace.loss, np.finfo(float).tiny)
        if workspace.loss > previous_loss+roundoff:
            raise FloatingPointError(f"Sequential sweep increased the full {loss_kind} loss")
        profiles, gradient, projected, norm = diagnostics()
        history.append(dict(sweep=sweep, loss=workspace.loss, projected_gradient_norm=norm,
                            loss_decrease=previous_loss-workspace.loss, max_step=max_step,
                            updates=updates, shift_change=shift_change))
        converged = norm <= threshold and (not optimize_shift or shift_change <= shift_tol)
    rho, cap = partition_limits(decomposition.two_body.amplitude_bound, amplitude_limit, max_abs_t)
    return InnerResult(workspace.snapshot(), initial_loss, workspace.loss, bounds, profiles, gradient,
                       projected, threshold, tuple(history), bool(converged),
                       "converged" if converged else "max_sweeps", rho, cap, loss_kind,
                       optimize_shift, shift_gradient() if optimize_shift else None, input_loss)


# --------------------------------------------------------------------------- gradients

def commutator_norm2_derivatives(d, e, overlap):
    d, e, C = (np.asarray(x, dtype=np.float64) for x in (d, e, overlap))
    value = commutator_norm2(d, e, C)
    R = (C*e[None, :]) @ C.T
    S = C.T @ (d[:, None]*C)
    with np.errstate(over="raise", invalid="raise"):
        gd = 4*d*((C*C) @ (e*e))-4*((R*R) @ d)
        ge = 4*e*((C*C).T @ (d*d))-4*((S*S) @ e)
        gC = 4*(d*d)[:, None]*C*(e*e)[None, :]
        gC -= 8*((d[:, None]*R)*d[None, :]) @ (C*e[None, :])
    return value, gd, ge, gC


def eigh_vjp(eigenvalues, eigenvectors, grad_values, grad_vectors, settings):
    ell, X, ge, gX = (np.asarray(x, dtype=np.float64)
                      for x in (eigenvalues, eigenvectors, grad_values, grad_vectors))
    if not np.allclose(X.T@X, np.eye(ell.size), atol=1e-12, rtol=1e-10):
        raise ValueError("Eigenvectors must be orthonormal")
    gap = ell[None, :]-ell[:, None]
    mask = ~np.eye(ell.size, dtype=bool)
    tolerance = settings.gap_atol+settings.gap_rtol*np.max(np.abs(ell), initial=0.)
    if np.any(np.abs(gap[mask]) <= tolerance):
        raise GaugeRegularityError("Repeated or near-degenerate Cholesky eigenvalues")
    inverse = np.zeros_like(gap)
    np.divide(1., gap, out=inverse, where=mask)
    term = (X.T@gX)*inverse
    result = X@(np.diag(ge)+(term+term.T)/2)@X.T
    return _readonly((result+result.T)/2)


def loss_pullback(decomposition, loss_kind="weighted", rows=None):
    """gradients._loss_pullback: (loss, channel-leg adjoints, dL/dX).

    rows=(start, stop) restricts the outer pair index i to that range (only
    used to split very large evaluations across processes; the partial results
    add up to the full one). The default is the complete reference loop.
    """
    evaluator = OverlapEvaluator(decomposition, loss_kind=loss_kind)
    root = evaluator.root_scales
    two = decomposition.two_body
    offset = len(decomposition.one_body.channels)
    blocks = []
    for index, channel in enumerate(decomposition.one_body.channels):
        pair = [None, None]
        pair[channel.spin] = (-1, (channel.mode,), np.array([root[index]*channel.d]), ())
        blocks.append(tuple(pair))
    for index, channel in enumerate(two.channels):
        spin_legs = [[], []]
        scale = root[offset+index]
        for leg, (spin, mode, d) in enumerate(zip(channel.spins, channel.modes, channel.d)):
            spin_legs[spin].append((mode, scale*d, leg))
        blocks.append(tuple(None if not legs else (
            channel.bix, tuple(x[0] for x in legs), np.array([x[1] for x in legs]),
            tuple((index, x[2]) for x in legs)) for legs in spin_legs))
    g_amplitude = np.zeros((len(two.channels), 2))
    gX = np.zeros_like(two.gauge.X)
    terms = []
    start, stop = (0, len(blocks)) if rows is None else rows
    for i in range(start, stop):
        left_pair = blocks[i]
        for right_pair in blocks[i+1:]:
            for left, right in zip(left_pair, right_pair):
                if left is None or right is None or left[0] == right[0]:
                    continue
                a, ma, d, legs_a = left
                b, mb, e, legs_b = right
                C = evaluator._overlap(a, b)[np.ix_(ma, mb)]
                value, gd, ge, gC = commutator_norm2_derivatives(d, e, C)
                terms.append(value)
                for key, value in zip(legs_a, gd):
                    g_amplitude[key] += value
                for key, value in zip(legs_b, ge):
                    g_amplitude[key] += value
                if a >= 0:
                    gX[a][:, ma] += evaluator._bases[b][:, mb]@gC.T
                if b >= 0:
                    gX[b][:, mb] += evaluator._bases[a][:, ma]@gC
    if evaluator.loss_kind != "raw":
        g_amplitude *= root[offset:, None]
    return math.fsum(terms), g_amplitude, gX


def _branch_signs(source):
    return np.array([sign for _, sign in _BRANCHES[source.kind]])


def _amplitude_pullback(two, g_amplitude, settings):
    gell = np.zeros_like(two.gauge.eigenvalues)
    gt = np.zeros(len(two.sources))
    for i, (source, (start, stop)) in enumerate(zip(two.sources, two.source_slices)):
        signs = _branch_signs(source)
        gleft, gright = np.sum(signs[:, None]*g_amplitude[start:stop], axis=0)
        left, right = two.amplitudes[i]
        gt[i] = gleft*left-gright*right
        g, (p, q) = source.bix, source.modes
        ell = two.gauge.eigenvalues[g]
        if two.partition == "balanced_log":
            tolerance = settings.zero_atol+settings.zero_rtol*np.max(np.abs(ell), initial=0.)
            if abs(ell[p]) <= tolerance or abs(ell[q]) <= tolerance:
                raise GaugeRegularityError(f"Source {i}, factor {g}: balanced amplitudes meet a "
                                           "zero/near-zero eigenvalue")
            common = .5*(gleft*left+gright*right)
            gell[g, p] += common/ell[p]
            gell[g, q] += common/ell[q]
        else:
            gell[g, p] += gleft*math.exp(.5*math.log(two.tau2)+two.t[i])
            gell[g, q] -= gright*math.exp(.5*math.log(two.tau2)-two.t[i])
    return gell, gt


def _normalized_state(two, bounds, settings):
    widths = bounds[:, 1]-bounds[:, 0]
    if np.any(widths <= settings.bound_switch_atol):
        raise GaugeRegularityError("Collapsed/near-collapsed partition interval")
    if np.any(two.t < bounds[:, 0]) or np.any(two.t > bounds[:, 1]):
        raise ValueError("Partitions lie outside the requested closed bounds")
    return 2*(two.t-bounds[:, 0])/widths-1


def _bound_pullback(two, gt, z, amplitude_limit, max_abs_t, settings):
    rho, cap = partition_limits(two.amplitude_bound, amplitude_limit, max_abs_t)
    log_rho = math.log(rho)
    result = np.zeros_like(two.gauge.eigenvalues)
    for i, source in enumerate(two.sources):
        g, (p, q) = source.bix, source.modes
        left, right = two.gauge.eigenvalues[g, p], two.gauge.eigenvalues[g, q]
        if two.partition == "balanced_log":
            base = .5*(math.log(abs(left))+math.log(abs(right))+math.log(two.tau2))
            log_left = log_right = base
        else:
            log_left = -np.inf if left == 0 else math.log(abs(left))+.5*math.log(two.tau2)
            log_right = -np.inf if right == 0 else math.log(abs(right))+.5*math.log(two.tau2)
        lower_amp, upper_amp = log_right-log_rho, log_rho-log_left
        lower_weight, upper_weight = gt[i]*(1-z[i])/2, gt[i]*(1+z[i])/2
        if lower_weight != 0:
            if abs(lower_amp+cap) <= settings.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: lower amplitude bound ties the finite t cap")
            if lower_amp > -cap:
                if two.partition == "balanced_log":
                    result[g, p] += .5*lower_weight/left
                    result[g, q] += .5*lower_weight/right
                else:
                    result[g, q] += lower_weight/right
        if upper_weight != 0:
            if abs(upper_amp-cap) <= settings.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: upper amplitude bound ties the finite t cap")
            if upper_amp < cap:
                if two.partition == "balanced_log":
                    result[g, p] -= .5*upper_weight/left
                    result[g, q] -= .5*upper_weight/right
                else:
                    result[g, p] -= upper_weight/left
    return result


def assemble(chol, decomposition, settings, *, bounds=None, amplitude_limit=None, max_abs_t=30.,
             loss_kind="weighted", pullback=None):
    """gradients._assemble: loss and gauge derivatives at the stored decomposition.

    pullback: an already computed loss_pullback result (e.g. combined from row
    chunks); None computes it here.
    """
    two = decomposition.two_body
    loss, gamp, gX = loss_pullback(decomposition, loss_kind) if pullback is None else pullback
    gell, gt = _amplitude_pullback(two, gamp, settings)
    extra = np.zeros_like(gell)
    if bounds is not None:
        z = _normalized_state(two, bounds, settings)
        extra = _bound_pullback(two, gt, z, amplitude_limit, max_abs_t, settings)
    gL, extraL = [], []
    for g in range(len(gell)):
        try:
            gL.append(eigh_vjp(two.gauge.eigenvalues[g], two.gauge.X[g], gell[g], gX[g], settings))
        except GaugeRegularityError as exc:
            raise GaugeRegularityError(f"Factor {g}: {exc}") from exc
        extraL.append((two.gauge.X[g]*extra[g])@two.gauge.X[g].T)
    gL, extraL = np.array(gL), np.array(extraL)
    nchol = chol.shape[0]
    original = chol.reshape(nchol, -1)
    fixed = gL.reshape(nchol, -1)@original.T
    correction = extraL.reshape(nchol, -1)@original.T
    ambient = fixed+correction
    local = ambient@two.gauge.O.T
    generator = (local-local.T)/2
    tangent = generator@two.gauge.O
    return dict(loss=float(loss), gradient_O=_readonly(ambient), tangent_gradient=_readonly(tangent),
                generator_gradient=_readonly(generator), fixed_gradient_O=_readonly(fixed),
                bound_correction_O=_readonly(correction), partition_gradient=_readonly(gt),
                eigenvalue_gradient=_readonly(gell+extra))


# --------------------------------------------------------------------------- evaluator

class ReferenceOracle:
    """Array-level twin of GaugeEvaluator at 3680f29 (backend numpy, no validation)."""

    def __init__(self, h1, chol, nelec, *, tau1, tau2, partition="balanced_log", amplitude_bound=1.0,
                 loss_kind="weighted", shift=None, gradient_settings=None):
        self.h1 = np.array(h1, dtype=np.float64, order="C", copy=True)
        self.chol = np.array(chol, dtype=np.float64, order="C", copy=True)
        self.nelec = None if nelec is None else tuple(int(v) for v in nelec)
        self.policy = sector_policy(self.nelec)
        self.nchol, self.nbasis = self.chol.shape[0], self.chol.shape[1]
        self.tau1, self.tau2 = float(tau1), float(tau2)
        self.partition = partition
        self.amplitude_bound = float(amplitude_bound)
        self.loss_kind = loss_kind
        self.settings = GradientSettings() if gradient_settings is None else gradient_settings
        self.sources = enumerate_sources(self.policy, self.nchol, self.nbasis)
        self.one_body = build_one_body(self.h1, self.tau1, _shift_pair(shift), self.policy)

    def _one_body(self, shift=None):
        return self.one_body if shift is None else one_body_at_shift(self.one_body, shift)

    def gauge(self, O):
        return rotate_and_diagonalize(self.chol, O)

    def _decomposition(self, gauge, t, shift=None):
        t, d = fixed_partition_amplitudes(gauge.eigenvalues, self.sources, self.tau2, t,
                                          self.partition, self.amplitude_bound)
        two = _two_body(gauge, self.policy, self.nelec, self.tau2, self.partition,
                        self.amplitude_bound, self.sources, t, d)
        return Decomposition(self._one_body(shift), two)

    def decomposition(self, O=None, *, t=0., shift=None):
        return self._decomposition(self.gauge(O), t, shift)

    def bounds(self, gauge, amplitude_limit=None, max_abs_t=30.):
        return partition_intervals(gauge.eigenvalues, self.sources, self.tau2, self.partition,
                                   self.amplitude_bound, amplitude_limit=amplitude_limit, max_abs_t=max_abs_t)

    def objective_loss(self, O=None, *, t=0., shift=None, loss_kind=None):
        """loss.objective_loss(decomposition, loss_kind, method='overlap')."""
        decomposition = self.decomposition(O, t=t, shift=shift)
        return OverlapEvaluator(decomposition, loss_kind=loss_kind or self.loss_kind).total()

    def fixed(self, O=None, *, t=0., shift=None):
        """GaugeEvaluator.fixed: derivative with t and the shift held fixed."""
        result = assemble(self.chol, self.decomposition(O, t=t, shift=shift), self.settings,
                          loss_kind=self.loss_kind)
        result["mode"] = "fixed_t"
        return result

    def fixed_z(self, O=None, *, z=0., amplitude_limit=None, max_abs_t=30., shift=None):
        """GaugeEvaluator.fixed_z: derivative at fixed normalized coordinates z."""
        gauge = self.gauge(O)
        bounds = self.bounds(gauge, amplitude_limit, max_abs_t)
        z = np.asarray(z, dtype=np.float64)
        z = np.full(len(self.sources), float(z)) if z.ndim == 0 else z.copy()
        t = bounds[:, 0]+(z+1)*(bounds[:, 1]-bounds[:, 0])/2
        t = np.where(z == 1, bounds[:, 1], np.where(z == -1, bounds[:, 0], t))
        result = assemble(self.chol, self._decomposition(gauge, t, shift), self.settings, bounds=bounds,
                          amplitude_limit=amplitude_limit, max_abs_t=max_abs_t, loss_kind=self.loss_kind)
        result["mode"] = "fixed_z"
        return result

    def relaxed(self, O=None, *, t0=None, inner=None):
        """GaugeEvaluator.relaxed: inner solve at this gauge, then the envelope gradient."""
        inner = InnerSettings() if inner is None else inner
        gauge = self.gauge(O)
        bounds = self.bounds(gauge, inner.amplitude_limit, inner.max_abs_t)
        if t0 is None:
            t = bounds.mean(axis=1)
        else:
            t0 = np.asarray(t0, dtype=np.float64)
            t0 = np.full(len(self.sources), float(t0)) if t0.ndim == 0 else t0.copy()
            t = np.clip(t0, bounds[:, 0], bounds[:, 1])
        shift = None
        if inner.optimize_shift:
            lower = shift_bounds(self.one_body, inner.one_body_limit)
            shift = tuple(mu if low is None else max(mu, low) for mu, low in zip(self.one_body.shift, lower))
        result = solve_partitions(self._decomposition(gauge, t, shift), inner, self.loss_kind)
        if not result.converged:
            raise InnerConvergenceError("Relaxed gradient requires a converged partition solve")
        derivative = assemble(self.chol, result.decomposition, self.settings, bounds=result.bounds,
                              amplitude_limit=result.amplitude_limit, max_abs_t=result.max_abs_t,
                              loss_kind=result.loss_kind)
        derivative.update(mode="relaxed", inner_loss=result.loss, t=_readonly(result.decomposition.two_body.t),
                          shift=tuple(float(mu) for mu in result.decomposition.one_body.shift),
                          converged=result.converged, status=result.status,
                          sweeps=result.history[-1]["sweep"],
                          projected_gradient_norm=result.history[-1]["projected_gradient_norm"],
                          bounds=result.bounds, inner_result=result)
        return derivative
