"""Exact, fixed-partition density channels for the three particle sectors.

All coefficients are positive, UNNORMALIZED Green weights. Both one-body
and two-body channels are required to reconstruct the original Hamiltonian.
The one-body matrix is the original h; do not apply a second correction.
"""

from numbers import Real,Integral

import numpy as np

from .models import (
    Channel, 
    DensitySource, 
    FixedDecomposition, 
    OneBodyChannel,
    OneBodyDecomposition, 
    TwoBodyDecomposition,
    PreparedProblem, 
    ValidationReport,
)
from .problem import select_sector_policy
from .problem import (
    _metric, 
    _real_array, 
    _require_report, 
    validate_gauge_state,
)

def positive_scale(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite, real, and strictly positive")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite, real, and strictly positive")
    return float(value)


def fixed_partition_amplitudes(eigenvalues, sources, tau2, t=None,
                               partition="balanced_log", amplitude_bound=1.0):
    """Return owned (t, d) arrays, with d[i]=(d_left,d_right) for source i.

    balanced_log: sigma*sqrt(k)*exp(t), sigma*sign(-W)*sqrt(k)*exp(-t),
                  k=abs(W*tau2), sigma=sign(ell_left).
                  Both legs are zero if W=0.
    anchored_log: ell_left*sqrt(tau2)*exp(t),
                 -ell_right*sqrt(tau2)*exp(-t).

    t=None means zero log ratios; a finite scalar broadcasts to every source.
    All requested amplitudes must satisfy abs(d)<amplitude_bound<=1. The
    paired real-exponential domain is checked, never repaired by clipping.
    """
    tau2 = positive_scale(tau2, "tau2")
    bound = positive_scale(amplitude_bound, "amplitude_bound")
    if bound > 1:
        raise ValueError("amplitude_bound must be <=1 for paired real exponentials")
    if partition not in ("balanced_log", "anchored_log"):
        raise ValueError("partition must be 'balanced_log' or 'anchored_log'")
    ell = _real_array(eigenvalues, "eigenvalues")
    if ell.ndim != 2:
        raise ValueError("eigenvalues must have shape (r,n)")
    count = len(sources)
    if t is None:
        ratios = np.zeros(count, dtype=np.float64)
    else:
        supplied = _real_array(t, "t")
        if supplied.ndim == 0:
            ratios = np.full(count, float(supplied))
        elif supplied.shape == (count,):
            ratios = supplied.copy()
        else:
            raise ValueError(f"t must be a scalar or a length-{count} vector")
    amplitudes = np.zeros((count, 2), dtype=np.float64)
    log_tau = np.log(tau2)
    log_bound = np.log(bound)
    for i, source in enumerate(sources):
        g, (p, q) = source.bix, source.modes
        if not (0 <= g < ell.shape[0] and 0 <= p < ell.shape[1] and 0 <= q < ell.shape[1]):
            raise ValueError(f"Invalid basis or mode indices in source {i}")
        left, right = ell[g, p], ell[g, q]
        if partition == "balanced_log":
            if left == 0 or right == 0:
                continue  # fixed formal identity channels, positive analytic weights
            base = 0.5 * (np.log(abs(left)) + np.log(abs(right)) + log_tau)
            logs = (base + ratios[i], base - ratios[i])
        else:
            logs = ((-np.inf if left == 0 else np.log(abs(left)) + 0.5*log_tau + ratios[i]),
                    (-np.inf if right == 0 else np.log(abs(right)) + 0.5*log_tau - ratios[i]))
        if max(logs) >= log_bound:
            raise ValueError(
                f"Source {i} (bix={g}, modes={(p,q)}): partition leaves abs(d)<{bound}. "
                "Choose a smaller tau2 or a different fixed t; amplitudes were not clipped."
            )
        amplitudes[i] = (np.sign(left)*np.exp(logs[0]), -np.sign(right)*np.exp(logs[1]))
        if any(log != -np.inf and d == 0 for log, d in zip(logs, amplitudes[i])):
            raise ValueError(f"Source {i}: amplitude underflow; choose a less extreme partition")
        if not np.isfinite(amplitudes[i]).all() or np.max(np.abs(amplitudes[i])) >= bound:
            raise ValueError(f"Source {i}: amplitudes lie outside the requested open domain")
    ratios.setflags(write=False)
    amplitudes.setflags(write=False)
    return ratios, amplitudes

def _problem_policy(problem):
    if not isinstance(problem, PreparedProblem):
        raise TypeError("problem must be a PreparedProblem returned by prepare_problem/load_problem")
    _, policy = select_sector_policy(problem.nelec, problem.data.nbasis)
    if problem.policy != policy:
        raise ValueError("Problem policy is inconsistent with its particle numbers")
    return policy


def enumerate_sources(problem):
    """Fixed ordering, including zero couplings (no thresholds/deduplication).

    hybrid: bix then alpha<=beta; diagonal UD "two_branch", off-diagonal "spin_quartet".
    ab_only: bix then all ordered spatial (alpha,beta), always spins=(up,down).
    aa_only: bix then alpha<beta, spins=(up,up).
    """
    policy = _problem_policy(problem)
    r, n, _ = problem.data.chol.shape
    sources = []
    for g in range(r):
        for p in range(n):
            if policy == "ab_only":
                for q in range(n):
                    sources.append(DensitySource(g, (p,q), "two_branch", (0,1)))
            elif policy == "aa_only":
                for q in range(p+1, n):
                    sources.append(DensitySource(g, (p,q), "two_branch", (0,0)))
            else:
                for q in range(p, n):
                    sources.append(DensitySource(g, (p,q), "two_branch" if p == q else "spin_quartet",
                                                 (0,1) if p == q else None))
    return tuple(sources)


def build_two_body_channels(problem, gauge, tau2, t=None, *, partition="balanced_log",
                            amplitude_bound=1.0):
    """Build a reference two-body decomposition at a supplied validated gauge.

    t is indexed by enumerate_sources(problem), not by individual channels.
    One "two_branch" source emits two equal-weight +/- branches; an off-diagonal
    hybrid source emits a shared-amplitude four-spin quartet. The full state
    is validated here for reference correctness. Future inner-loss kernels
    should reuse validated state/source data rather than repeat ERI checks.
    """
    policy = _problem_policy(problem)
    tau2 = positive_scale(tau2, "tau2")
    validate_gauge_state(problem.data, gauge, problem.report.options)
    sources = enumerate_sources(problem)
    ratios, amplitudes = fixed_partition_amplitudes(
        gauge.eigenvalues, sources, tau2, t, partition, amplitude_bound)
    weight = 1.0 / tau2
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("tau2 gives a nonrepresentable positive channel coefficient")
    channels, slices = [], []
    for source, (left, right) in zip(sources, amplitudes):
        start = len(channels)
        if source.kind == "two_branch":
            for sign in (1.0, -1.0):
                channels.append(Channel(source.bix, source.spins, weight/2,
                                        source.modes, (sign*left, sign*right)))
        else:
            for spins, sign in (((0,0),1.0), ((0,1),-1.0), ((1,0),-1.0), ((1,1),1.0)):
                channels.append(Channel(source.bix, spins, weight, source.modes,
                                        (sign*left, sign*right)))
        slices.append((start, len(channels)))
    return TwoBodyDecomposition(gauge, policy, problem.nelec, tau2, partition,
                                float(amplitude_bound), sources, ratios, amplitudes,
                                tuple(slices), tuple(channels))


def shift_pair(shift):
    """Canonical (mu_up, mu_down) floats; None means (0, 0)."""
    if shift is None:
        return (0.0, 0.0)
    try:
        values = tuple(shift)
    except TypeError as exc:
        raise ValueError("shift must be a pair (mu_up, mu_down)") from exc
    if len(values) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                               or not np.isfinite(v) for v in values):
        raise ValueError("shift must contain two finite real numbers (mu_up, mu_down)")
    return (float(values[0]), float(values[1]))


def _one_body_from_eigensystem(X, ell, tau1, shift, spins, nelec):
    """Channels d[k,sigma]=(mu_sigma-epsilon_k)*tau1 on a fixed h eigensystem."""
    shift = shift_pair(shift)
    if nelec is None and any(shift):
        raise ValueError("A nonzero one-body shift requires particle numbers: pass nelec "
                         "to prepare_problem (sum_sigma mu_sigma N_sigma must be a constant)")
    if 1 not in spins and shift[1] != 0:
        raise ValueError("This policy emits no down-spin one-body channels; mu_down must be 0")
    weight = 1.0 / tau1
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("tau1 gives a nonrepresentable positive coefficient")
    channels = []
    for s in spins:
        ds = (shift[s] - ell) * tau1
        if not np.isfinite(ds).all() or np.any(1.0 + ds <= 0):
            raise ValueError("One-body amplitudes require 1-(epsilon-mu)*tau1 > 0; "
                             "choose smaller tau1 or a larger shift")
        channels.extend(OneBodyChannel(s, weight, k, float(d)) for k, d in enumerate(ds))
    return OneBodyDecomposition(X, ell, tau1, tuple(channels), shift)


def build_one_body_channels(problem, tau1, shift=None):
    """Jointly diagonalize bare h and retain one channel per mode and spin.

    d=(mu_sigma-epsilon)*tau1, weight=1/tau1, with shift=(mu_up, mu_down)
    defaulting to zero. Zero amplitudes are retained as identity channels.
    Only up spin is emitted for aa_only (mu_down must then be 0). Real
    exponentials require d>-1; positive d>=1 is allowed because this identity
    has no minus branch. A nonzero shift requires problem.nelec, since
    sum_sigma mu_sigma N_sigma is folded into Lambda.
    """
    policy = _problem_policy(problem)
    tau1 = positive_scale(tau1, "tau1")
    h = _real_array(problem.data.h1, "h1")
    n = problem.data.nbasis
    if h.shape != (n,n):
        raise ValueError("h1 must be square")
    _require_report(ValidationReport({"h1_symmetry": _metric(h, h.T, problem.report.options)},
                                     problem.report.options))
    ell, X = np.linalg.eigh(h)
    ell.setflags(write=False)
    X.setflags(write=False)
    spins = (0,) if policy == "aa_only" else (0,1)
    return _one_body_from_eigensystem(X, ell, tau1, shift, spins, problem.nelec)


def one_body_at_shift(one_body, shift, nelec):
    """Rebuild one-body channels at a new shift, reusing the cached eigensystem."""
    if not isinstance(one_body, OneBodyDecomposition):
        raise TypeError("one_body must be a OneBodyDecomposition")
    return _one_body_from_eigensystem(one_body.X, one_body.eigenvalues, one_body.tau1,
                                      shift, one_body.spins, nelec)


def build_fixed_decomposition(problem, gauge, tau1, tau2, t=None, *,
                              partition="balanced_log", amplitude_bound=1.0, shift=None):
    """Build both lists with the same policy, without normalizing either list."""
    two_body = build_two_body_channels(problem, gauge, tau2, t,
                                       partition=partition, amplitude_bound=amplitude_bound)
    one_body = build_one_body_channels(problem, tau1, shift)
    result = FixedDecomposition(one_body, two_body)
    if not np.isfinite(result.denom) or result.denom <= 0:
        raise ValueError("The combined positive coefficient sum is not representable")
    return result


def two_body_rotation_matrix(channel, gauge):
    """Return [u_up,u_down], using None for an inactive identity spin block.

    Multiply actual rank-one factors in the stored order. This also retains
    the d_left*d_right term for a manually supplied repeated same-spin mode;
    generated step-2 channels never have repeated same-spin legs.
    """
    if not isinstance(channel, Channel):
        raise TypeError("channel must be a Channel")
    r, n, _ = gauge.X.shape
    if channel.bix >= r or any(p >= n for p in channel.modes):
        raise ValueError("Channel basis/mode index is out of range")
    U = [None, None]
    for spin, p, d in zip(channel.spins, channel.modes, channel.d):
        x = gauge.X[channel.bix, :, p]
        factor = np.eye(n) + d * np.outer(x, x)
        U[spin] = factor if U[spin] is None else U[spin] @ factor
    return U


def one_body_rotation_matrix(channel, one_body):
    """Single rank-one factor in the jointly diagonalized h eigenbasis."""
    n = one_body.X.shape[0]
    if channel.spin not in (0,1) or not 0 <= channel.mode < n:
        raise ValueError("Invalid one-body channel index")
    x = one_body.X[:, channel.mode]
    U = [None, None]
    U[channel.spin] = np.eye(n) + channel.d * np.outer(x, x)
    return U


def get_rotation_matrix(decomposition, index):
    """Index the combined list: one-body first, then two-body channels."""
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral):
        raise TypeError("Channel index must be an integer")
    if not 0 <= index < decomposition.nterms:
        raise IndexError("Channel index is out of range")
    n_one = len(decomposition.one_body.channels)
    if index < n_one:
        return one_body_rotation_matrix(decomposition.one_body.channels[index],
                                         decomposition.one_body)
    return two_body_rotation_matrix(decomposition.two_body.channels[index-n_one],
                                     decomposition.two_body.gauge)
