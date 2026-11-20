"""Exact, fixed-partition density channels for the three particle sectors.

All coefficients are positive, UNNORMALIZED Green weights. Both one-body
and two-body channels are required to reconstruct the original Hamiltonian.
The one-body matrix is the original h; do not apply a second correction.

Channels are stored as arrays (models.ChannelTable, models.OneBodyTable).
The builders here produce canonical decompositions: the channel table is the
canonical layout of the source topology (models.SourceTable) at the source
amplitudes. with_channels builds decompositions from caller-supplied channel
lists (validated there); those are non-canonical and are accepted by the loss
kernels and rotation matrices, not by the partition solver or gradients.
"""

from numbers import Integral, Real

import numpy as np

from .models import (
    SOURCE_KINDS,
    Channel,
    ChannelTable,
    DensitySource,
    FixedDecomposition,
    OneBodyChannel,
    OneBodyDecomposition,
    OneBodyTable,
    PreparedProblem,
    SourceTable,
    TwoBodyDecomposition,
    ValidationReport,
    _frozen,
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


# The single branch table. Each source shares one amplitude pair (d_left, d_right)
# among its channels: channel = (spins, sign*d_left, sign*d_right) with weight
# _BRANCH_WEIGHT[kind]/tau2. spins=None means the source's own spins.
_BRANCHES = {
    "two_branch": ((None, 1.0), (None, -1.0)),
    "spin_quartet": (((0, 0), 1.0), ((0, 1), -1.0), ((1, 0), -1.0), ((1, 1), 1.0)),
}
_BRANCH_WEIGHT = {"two_branch": 0.5, "spin_quartet": 1.0}


def _branches(source):
    if source.kind not in _BRANCHES:
        raise ValueError(f"Unsupported source kind {source.kind!r}")
    return _BRANCHES[source.kind]


# --------------------------------------------------------------------------- sources

def source_table(sources):
    """SourceTable (with its canonical channel layout) from DensitySource records.

    Rejects unknown kinds and quartets on a repeated mode; the layout follows
    the single branch table above.
    """
    if isinstance(sources, SourceTable):
        return sources
    sources = tuple(sources)
    bix, modes, kind, spins = [], [], [], []
    channel_source, channel_sign, channel_spins, channel_weight, slices = [], [], [], [], []
    stop = 0
    for i, source in enumerate(sources):
        branches = _branches(source)
        if source.kind == "spin_quartet" and source.modes[0] == source.modes[1]:
            raise ValueError("A spin quartet needs two distinct modes")
        bix.append(source.bix)
        modes.append(tuple(source.modes))
        kind.append(SOURCE_KINDS.index(source.kind))
        spins.append((-1, -1) if source.spins is None else tuple(source.spins))
        for branch_spins, sign in branches:
            sp = source.spins if branch_spins is None else branch_spins
            if sp[0] == sp[1] and source.modes[0] == source.modes[1]:
                raise ValueError("Source legs must be distinct spin-orbital modes")
            channel_source.append(i)
            channel_sign.append(sign)
            channel_spins.append(tuple(sp))
            channel_weight.append(_BRANCH_WEIGHT[source.kind])
        slices.append((stop, stop+len(branches)))
        stop += len(branches)
    count, nch = len(sources), len(channel_source)
    return SourceTable(
        _frozen(bix, np.int64, (count,)), _frozen(modes, np.int64, (count, 2)),
        _frozen(kind, np.int64, (count,)), _frozen(spins, np.int64, (count, 2)),
        _frozen(slices, np.int64, (count, 2)), _frozen(channel_source, np.int64, (nch,)),
        _frozen(channel_sign, np.float64, (nch,)), _frozen(channel_spins, np.int64, (nch, 2)),
        _frozen(channel_weight, np.float64, (nch,)))


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
    Returns a SourceTable (a read-only sequence of DensitySource).
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
    return source_table(sources)


# --------------------------------------------------------------------------- amplitudes

def source_eigenvalues(eigenvalues, sources):
    """(ell_left, ell_right) per source: the factor eigenvalues of its two modes."""
    ell = np.asarray(eigenvalues, dtype=np.float64)
    return ell[sources.bix, sources.modes[:, 0]], ell[sources.bix, sources.modes[:, 1]]


def partition_log_bases(left, right, log_tau, partition):
    """log|d_L|, log|d_R| at t=0 for every source; -inf marks an absent leg.

    balanced_log: both legs are absent when W = left*right = 0. anchored_log:
    a leg is absent when its own eigenvalue is zero. The amplitude map, the
    partition intervals and the bound pullback all start from these.
    """
    with np.errstate(divide="ignore"):
        log_left, log_right = np.log(np.abs(left)), np.log(np.abs(right))
    if partition == "balanced_log":
        base = 0.5*(log_left + log_right + log_tau)
        return base, base
    return log_left + 0.5*log_tau, log_right + 0.5*log_tau


def partition_map(left, right, t, log_tau, partition):
    """Amplitudes (d_L, d_R), shape (count, 2), and their logs; no domain checks.

    d_L = sign(ell_L) exp(log_L + t), d_R = -sign(ell_R) exp(log_R - t); an
    absent leg is exactly +0.0. This is the only implementation of the map:
    fixed_partition_amplitudes adds the checks, the inner solver uses it as is.
    """
    base_left, base_right = partition_log_bases(left, right, log_tau, partition)
    logs_left, logs_right = base_left + t, base_right - t
    amplitudes = np.zeros((np.shape(left)[0], 2), dtype=np.float64)
    with np.errstate(over="ignore"):
        live = logs_left != -np.inf
        amplitudes[live, 0] = np.sign(left[live])*np.exp(logs_left[live])
        live = logs_right != -np.inf
        amplitudes[live, 1] = -np.sign(right[live])*np.exp(logs_right[live])
    return amplitudes, logs_left, logs_right


def _source_index_arrays(sources):
    if isinstance(sources, SourceTable):
        return sources.bix, sources.modes
    rows = [(s.bix, s.modes[0], s.modes[1]) for s in sources]
    array = np.array(rows, dtype=np.int64).reshape(-1, 3)
    return array[:, 0], array[:, 1:]


def fixed_partition_amplitudes(eigenvalues, sources, tau2, t=None,
                               partition="balanced_log", amplitude_bound=1.0):
    """Return owned (t, d) arrays, with d[i]=(d_left,d_right) for source i.

    balanced_log: sigma*sqrt(k)*exp(t), sigma*sign(-W)*sqrt(k)*exp(-t),
                  k=abs(W*tau2), sigma=sign(ell_left).
                  Both legs are zero if W=0.
    anchored_log: ell_left*sqrt(tau2)*exp(t),
                 -ell_right*sqrt(tau2)*exp(-t).

    sources: a SourceTable or a sequence of DensitySource. t=None means zero
    log ratios; a finite scalar broadcasts to every source. All requested
    amplitudes must satisfy abs(d)<amplitude_bound<=1. The paired
    real-exponential domain is checked, never repaired by clipping.
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
    bix, modes = _source_index_arrays(sources)
    count = bix.shape[0]
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
    if count and (np.any(bix < 0) or np.any(bix >= ell.shape[0])
                  or np.any(modes < 0) or np.any(modes >= ell.shape[1])):
        bad = int(np.flatnonzero((bix < 0) | (bix >= ell.shape[0]) | np.any(modes < 0, axis=1)
                                 | np.any(modes >= ell.shape[1], axis=1))[0])
        raise ValueError(f"Invalid basis or mode indices in source {bad}")
    left, right = ell[bix, modes[:, 0]], ell[bix, modes[:, 1]]
    amplitudes, logs_left, logs_right = partition_map(left, right, ratios, np.log(tau2), partition)
    _check_amplitude_domain(amplitudes, logs_left, logs_right, np.log(bound), bound, bix, modes)
    ratios.setflags(write=False)
    amplitudes.setflags(write=False)
    return ratios, amplitudes


def _check_amplitude_domain(amplitudes, logs_left, logs_right, log_bound, bound, bix, modes):
    """Reject the first source whose amplitudes leave abs(d)<bound or underflow.

    Per source the tests run in order: the logs reach log(bound); a present
    leg underflows to zero; an amplitude is nonfinite or reaches the bound.
    """
    over = np.maximum(logs_left, logs_right) >= log_bound
    underflow = (((logs_left != -np.inf) & (amplitudes[:, 0] == 0))
                 | ((logs_right != -np.inf) & (amplitudes[:, 1] == 0)))
    outside = ~np.isfinite(amplitudes).all(axis=1) | (np.max(np.abs(amplitudes), axis=1, initial=0.0) >= bound)
    failed = over | underflow | outside
    if not np.any(failed):
        return
    i = int(np.flatnonzero(failed)[0])
    if over[i]:
        g, p, q = int(bix[i]), int(modes[i, 0]), int(modes[i, 1])
        raise ValueError(
            f"Source {i} (bix={g}, modes={(p,q)}): partition leaves abs(d)<{bound}. "
            "Choose a smaller tau2 or a different fixed t; amplitudes were not clipped."
        )
    if underflow[i]:
        raise ValueError(f"Source {i}: amplitude underflow; choose a less extreme partition")
    raise ValueError(f"Source {i}: amplitudes lie outside the requested open domain")


# --------------------------------------------------------------------------- channel tables

def canonical_table(sources, amplitudes, tau2):
    """ChannelTable of the canonical layout: channel c = sign_c*(d_left, d_right) of its source."""
    src = sources.channel_source
    sign = sources.channel_sign
    d = np.empty((src.shape[0], 2), dtype=np.float64)
    d[:, 0] = sign*amplitudes[src, 0]
    d[:, 1] = sign*amplitudes[src, 1]
    modes = sources.modes[src]
    return ChannelTable(_frozen(sources.bix[src], np.int64), _frozen(sources.channel_spins, np.int64),
                        _frozen(modes, np.int64), _frozen(d, np.float64),
                        _frozen(sources.channel_weight/tau2, np.float64))


def _two_body_from_amplitudes(gauge, policy, nelec, tau2, partition, amplitude_bound,
                              sources, t, amplitudes):
    """Assemble a canonical TwoBodyDecomposition from per-source log ratios and amplitudes."""
    return TwoBodyDecomposition(gauge, policy, nelec, tau2, partition, float(amplitude_bound),
                                sources, t, amplitudes, canonical_table(sources, amplitudes, tau2), True)


def build_two_body_channels(problem, gauge, tau2, t=None, *, partition="balanced_log",
                            amplitude_bound=1.0):
    """Build a reference two-body decomposition at a supplied validated gauge.

    t is indexed by enumerate_sources(problem), not by individual channels.
    One "two_branch" source emits two equal-weight +/- branches; an off-diagonal
    hybrid source emits a shared-amplitude four-spin quartet. The gauge is
    validated here against the original tensors (reusing the problem's input
    report). GaugeEvaluator builds the same channels without these checks.
    """
    policy = _problem_policy(problem)
    tau2 = positive_scale(tau2, "tau2")
    validate_gauge_state(problem.data, gauge, problem.report.options, input_report=problem.report)
    sources = enumerate_sources(problem)
    ratios, amplitudes = fixed_partition_amplitudes(
        gauge.eigenvalues, sources, tau2, t, partition, amplitude_bound)
    weight = 1.0 / tau2
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("tau2 gives a nonrepresentable positive channel coefficient")
    return _two_body_from_amplitudes(gauge, policy, problem.nelec, tau2, partition,
                                     amplitude_bound, sources, ratios, amplitudes)


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
    n = np.asarray(ell).shape[0]
    blocks = []
    for s in spins:
        ds = (shift[s] - ell) * tau1
        if not np.isfinite(ds).all() or np.any(1.0 + ds <= 0):
            failed = ds[ds<-1.]
            raise ValueError("One-body amplitudes require 1-(epsilon-mu)*tau1 > 0; "
                             "choose smaller tau1 or a larger shift; "
                             f"current failed modes={failed}")
        blocks.append(ds)
    count = n*len(spins)
    table = OneBodyTable(_frozen(np.repeat(np.asarray(spins, dtype=np.int64), n), np.int64, (count,)),
                         _frozen(np.tile(np.arange(n), len(spins)), np.int64, (count,)),
                         _frozen(np.concatenate(blocks) if blocks else np.zeros(0), np.float64, (count,)),
                         _frozen(np.full(count, weight), np.float64, (count,)))
    return OneBodyDecomposition(X, ell, tau1, table, shift)


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


# --------------------------------------------------------------------------- caller-supplied channels

def _two_body_table(channels, nchol, nbasis):
    """Validated ChannelTable from a ChannelTable or a sequence of Channel records."""
    if isinstance(channels, ChannelTable):
        bix, spins, modes, d, a = channels.bix, channels.spins, channels.modes, channels.d, channels.a
    else:
        channels = tuple(channels)
        if any(not isinstance(c, Channel) for c in channels):
            raise TypeError("two-body channels must be Channel records or a ChannelTable")
        count = len(channels)
        bix = np.array([c.bix for c in channels], dtype=np.int64).reshape(count)
        spins = np.array([c.spins for c in channels], dtype=np.int64).reshape(count, 2)
        modes = np.array([c.modes for c in channels], dtype=np.int64).reshape(count, 2)
        d = np.array([c.d for c in channels], dtype=np.float64).reshape(count, 2)
        a = np.array([c.a_i for c in channels], dtype=np.float64).reshape(count)
    if np.any(bix < 0) or np.any(bix >= nchol) or np.any(modes < 0) or np.any(modes >= nbasis):
        raise ValueError("Channel basis/mode index is out of range")
    if not np.all((spins == 0) | (spins == 1)):
        raise ValueError("spins must contain 0 (up) or 1 (down)")
    if not np.isfinite(d).all() or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Channel amplitudes must be finite and weights finite and positive")
    return ChannelTable(_frozen(bix, np.int64), _frozen(spins, np.int64), _frozen(modes, np.int64),
                        _frozen(d, np.float64), _frozen(a, np.float64))


def _one_body_table(channels, nbasis):
    if isinstance(channels, OneBodyTable):
        spin, mode, d, a = channels.spin, channels.mode, channels.d, channels.a
    else:
        channels = tuple(channels)
        if any(not isinstance(c, OneBodyChannel) for c in channels):
            raise TypeError("one-body channels must be OneBodyChannel records or a OneBodyTable")
        spin = np.array([c.spin for c in channels], dtype=np.int64)
        mode = np.array([c.mode for c in channels], dtype=np.int64)
        d = np.array([c.d for c in channels], dtype=np.float64)
        a = np.array([c.a_i for c in channels], dtype=np.float64)
    if not np.all((spin == 0) | (spin == 1)) or np.any(mode < 0) or np.any(mode >= nbasis):
        raise ValueError("Invalid one-body channel index")
    if not np.isfinite(d).all() or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("One-body amplitudes must be finite and weights finite and positive")
    return OneBodyTable(_frozen(spin, np.int64), _frozen(mode, np.int64), _frozen(d, np.float64),
                        _frozen(a, np.float64))


def with_channels(decomposition, *, one_body=None, two_body=None):
    """A decomposition with caller-supplied channel lists (the entry point for them).

    one_body / two_body: sequences of OneBodyChannel / Channel records (or
    tables) replacing the corresponding list; None keeps it. The channels are
    validated here (indices, spins, finite amplitudes, positive weights). A
    replaced two-body list makes the decomposition non-canonical, so the
    partition solver and the gauge gradients reject it, while the loss
    kernels, rotation matrices and Fock-space checks accept it. Gauge,
    sources, t and amplitudes are kept for reference.
    """
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    from dataclasses import replace
    one, two = decomposition.one_body, decomposition.two_body
    r, n, _ = two.gauge.X.shape
    if one_body is not None:
        one = replace(one, table=_one_body_table(one_body, n))
    if two_body is not None:
        two = replace(two, table=_two_body_table(two_body, r, n), canonical=False)
    return FixedDecomposition(one, two)


# --------------------------------------------------------------------------- rotation matrices

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
    n_one = len(decomposition.one_body.table)
    if index < n_one:
        return one_body_rotation_matrix(decomposition.one_body.channels[index],
                                         decomposition.one_body)
    return two_body_rotation_matrix(decomposition.two_body.channels[index-n_one],
                                     decomposition.two_body.gauge)
