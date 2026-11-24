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


def leg_usage(sources):
    """(count, 2, 2) mask [source, leg, spin]: the spins each leg carries in some channel.

    A two-branch leg carries its source's spin; a quartet leg carries both.
    """
    used = np.zeros((len(sources), 2, 2), dtype=bool)
    used[sources.channel_source, 0, sources.channel_spins[:, 0]] = True
    used[sources.channel_source, 1, sources.channel_spins[:, 1]] = True
    return used


def leg_values(spin_eigenvalues, sources):
    """(count, 2, 2) per-spin eigenvalues lambda [source, leg, spin] of each source's two modes."""
    lam = np.asarray(spin_eigenvalues, dtype=np.float64)
    bix, modes = sources.bix, sources.modes
    return np.stack([lam[:, bix, modes[:, 0]].T, lam[:, bix, modes[:, 1]].T], axis=1)


def _log_leg_scale(values, used):
    """log of the RMS of |values| over the used spins of each leg; -inf if they all vanish."""
    magnitude = np.where(used, np.abs(values), 0.0)
    peak = np.max(magnitude, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(peak[:, :, None] > 0, magnitude/peak[:, :, None], 0.0)
        mean = np.sum(ratio*ratio, axis=2)/np.maximum(np.sum(used, axis=2), 1)
        return np.where(peak > 0, np.log(peak) + 0.5*np.log(mean), -np.inf)


def leg_log_bases(spin_eigenvalues, sources, log_tau, partition):
    """Per-spin map at t=0: (logs, signs), each (count, 2, 2) = [source, leg, spin].

    d_L,s = lam_L,s kappa_L e^t and d_R,s = -lam_R,s kappa_R e^-t with
    kappa_L kappa_R = tau2, so d_L,s d_R,s' = -lam_L,s lam_R,s' tau2 for every
    spin pair: one t serves a whole quartet. anchored_log: kappa_L = kappa_R =
    sqrt(tau2). balanced_log: kappa_L = sqrt(tau2) (g_R/g_L)^(1/2), where the
    leg scale g is |lam| for a one-spin leg and the RMS over spins for a quartet
    leg; both legs are absent when g_L or g_R vanishes. With equal spins both
    reduce to the shared map (partition_log_bases). -inf marks an absent or
    unused leg; logs are log|d| at t=0.
    """
    values = leg_values(spin_eigenvalues, sources)
    used = leg_usage(sources)
    with np.errstate(divide="ignore"):
        logs = np.where(used, np.log(np.abs(values)), -np.inf)
    if partition == "balanced_log":
        scale = _log_leg_scale(values, used)
        absent = np.any(scale == -np.inf, axis=1)
        with np.errstate(invalid="ignore"):
            offset = 0.5*(scale[:, ::-1] - scale)
            logs = logs + 0.5*log_tau + offset[:, :, None]
        logs[absent] = -np.inf
        logs = np.where(used, logs, -np.inf)
    else:
        logs = logs + 0.5*log_tau
    signs = np.sign(values)
    signs[:, 1, :] *= -1.0
    return logs, signs


def leg_partition_map(logs, signs, t):
    """Per-spin amplitudes (count, 2, 2) at log ratios t, and their logs; no domain checks.

    The split-spin counterpart of partition_map; an absent leg is exactly +0.0.
    """
    t = np.asarray(t, dtype=np.float64)
    shifted = logs + np.stack([t, -t], axis=1)[:, :, None]
    amplitudes = np.zeros(np.shape(logs), dtype=np.float64)
    live = shifted != -np.inf
    with np.errstate(over="ignore"):
        amplitudes[live] = signs[live]*np.exp(shifted[live])
    return amplitudes, shifted


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

    eigenvalues of shape (2, r, n) are per spin (split Cholesky shifts): d is
    then (count, 2, 2) = [source, leg, spin] from leg_log_bases, with zeros
    for spins a leg does not carry.
    """
    if np.ndim(eigenvalues) == 3:
        return _split_partition_amplitudes(eigenvalues, sources, tau2, t, partition, amplitude_bound)
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


def _map_arguments(tau2, amplitude_bound, partition):
    tau2 = positive_scale(tau2, "tau2")
    bound = positive_scale(amplitude_bound, "amplitude_bound")
    if bound > 1:
        raise ValueError("amplitude_bound must be <=1 for paired real exponentials")
    if partition not in ("balanced_log", "anchored_log"):
        raise ValueError("partition must be 'balanced_log' or 'anchored_log'")
    return tau2, bound


def _split_partition_amplitudes(eigenvalues, sources, tau2, t, partition, amplitude_bound):
    """fixed_partition_amplitudes for per-spin eigenvalues (2, r, n): d is (count, 2, 2)."""
    tau2, bound = _map_arguments(tau2, amplitude_bound, partition)
    lam = _real_array(eigenvalues, "eigenvalues")
    if lam.ndim != 3 or lam.shape[0] != 2:
        raise ValueError("per-spin eigenvalues must have shape (2,r,n)")
    sources = source_table(sources)
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
    bix, modes = sources.bix, sources.modes
    if count and (np.any(bix < 0) or np.any(bix >= lam.shape[1])
                  or np.any(modes < 0) or np.any(modes >= lam.shape[2])):
        bad = int(np.flatnonzero((bix < 0) | (bix >= lam.shape[1]) | np.any(modes < 0, axis=1)
                                 | np.any(modes >= lam.shape[2], axis=1))[0])
        raise ValueError(f"Invalid basis or mode indices in source {bad}")
    logs, signs = leg_log_bases(lam, sources, np.log(tau2), partition)
    amplitudes, shifted = leg_partition_map(logs, signs, ratios)
    flat_logs, flat = shifted.reshape(count, 4), amplitudes.reshape(count, 4)
    over = np.max(flat_logs, axis=1, initial=-np.inf) >= np.log(bound)
    underflow = np.any((flat_logs != -np.inf) & (flat == 0), axis=1)
    outside = ~np.isfinite(flat).all(axis=1) | (np.max(np.abs(flat), axis=1, initial=0.0) >= bound)
    _report_domain(over, underflow, outside, bound, bix, modes)
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
    _report_domain(over, underflow, outside, bound, bix, modes)


def _report_domain(over, underflow, outside, bound, bix, modes):
    """Raise for the first failing source, in the order over, underflow, outside."""
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
    """ChannelTable of the canonical layout: channel c = sign_c*(d_left, d_right) of its source.

    Per-spin amplitudes (count, 2, 2) give each channel the values of its own leg spins.
    """
    src = sources.channel_source
    sign = sources.channel_sign
    d = np.empty((src.shape[0], 2), dtype=np.float64)
    if np.ndim(amplitudes) == 3:
        spins = sources.channel_spins
        d[:, 0] = sign*amplitudes[src, 0, spins[:, 0]]
        d[:, 1] = sign*amplitudes[src, 1, spins[:, 1]]
    else:
        d[:, 0] = sign*amplitudes[src, 0]
        d[:, 1] = sign*amplitudes[src, 1]
    modes = sources.modes[src]
    return ChannelTable(_frozen(sources.bix[src], np.int64), _frozen(sources.channel_spins, np.int64),
                        _frozen(modes, np.int64), _frozen(d, np.float64),
                        _frozen(sources.channel_weight/tau2, np.float64))


def _two_body_from_amplitudes(gauge, policy, nelec, tau2, partition, amplitude_bound,
                              sources, t, amplitudes, cholesky_shift=None):
    """Assemble a canonical TwoBodyDecomposition from per-source log ratios and amplitudes."""
    return TwoBodyDecomposition(gauge, policy, nelec, tau2, partition, float(amplitude_bound),
                                sources, t, amplitudes, canonical_table(sources, amplitudes, tau2), True,
                                cholesky_shift)


# --------------------------------------------------------------------------- Cholesky shifts

def cholesky_shift_array(cholesky_shift, nchol, policy=None):
    """Canonical read-only (r, 2) Cholesky shifts mu[gamma, sigma], or None for no shift.

    A length-r vector is a tied shift (the same mu_gamma for both spins); an
    (r, 2) array gives (mu_up, mu_down) for each rotated factor. All zeros
    means no shift and gives None, so a zero shift runs exactly today's code.
    For aa_only, which has no down-spin channels, a length-r vector shifts the
    up spin only (mu_down = 0).
    """
    if cholesky_shift is None:
        return None
    mu = _real_array(cholesky_shift, "cholesky_shift")
    if mu.shape == (nchol,):
        mu = np.stack([mu, np.zeros_like(mu) if policy == "aa_only" else mu], axis=1)
    elif mu.shape != (nchol, 2):
        raise ValueError(f"cholesky_shift must have shape ({nchol},) or ({nchol}, 2); got {mu.shape}")
    if not np.any(mu):
        return None
    return _frozen(mu, np.float64)


def check_cholesky_shift(mu, policy, nelec):
    """A shift needs particle numbers; aa_only has no down-spin channels, so mu[:, 1] must be 0."""
    if mu is None:
        return
    if nelec is None:
        raise ValueError("A Cholesky shift requires particle numbers: pass nelec to prepare_problem "
                         "(h_sigma and E_c depend on N_up and N_down)")
    if policy == "aa_only" and np.any(np.asarray(mu)[:, 1] != 0):
        raise ValueError("This policy emits no down-spin channels; cholesky_shift[:, 1] must be 0")


def split_cholesky_shift(mu):
    """True when the shift differs between the spins of some factor."""
    return mu is not None and not np.array_equal(mu[:, 0], mu[:, 1])


def map_eigenvalues(eigenvalues, mu, per_spin=False):
    """Partition-map eigenvalues: ell without a shift, ell - mu_gamma (r, n) when tied, (2, r, n) when split.

    per_spin=True always gives the (2, r, n) per-spin eigenvalues, also for a
    tied or absent shift (the representation a per-spin shift gradient needs).
    """
    if mu is None and not per_spin:
        return eigenvalues
    ell = np.asarray(eigenvalues, dtype=np.float64)
    if per_spin or split_cholesky_shift(mu):
        if mu is None:
            return np.stack([ell, ell])
        return ell[None, :, :] - np.asarray(mu).T[:, :, None]
    return ell - np.asarray(mu)[:, 0][:, None]


def one_body_matrices(h, chol, mu, nelec):
    """(h_up, h_down) with h_s = h + sum_g (m_g - mu[g,s]) L_g and m_g = sum_s N_s mu[g,s].

    chol holds the rotated factors L_g (unshifted). With the shifted factors
    L_g - mu[g,s] I this reproduces H on the (N_up, N_down) sector up to the
    constant FixedDecomposition.cholesky_energy. A tied shift gives
    h + (N-1) sum_g mu_g L_g for both spins.
    """
    mu = np.asarray(mu, dtype=np.float64)
    m = mu @ np.asarray(nelec, dtype=np.float64)
    return tuple(h + np.einsum("g,gpq->pq", m - mu[:, s], chol) for s in (0, 1))


def build_two_body_channels(problem, gauge, tau2, t=None, *, partition="balanced_log",
                            amplitude_bound=1.0, cholesky_shift=None, per_spin=False):
    """Build a reference two-body decomposition at a supplied validated gauge.

    t is indexed by enumerate_sources(problem), not by individual channels.
    One "two_branch" source emits two equal-weight +/- branches; an off-diagonal
    hybrid source emits a shared-amplitude four-spin quartet. The gauge is
    validated here against the original tensors (reusing the problem's input
    report). GaugeEvaluator builds the same channels without these checks.

    cholesky_shift: None, (r,) tied or (r, 2) per-spin shifts of the rotated
    factors (cholesky_shift_array); the channels then use ell - mu. A split
    shift stores per-spin amplitudes (TwoBodyDecomposition.split);
    per_spin=True does so for any shift (needed for a per-spin shift gradient).
    """
    policy = _problem_policy(problem)
    tau2 = positive_scale(tau2, "tau2")
    validate_gauge_state(problem.data, gauge, problem.report.options, input_report=problem.report)
    mu = cholesky_shift_array(cholesky_shift, problem.data.nchol, policy)
    check_cholesky_shift(mu, policy, problem.nelec)
    sources = enumerate_sources(problem)
    ratios, amplitudes = fixed_partition_amplitudes(
        map_eigenvalues(gauge.eigenvalues, mu, per_spin), sources, tau2, t, partition, amplitude_bound)
    weight = 1.0 / tau2
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("tau2 gives a nonrepresentable positive channel coefficient")
    return _two_body_from_amplitudes(gauge, policy, problem.nelec, tau2, partition,
                                     amplitude_bound, sources, ratios, amplitudes, mu)


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


def _one_body_from_eigensystem(X, ell, tau1, shift, spins, nelec, cholesky_shift=None):
    """Channels d[k,sigma]=(mu_sigma-epsilon_k)*tau1 on a fixed eigensystem.

    ell (n,) is shared by the spins; ell (2, n) with X (2, n, n) is per spin.
    """
    shift = shift_pair(shift)
    if nelec is None and any(shift):
        raise ValueError("A nonzero one-body shift requires particle numbers: pass nelec "
                         "to prepare_problem (sum_sigma mu_sigma N_sigma must be a constant)")
    if 1 not in spins and shift[1] != 0:
        raise ValueError("This policy emits no down-spin one-body channels; mu_down must be 0")
    weight = 1.0 / tau1
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError("tau1 gives a nonrepresentable positive coefficient")
    per_spin = np.ndim(ell) == 2
    n = np.shape(ell)[-1]
    blocks = []
    for s in spins:
        ds = (shift[s] - (ell[s] if per_spin else ell)) * tau1
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
    return OneBodyDecomposition(X, ell, tau1, table, shift, cholesky_shift)


def one_body_eigensystem(h, chol, mu, nelec, spins, per_spin=False):
    """(eigenvalues, X) of the one-body matrices for Cholesky shift mu (None: bare h).

    Per spin (2, n), (2, n, n) when both spins carry channels and the shift is
    split (or per_spin=True); otherwise one shared eigensystem (n,), (n, n),
    of the matrix of the first spin that carries channels. Read-only arrays.
    """
    both = len(spins) == 2
    if mu is None:
        ell, X = np.linalg.eigh(h)
        if per_spin and both:
            ell, X = np.stack([ell, ell]), np.stack([X, X])
    else:
        matrices = one_body_matrices(h, chol, mu, nelec)
        if both and (per_spin or split_cholesky_shift(mu)):
            pairs = [np.linalg.eigh(m) for m in matrices]
            ell, X = np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
        else:
            ell, X = np.linalg.eigh(matrices[spins[0]])
    ell.setflags(write=False)
    X.setflags(write=False)
    return ell, X


def build_one_body_channels(problem, tau1, shift=None, *, gauge=None, cholesky_shift=None, per_spin=False):
    """Jointly diagonalize bare h and retain one channel per mode and spin.

    d=(mu_sigma-epsilon)*tau1, weight=1/tau1, with shift=(mu_up, mu_down)
    defaulting to zero. Zero amplitudes are retained as identity channels.
    Only up spin is emitted for aa_only (mu_down must then be 0). Real
    exponentials require d>-1; positive d>=1 is allowed because this identity
    has no minus branch. A nonzero shift requires problem.nelec, since
    sum_sigma mu_sigma N_sigma is folded into Lambda.

    cholesky_shift (with the gauge whose rotated factors it shifts)
    diagonalizes h_sigma = h + sum_g (m_g - mu[g,sigma]) L_g instead of h
    (one_body_matrices): one eigensystem for a tied shift, one per spin for a
    split shift or with per_spin=True (when both spins carry channels).
    """
    policy = _problem_policy(problem)
    tau1 = positive_scale(tau1, "tau1")
    h = _real_array(problem.data.h1, "h1")
    n = problem.data.nbasis
    if h.shape != (n,n):
        raise ValueError("h1 must be square")
    _require_report(ValidationReport({"h1_symmetry": _metric(h, h.T, problem.report.options)},
                                     problem.report.options))
    mu = cholesky_shift_array(cholesky_shift, problem.data.nchol, policy)
    check_cholesky_shift(mu, policy, problem.nelec)
    if mu is not None and gauge is None:
        raise ValueError("A Cholesky shift needs the gauge whose rotated factors it shifts")
    spins = (0,) if policy == "aa_only" else (0,1)
    ell, X = one_body_eigensystem(h, None if mu is None else gauge.chol, mu, problem.nelec, spins, per_spin)
    return _one_body_from_eigensystem(X, ell, tau1, shift, spins, problem.nelec, mu)


def one_body_at_shift(one_body, shift, nelec):
    """Rebuild one-body channels at a new shift, reusing the cached eigensystem."""
    if not isinstance(one_body, OneBodyDecomposition):
        raise TypeError("one_body must be a OneBodyDecomposition")
    return _one_body_from_eigensystem(one_body.X, one_body.eigenvalues, one_body.tau1,
                                      shift, one_body.spins, nelec, one_body.cholesky_shift)


def build_fixed_decomposition(problem, gauge, tau1, tau2, t=None, *,
                              partition="balanced_log", amplitude_bound=1.0, shift=None,
                              cholesky_shift=None, per_spin=False):
    """Build both lists with the same policy, without normalizing either list.

    cholesky_shift: None, (r,) tied or (r, 2) per-spin shifts of the gauge's
    rotated factors; see build_two_body_channels and build_one_body_channels.
    per_spin=True stores both lists per spin even for a tied or absent shift.
    On the (N_up, N_down) sector sum_i a_i U_i = lambda_electronic I - H.
    """
    two_body = build_two_body_channels(problem, gauge, tau2, t, partition=partition,
                                       amplitude_bound=amplitude_bound, cholesky_shift=cholesky_shift,
                                       per_spin=per_spin)
    one_body = build_one_body_channels(problem, tau1, shift, gauge=gauge, cholesky_shift=cholesky_shift,
                                       per_spin=per_spin)
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
    """Single rank-one factor in its spin's h eigenbasis."""
    n = one_body.spin_X(0).shape[0]
    if channel.spin not in (0,1) or not 0 <= channel.mode < n:
        raise ValueError("Invalid one-body channel index")
    x = one_body.spin_X(channel.spin)[:, channel.mode]
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
