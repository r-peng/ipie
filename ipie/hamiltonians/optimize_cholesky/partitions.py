"""Fixed-gauge partition (and optional shift) optimization of L_raw or L_w.

Each source owns one t, shared by its two branches or four-spin quartet.
With optimize_shift=True the spin-resolved one-body shifts (mu_up, mu_down)
are inner variables too: mu starts at mu*(t0), and every sweep updates all
t's, then each mu exactly.

The solver works in mode space (loss.py): a source's legs load the modes of
its own Cholesky factor, and pairs within one basis commute, so given the
loads of every other basis the loss is affine in a factor's loads. Hence all
sources of one factor decouple, each with the exact profile
C + A exp(2t) + B exp(-2t), and one block step minimizes all of them at once.
A sweep visits the factors in order (the same iterates as a sequential
source-by-source sweep), then the shift. No ERIs, eigendecompositions or
rotations are rebuilt.
"""

from dataclasses import replace
import math
from numbers import Integral

import numpy as np

from .channels import (
    canonical_table,
    fixed_partition_amplitudes,
    partition_log_bases,
    partition_map,
    positive_scale,
    source_eigenvalues,
)
from .models import (
    FixedDecomposition,
    InnerOptions,
    PartitionCoefficients,
    PartitionProfiles,
    GaugeFeasibilityError,
    PartitionOptimizationResult,
    PartitionSweep,
    _frozen,
)
from .loss import ModeSpace, channel_loads, check_bases, loss_kind_name
from .problem import _real_array
from .channels import shift_pair
from .shift import (
    check_one_body_shift,
    one_body_limit_value,
    shift_bounds,
    shift_profiles,
    shift_tolerance,
    with_shift,
)


def _readonly(array):
    result = np.array(array, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _validate_partition_state(decomposition):
    """Check the canonical source-sharing contract once, without ERI checks.

    The decomposition must be canonical: its stored amplitudes must be the
    partition map at its t, and its channel table the canonical layout of its
    sources at those amplitudes. Decompositions from channels.with_channels
    are rejected here.
    """
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    two = decomposition.two_body
    count = len(two.sources)
    if len(two.source_slices) != count or np.shape(two.t) != (count,):
        raise ValueError("Source slices and t must have one entry per source")
    t, amplitudes = fixed_partition_amplitudes(
        two.gauge.eigenvalues, two.sources, two.tau2, two.t,
        two.partition, two.amplitude_bound)
    if np.shape(two.amplitudes) != (count, 2) or not np.array_equal(two.amplitudes, amplitudes):
        raise ValueError("Stored amplitudes disagree with the source partition map; rebuild channels")
    if not two.canonical:
        raise ValueError("Channels disagree with the shared source amplitudes/branch order: "
                         "the decomposition is not canonical (built with with_channels)")
    if two.table != canonical_table(two.sources, amplitudes, two.tau2):
        raise ValueError("Channels disagree with the shared source amplitudes/branch order")
    return t, amplitudes


def with_partitions(decomposition, t):
    """Return new step-2 channels at t, preserving gauge, h and all weights.

    Validates the source map and open amplitude domain but does not repeat
    the original input/ERI checks. The input snapshot is never modified.
    """
    _validate_partition_state(decomposition)
    return _with_partitions(decomposition, t)


def _with_partitions(decomposition, t):
    """with_partitions without re-checking the input decomposition."""
    two = decomposition.two_body
    ratios, amplitudes = fixed_partition_amplitudes(
        two.gauge.eigenvalues, two.sources, two.tau2, t,
        two.partition, two.amplitude_bound)
    return replace(decomposition, two_body=replace(
        two, t=ratios, amplitudes=amplitudes,
        table=canonical_table(two.sources, amplitudes, two.tau2), canonical=True))


def partition_bounds(decomposition, *, amplitude_limit=None, max_abs_t=30.0):
    """Closed, fixed-gauge feasible intervals, shape (number_of_sources,2).

    With base magnitudes |d_L(0)|, |d_R(0)|, intersect
      log(|d_R(0)|/rho) <= t <= log(rho/|d_L(0)|)
    with [-max_abs_t,max_abs_t]. A zero leg supplies no amplitude bound.
    rho defaults to (1-1e-8)*two_body.amplitude_bound and must be strictly
    below that open-domain bound. max_abs_t is a finite numerical box:
    for anchored_log at W=0 a surviving leg otherwise reaches its infimum
    only at infinite t. Such box-limited results are visible in bounds.

    The initial t must be feasible for optimize_partitions; this function
    only constructs bounds. It never clips individual amplitudes.
    """
    _validate_partition_state(decomposition)
    two = decomposition.two_body
    return _partition_intervals(two.gauge.eigenvalues, two.sources, two.tau2,
                                two.partition, two.amplitude_bound,
                                amplitude_limit=amplitude_limit, max_abs_t=max_abs_t)


def _partition_limits(amplitude_bound, amplitude_limit, max_abs_t):
    cap = positive_scale(max_abs_t, "max_abs_t")
    rho = (1.0-1e-8)*amplitude_bound if amplitude_limit is None else positive_scale(
        amplitude_limit, "amplitude_limit")
    if not 0 < rho < amplitude_bound or not np.log(rho) < np.log(amplitude_bound):
        raise ValueError("amplitude_limit must be strictly below the open amplitude_bound")
    return rho, cap


def interval_logs(eigenvalues, sources, tau2, partition, log_rho):
    """Uncapped interval ends log(|d_R(0)|/rho) and log(rho/|d_L(0)|) of every source.

    -inf (lower) / +inf (upper) mark a leg that supplies no amplitude bound.
    Shared by the partition intervals and the bound pullback.
    """
    left, right = source_eigenvalues(eigenvalues, sources)
    log_left, log_right = partition_log_bases(left, right, math.log(tau2), partition)
    return log_right - log_rho, log_rho - log_left


def _partition_intervals(eigenvalues, sources, tau2, partition, amplitude_bound, *,
                         amplitude_limit=None, max_abs_t=30.0):
    """Same bounds before a feasible starting decomposition has been built."""
    rho, cap = _partition_limits(amplitude_bound, amplitude_limit, max_abs_t)
    lower_amp, upper_amp = interval_logs(eigenvalues, sources, tau2, partition, math.log(rho))
    bounds = np.empty((len(sources), 2))
    bounds[:, 0] = np.maximum(-cap, lower_amp)
    bounds[:, 1] = np.minimum(cap, upper_amp)
    infeasible = bounds[:, 0] > bounds[:, 1]
    if np.any(infeasible):
        i = int(np.flatnonzero(infeasible)[0])
        raise GaugeFeasibilityError(f"Source {i} has no feasible partition within the requested bounds; "
                                    "reduce tau2, increase amplitude_limit or increase max_abs_t")
    return _readonly(bounds)


def analytic_partition_update(coefficients, bounds):
    """Exact bounded coordinate minimum; flat coordinates keep their value.

    For A,B>0, t*=log(B/A)/4. A=0 chooses the upper bound, B=0 the
    lower bound. Centered terms avoid forming extreme unscaled A or B.
    Bounds must be finite: infinite endpoints need not attain a minimum.
    """
    if not isinstance(coefficients, PartitionCoefficients):
        raise TypeError("coefficients must be PartitionCoefficients")
    interval = _real_array(bounds, "bounds")
    if interval.shape != (2,) or interval[0] > interval[1]:
        raise ValueError("bounds must be a finite (lower,upper) interval")
    left, right = coefficients.left_term, coefficients.right_term
    if not all(math.isfinite(x) and x >= 0 for x in (left, right)):
        raise ValueError("Coordinate coefficients must be finite and nonnegative")
    if left == right == 0:
        optimal = coefficients.t
    elif left == 0:
        optimal = interval[1]
    elif right == 0:
        optimal = interval[0]
    else:
        optimal = coefficients.t+0.25*(math.log(right)-math.log(left))
    return float(np.clip(optimal, interval[0], interval[1]))


def projected_partition_gradient(t, gradient, bounds):
    """KKT residual in t coordinates: remove gradients pointing out of a wall.

    At the lower wall retain min(g,0); at the upper wall retain max(g,0).
    At a fixed coordinate (lower=upper) the residual is zero. No relative
    loss-change test substitutes for this stationarity test.
    """
    t = _real_array(t, "t")
    gradient = _real_array(gradient, "gradient")
    bounds = _real_array(bounds, "bounds")
    if t.ndim != 1 or gradient.shape != t.shape or bounds.shape != (t.size,2):
        raise ValueError("Expected t,gradient vectors and bounds of shape (len(t),2)")
    if np.any(bounds[:,0] > bounds[:,1]) or np.any(t < bounds[:,0]) or np.any(t > bounds[:,1]):
        raise ValueError("t must be feasible for ordered closed bounds")
    result = gradient.copy()
    result[(t == bounds[:,0]) & (gradient > 0)] = 0.0
    result[(t == bounds[:,1]) & (gradient < 0)] = 0.0
    result[bounds[:,0] == bounds[:,1]] = 0.0
    return _readonly(result)


class _InnerState:
    """Mutable t, amplitudes, shift and mode loads of one canonical inner solve."""

    def __init__(self, decomposition, loss_kind, space=None):
        two, one = decomposition.two_body, decomposition.one_body
        self.decomposition = decomposition
        self.loss_kind = loss_kind
        # The mode space depends on the bases only (gauge and h), not on t or mu.
        self.space = space = ModeSpace.of(decomposition) if space is None else space
        sources = two.sources
        self.ell = np.asarray(two.gauge.eigenvalues, dtype=np.float64)
        self.tau2, self.partition = float(two.tau2), two.partition
        self.t = np.array(two.t, dtype=np.float64, copy=True)
        self.amplitudes = np.array(two.amplitudes, dtype=np.float64, copy=True)
        self.src_bix, self.src_modes = sources.bix, sources.modes
        self.src_left, self.src_right = source_eigenvalues(self.ell, sources)
        self.log_tau = math.log(self.tau2)
        # Canonical layout per channel: owning source, branch sign, spins, global modes, pair scale.
        self.src, self.sign, self.spins = sources.channel_source, sources.channel_sign, sources.channel_spins
        self.gm = sources.bix[self.src][:, None]*space.n + sources.modes[self.src]
        weighted = loss_kind == "weighted"
        self.scale = two.table.a if weighted else np.ones(len(two.table))
        g = np.arange(space.nchol)
        s_lo = np.searchsorted(sources.bix, g, "left")
        s_hi = np.searchsorted(sources.bix, g, "right")
        self.factor_sources = list(zip(s_lo.tolist(), s_hi.tolist()))
        self.factor_channels = [(int(sources.slices[lo, 0]) if hi > lo else 0,
                                 int(sources.slices[hi-1, 1]) if hi > lo else 0)
                                for lo, hi in self.factor_sources]
        # Per-factor channel data in the factor's local (spin, mode) index space, for the block steps.
        n = space.n
        local = self.spins*n + sources.modes[self.src]                     # (channels, 2) in [0, 2n)
        self.factor_blocks = []
        for (lo, hi), (c_lo, c_hi) in zip(self.factor_sources, self.factor_channels):
            self.factor_blocks.append((self.src[c_lo:c_hi] - lo, self.sign[c_lo:c_hi, None],
                                       self.scale[c_lo:c_hi], local[c_lo:c_hi, 0], local[c_lo:c_hi, 1]))
        # One-body channels.
        self.one_body = one
        self.ob_spin, self.ob_mode = one.table.spin, one.table.mode
        self.ob_scale = one.table.a if weighted else np.ones(len(one.table))
        self.eps, self.tau1 = np.asarray(one.eigenvalues, dtype=np.float64), float(one.tau1)
        self.shift = list(shift_pair(one.shift))
        self.shift_changed = False
        # Loads; the off-diagonal part (h, OO) does not depend on t or mu.
        self.q, O = channel_loads(decomposition, loss_kind, space)
        self.h, self.OO = space.offdiagonal_terms(O)

    def loss(self):
        return self.space.loss(self.q, self.h, self.OO)

    def source_terms(self, field, channels=None, sources=None):
        """(left, right) of each source: loads of its +t and -t legs times the field."""
        lo, hi = (0, self.src.shape[0]) if channels is None else channels
        s_lo, s_hi = (0, self.t.shape[0]) if sources is None else sources
        src = self.src[lo:hi]
        d = self.sign[lo:hi, None]*self.amplitudes[src]
        s = self.scale[lo:hi]
        G = np.maximum(field, 0.0)                     # dL/dq >= 0; clamp roundoff
        left = s*d[:, 0]*d[:, 0]*G[self.spins[lo:hi, 0], self.gm[lo:hi, 0]]
        right = s*d[:, 1]*d[:, 1]*G[self.spins[lo:hi, 1], self.gm[lo:hi, 1]]
        return (np.bincount(src-s_lo, weights=left, minlength=s_hi-s_lo),
                np.bincount(src-s_lo, weights=right, minlength=s_hi-s_lo))

    def set_amplitudes(self, lo, hi, t):
        """The partition map (channels.partition_map) for sources [lo, hi) at log ratios t."""
        self.amplitudes[lo:hi] = partition_map(self.src_left[lo:hi], self.src_right[lo:hi], t,
                                               self.log_tau, self.partition)[0]
        self.t[lo:hi] = t

    def factor_terms(self, g, field):
        """(left, right) of factor g's sources from the factor's own field columns (2, n)."""
        lo, hi = self.factor_sources[g]
        src, sign, s, m0, m1 = self.factor_blocks[g]
        d = sign*self.amplitudes[lo:hi][src]
        G = np.maximum(field, 0.0).reshape(-1)        # dL/dq >= 0; clamp roundoff
        left = s*d[:, 0]*d[:, 0]*G[m0]
        right = s*d[:, 1]*d[:, 1]*G[m1]
        return (np.bincount(src, weights=left, minlength=hi-lo),
                np.bincount(src, weights=right, minlength=hi-lo))

    def factor_loads(self, g):
        """Recompute the loads of Cholesky factor g from its channels."""
        n, columns = self.space.n, self.space.factor(g)
        lo, hi = self.factor_sources[g]
        if hi == lo:
            self.q[:, columns] = 0.0
            return
        src, sign, s, m0, m1 = self.factor_blocks[g]
        d = sign*self.amplitudes[lo:hi][src]
        loads = np.bincount(m0, weights=s*d[:, 0]**2, minlength=2*n)
        loads += np.bincount(m1, weights=s*d[:, 1]**2, minlength=2*n)
        self.q[:, columns] = loads.reshape(2, n)

    def shift_profiles(self, lower):
        one_body = replace(self.one_body, shift=tuple(self.shift))
        field = np.maximum(self.space.field(self.q, self.h, self.space.hslice), 0.0)
        return shift_profiles(one_body, field, self.ob_scale, lower)

    def set_shift(self, shift):
        """One-body loads at a new (mu_up, mu_down); the domain 1+d>0 is checked."""
        d = (np.asarray(shift, dtype=np.float64)[self.ob_spin] - self.eps[self.ob_mode])*self.tau1
        if not np.isfinite(d).all() or np.any(1.0+d <= 0):
            raise GaugeFeasibilityError("Shift leaves the one-body real-exponential domain 1+d>0")
        n = self.space.n
        self.q[:, self.space.hslice] = np.bincount(self.ob_spin*n + self.ob_mode, weights=self.ob_scale*d*d,
                                                   minlength=2*n).reshape(2, n)
        self.shift = [float(mu) for mu in shift]
        self.shift_changed = True

    def snapshot(self):
        result = _with_partitions(self.decomposition, self.t)
        return with_shift(result, tuple(self.shift)) if self.shift_changed else result


def _profile_change(left, right, step):
    """PartitionCoefficients.change for steps from the reference t (vectorized)."""
    exponent = 2.0*step
    small = np.abs(exponent) < 0.5
    out = np.empty_like(step)
    out[small] = left[small]*np.expm1(exponent[small]) + right[small]*np.expm1(-exponent[small])
    big = ~small
    out[big] = left[big]*(np.exp(exponent[big]) - 1.0) + right[big]*(np.exp(-exponent[big]) - 1.0)
    return out


def _analytic_updates(t, left, right, bounds):
    """analytic_partition_update for many sources at once."""
    optimal = t.copy()
    both = (left > 0) & (right > 0)
    optimal[both] = t[both] + 0.25*(np.log(right[both]) - np.log(left[both]))
    only_right = (left == 0) & (right > 0)
    only_left = (right == 0) & (left > 0)
    optimal[only_right] = bounds[only_right, 1]
    optimal[only_left] = bounds[only_left, 0]
    return np.clip(optimal, bounds[:, 0], bounds[:, 1])


def _profiles(t, loss, left, right):
    """PartitionProfiles with C = loss - left - right."""
    count = t.shape[0]
    return PartitionProfiles(_frozen(t, np.float64, (count,)), _frozen((loss-left)-right, np.float64, (count,)),
                             _frozen(left, np.float64, (count,)), _frozen(right, np.float64, (count,)))


def partition_coefficients(decomposition, source_index=None, *, loss_kind="weighted"):
    """A_i,B_i,C_i for one source (PartitionCoefficients), or PartitionProfiles for all.

    Other t's are held at their CURRENT values. The terms include every
    external one-/two-body channel with its actual spin and multiplicity.
    For loss_kind='raw' no a_i weights enter; for 'weighted' every pair
    carries a_i*a_j. C is the full loss minus the two nonnegative
    current-coordinate terms, and may be negative.
    """
    loss_kind = loss_kind_name(loss_kind)
    _validate_partition_state(decomposition)
    check_bases(decomposition)
    state = _InnerState(decomposition, loss_kind)
    loss = state.loss()
    left, right = state.source_terms(state.space.field(state.q, state.h))
    if source_index is None:
        return _profiles(state.t, loss, left, right)
    if isinstance(source_index, (bool, np.bool_)) or not isinstance(source_index, Integral):
        raise TypeError("source_index must be an integer")
    if not 0 <= source_index < state.t.shape[0]:
        raise IndexError("source_index is out of range")
    i = int(source_index)
    return PartitionCoefficients(i, float(state.t[i]), math.fsum((loss, -float(left[i]), -float(right[i]))),
                                 float(left[i]), float(right[i]))


def _inner_options(options):
    if options is None:
        return InnerOptions()
    if not isinstance(options, InnerOptions):
        raise TypeError("options must be InnerOptions")
    return options


def optimize_partitions(decomposition, options=None, *, loss_kind="weighted"):
    """Exact block coordinate updates (factor by factor), at a fixed Cholesky gauge.

    options: InnerOptions (defaults when None), the same settings the outer
    optimizer uses. loss_kind selects the unnormalized L_w ('weighted', default)
    or L_raw ('raw'). Stop only when ||projected_gradient_t||_inf <=
    gradient_atol + gradient_rtol*initial_projected_gradient_norm, evaluated
    AFTER a full sweep with every other coordinate at its latest value.
    max_sweeps=0 checks the input without updating it. Loss changes are
    diagnostics.

    optimize_shift=True also relaxes (mu_up, mu_down). mu is first set to its
    exact minimizer mu*(t0) (so no mu seed enters), and each sweep updates all
    t's and then each mu exactly. Since mu is then exactly optimal at the
    current t whenever the test is made, its gradient vanishes (or is
    wall-filtered at mu >= max(eps)-one_body_limit/tau1) and needs no
    threshold. The extra shift test requires the last sweep to have moved every
    mu by at most shift_atol + shift_rtol*(max(eps)-min(eps)). A nonzero shift
    requires particle numbers on the decomposition.

    A maximum-sweep exit returns converged=False, status='max_sweeps'; it
    is not silently treated as convergence. The decomposition is checked once
    here; GaugeEvaluator, which builds its own, skips the check.
    """
    options = _inner_options(options)
    loss_kind = loss_kind_name(loss_kind)
    _validate_partition_state(decomposition)
    check_one_body_shift(decomposition.one_body)
    check_bases(decomposition)
    return _solve_partitions(decomposition, options, loss_kind)


def _solve_partitions(decomposition, options, loss_kind, space=None):
    """optimize_partitions on a trusted canonical decomposition with checked options.

    space: the decomposition's ModeSpace if the caller already built it.
    """
    max_sweeps = options.max_sweeps
    atol, rtol = float(options.gradient_atol), float(options.gradient_rtol)
    amplitude_limit, max_abs_t = options.amplitude_limit, options.max_abs_t
    optimize_shift, one_body_limit = bool(options.optimize_shift), options.one_body_limit
    shift_atol, shift_rtol = options.shift_atol, options.shift_rtol
    rho1 = one_body_limit_value(one_body_limit)
    lower = shift_bounds(decomposition.one_body, rho1)
    shift_tol = shift_tolerance(decomposition.one_body.eigenvalues, shift_atol, shift_rtol)
    if optimize_shift and decomposition.two_body.nelec is None:
        raise ValueError("optimize_shift requires particle numbers: pass nelec to prepare_problem")
    two = decomposition.two_body
    bounds = _partition_intervals(two.gauge.eigenvalues, two.sources, two.tau2, two.partition,
                                  two.amplitude_bound, amplitude_limit=amplitude_limit,
                                  max_abs_t=max_abs_t)
    state = _InnerState(decomposition, loss_kind, space)
    if np.any(state.t < bounds[:,0]) or np.any(state.t > bounds[:,1]):
        raise ValueError("Initial t lies outside the optimizer bounds; use feasible initial partitions "
                         "or increase amplitude_limit/max_abs_t")
    input_loss = state.loss()

    def relax_shift():
        """Exact mu step at the current t; returns the largest |change|."""
        new = list(state.shift)
        for profile in state.shift_profiles(lower):
            new[profile.spin] = profile.update
        change = max(abs(a-b) for a, b in zip(new, state.shift))
        state.set_shift(new)
        return change

    def shift_gradient():
        result = [0.0, 0.0]
        for profile in state.shift_profiles(lower):
            result[profile.spin] = float(profile.projected_gradient)
        return tuple(result)

    if optimize_shift:
        # mu*(t0): the starting mu is determined by t0, never by a separate seed.
        relax_shift()
    loss = state.loss()

    def diagnostics():
        left, right = state.source_terms(state.space.field(state.q, state.h))
        gradient = _readonly(2.0*(left-right))
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite partition gradient")
        projected = projected_partition_gradient(state.t, gradient, bounds)
        return left, right, gradient, projected, float(np.max(np.abs(projected), initial=0.0))

    left, right, gradient, projected, norm = diagnostics()
    initial_loss = loss
    threshold = atol+rtol*norm
    if not math.isfinite(threshold):
        raise ValueError("Gradient stopping threshold is not representable")
    history = [PartitionSweep(0, loss, norm, 0.0, 0.0, 0, 0.0)]
    converged = norm <= threshold
    tiny = np.finfo(float).tiny
    for sweep in range(1, max_sweeps+1):
        if converged:
            break
        previous_loss = loss
        updates, max_step, shift_change = 0, 0.0, 0.0
        for g, (s_lo, s_hi) in enumerate(state.factor_sources):
            if s_hi == s_lo:
                continue
            field = state.space.field(state.q, state.h, state.space.factor(g))
            lf, rt = state.factor_terms(g, field)
            current = state.t[s_lo:s_hi]
            optimal = _analytic_updates(current, lf, rt, bounds[s_lo:s_hi])
            step = optimal-current
            change = _profile_change(lf, rt, step)
            roundoff = 128*np.finfo(float).eps*np.maximum(np.maximum(loss, lf), np.maximum(rt, tiny))
            if np.any((step != 0) & (change > roundoff)):
                raise FloatingPointError("Analytic coordinate update increased the predicted loss")
            accept = (step != 0) & (change <= 0)
            if not np.any(accept):
                continue
            state.set_amplitudes(s_lo, s_hi, np.where(accept, optimal, current))
            state.factor_loads(g)
            loss = math.fsum((loss, math.fsum(change[accept].tolist())))
            max_step = max(max_step, float(np.max(np.abs(step[accept]))))
            updates += int(np.count_nonzero(accept))
        if optimize_shift:
            # t's first, then mu: at the check below mu is exactly optimal at the latest t.
            shift_change = relax_shift()
        # Refresh the complete loss once per sweep.
        loss = state.loss()
        roundoff = 512*np.finfo(float).eps*max(previous_loss, loss, tiny)
        if loss > previous_loss+roundoff:
            raise FloatingPointError(f"Sequential sweep increased the full {loss_kind} loss")
        left, right, gradient, projected, norm = diagnostics()
        history.append(PartitionSweep(sweep, loss, norm, previous_loss-loss, max_step, updates, shift_change))
        converged = norm <= threshold and (not optimize_shift or shift_change <= shift_tol)

    # Snapshot through the checked partition map (the same values the state holds),
    # then report everything at the snapshot.
    snapshot = state.snapshot()
    final = _InnerState(snapshot, loss_kind, state.space)      # same gauge and h basis
    loss = final.loss()
    left, right = final.source_terms(final.space.field(final.q, final.h))
    gradient = _readonly(2.0*(left-right))
    projected = projected_partition_gradient(final.t, gradient, bounds)
    state = final                                      # shift_gradient() reports at the snapshot
    return PartitionOptimizationResult(
        snapshot, initial_loss, loss, bounds, _profiles(final.t, loss, left, right),
        gradient, projected, threshold, tuple(history), bool(converged),
        "converged" if converged else "max_sweeps",
        *_partition_limits(decomposition.two_body.amplitude_bound, amplitude_limit, max_abs_t),
        loss_kind=loss_kind, optimize_shift=optimize_shift, shift_bounds=lower,
        shift_gradient=shift_gradient() if optimize_shift else None, shift_tolerance=shift_tol, input_loss=input_loss,
        one_body_limit=one_body_limit, shift_atol=float(shift_atol), shift_rtol=float(shift_rtol))
