"""Fixed-gauge partition (and optional shift) optimization of L_raw or L_w.

Each source owns one t, shared by its two branches or four-spin quartet.
With optimize_shift=True the spin-resolved one-body shifts (mu_up, mu_down)
are inner variables too: mu starts at mu*(t0), and every sweep updates all
t's sequentially, then each mu exactly. The basis-overlap cache is reused
throughout a sequential (Gauss-Seidel) inner solve. No ERIs,
eigendecompositions or dense rotations are rebuilt. loss_kind='weighted'
uses sqrt(a_i)-scaled blocks, so every source profile keeps the exact
C + A exp(2t) + B exp(-2t) form.
"""

from dataclasses import replace
import math
from numbers import Integral, Real

import numpy as np

from .channels import fixed_partition_amplitudes, positive_scale
from .models import (
    Channel, 
    FixedDecomposition, 
    PartitionCoefficients, 
    GaugeFeasibilityError,
    PartitionOptimizationResult, 
    PartitionSweep,
)
from .loss import OverlapEvaluator, commutator_norm2, loss_kind_name
from .problem import _real_array
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


def _source_channels(source, amplitudes, tau2):
    """Canonical branch order used by build_two_body_channels."""
    left, right = amplitudes
    if source.kind == "two_branch":
        branches = ((source.spins, 1.0), (source.spins, -1.0))
        weight = 0.5/tau2
    elif source.kind == "spin_quartet" and source.modes[0] != source.modes[1]:
        branches = (((0,0), 1.0), ((0,1), -1.0), ((1,0), -1.0), ((1,1), 1.0))
        weight = 1.0/tau2
    else:
        raise ValueError("Unsupported source kind/modes for partition optimization")
    result = tuple(Channel(source.bix, spins, weight, source.modes,
                           (sign*left, sign*right)) for spins, sign in branches)
    if any(c.spins[0] == c.spins[1] and c.modes[0] == c.modes[1] for c in result):
        raise ValueError("Source legs must be distinct spin-orbital modes")
    return result


def _validate_partition_state(decomposition):
    """Check the step-2 source-sharing contract once, without ERI checks."""
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
    stop = 0
    for source, d, source_slice in zip(two.sources, amplitudes, two.source_slices):
        expected = _source_channels(source, d, two.tau2)
        if source_slice != (stop, stop+len(expected)):
            raise ValueError("source_slices must be contiguous and match the canonical branches")
        if tuple(two.channels[stop:stop+len(expected)]) != expected:
            raise ValueError("Channels disagree with the shared source amplitudes/branch order")
        stop += len(expected)
    if stop != len(two.channels):
        raise ValueError("Every two-body channel must belong to exactly one source")
    return t, amplitudes


def with_partitions(decomposition, t):
    """Return new step-2 channels at t, preserving gauge, h and all weights.

    Validates the source map and open amplitude domain but does not repeat
    the original input/ERI checks. The input snapshot is never modified.
    """
    _validate_partition_state(decomposition)
    two = decomposition.two_body
    ratios, amplitudes = fixed_partition_amplitudes(
        two.gauge.eigenvalues, two.sources, two.tau2, t,
        two.partition, two.amplitude_bound)
    channels = tuple(c for source, d in zip(two.sources, amplitudes)
                     for c in _source_channels(source, d, two.tau2))
    return replace(decomposition, two_body=replace(two, t=ratios,
                   amplitudes=amplitudes, channels=channels))


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


def _partition_intervals(eigenvalues, sources, tau2, partition, amplitude_bound, *,
                         amplitude_limit=None, max_abs_t=30.0):
    """Same bounds before a feasible starting decomposition has been built."""
    rho, cap = _partition_limits(amplitude_bound, amplitude_limit, max_abs_t)
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
            raise GaugeFeasibilityError(f"Source {i} has no feasible partition within the requested bounds; "
                             "reduce tau2, increase amplitude_limit or increase max_abs_t")
        bounds[i] = (lower, upper)
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


class _PartitionWorkspace:
    """Private mutable amplitudes/blocks; public input/evaluator stay fixed."""

    def __init__(self, decomposition, loss_kind="raw"):
        t, amplitudes = _validate_partition_state(decomposition)
        self.decomposition = decomposition
        self.two = decomposition.two_body
        self.one_body = decomposition.one_body
        self.shift = list(check_one_body_shift(self.one_body))
        self.shift_changed = False
        self.t, self.amplitudes = t.copy(), amplitudes.copy()
        self.evaluator = OverlapEvaluator(decomposition, loss_kind=loss_kind)
        self.loss_kind = self.evaluator.loss_kind
        self.root_scales = self.evaluator.root_scales
        self.blocks = list(self.evaluator.blocks)
        self.offset = len(decomposition.one_body.channels)
        self.spin_counts = []
        for start, stop in self.two.source_slices:
            counts = np.zeros((2,2), dtype=int)
            for channel in self.two.channels[start:stop]:
                for leg, spin in enumerate(channel.spins):
                    counts[leg,spin] += 1
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
        # Branch signs disappear in each isolated-leg squared norm. Counting
        # the spins retains the exact multiplicity of two-branch/quartet U's.
        # Every same-bix channel, including this source, commutes identically.
        # Every channel of one source shares one a_i, hence one loss scale.
        scale = float(self.root_scales[self.offset+self.two.source_slices[i][0]])
        for leg, mode in enumerate(source.modes):
            d = self.amplitudes[i,leg] if scale == 1.0 else scale*self.amplitudes[i,leg]
            values = []
            for blocks in self.blocks:
                for spin, multiplicity in enumerate(self.spin_counts[i][leg]):
                    block = blocks[spin]
                    if not multiplicity or d == 0 or block is None or block.basis == source.bix:
                        continue
                    C = self.evaluator._overlap(source.bix, block.basis)[np.ix_((mode,), block.modes)]
                    values.append(multiplicity*commutator_norm2((d,), block.kappa, C))
            terms.append(math.fsum(values))
        if not all(math.isfinite(x) for x in terms):
            raise FloatingPointError("Nonfinite coordinate coefficients")
        C = math.fsum((self.loss, -terms[0], -terms[1]))
        return PartitionCoefficients(i, float(self.t[i]), C, *terms)

    def update(self, i, t):
        source = self.two.sources[i]
        _, amplitudes = fixed_partition_amplitudes(
            self.two.gauge.eigenvalues, (source,), self.two.tau2, t,
            self.two.partition, self.two.amplitude_bound)
        self.t[i], self.amplitudes[i] = t, amplitudes[0]
        start, stop = self.two.source_slices[i]
        channels = _source_channels(source, amplitudes[0], self.two.tau2)
        for j, channel in zip(range(self.offset+start, self.offset+stop), channels):
            legs = [[], []]
            for spin, mode, d in zip(channel.spins, channel.modes, channel.d):
                legs[spin].append((mode,d))
            self.blocks[j] = tuple(self.evaluator._block(source.bix, spin_legs, self.root_scales[j])
                                   for spin_legs in legs)

    def shift_coefficients(self, lower):
        """Exact per-spin mu profiles with every t at its CURRENT value."""
        one_body = replace(self.one_body, shift=tuple(self.shift))
        return shift_profiles(self.blocks, self.evaluator._overlap, one_body,
                              self.root_scales, lower)

    def update_shift(self, shift):
        """Rebuild one-body blocks at a new (mu_up, mu_down); domain is checked."""
        one_body = self.one_body
        for j, channel in enumerate(one_body.channels):
            d = (float(shift[channel.spin])-one_body.eigenvalues[channel.mode])*one_body.tau1
            if not math.isfinite(d) or 1.0+d <= 0:
                raise GaugeFeasibilityError("Shift leaves the one-body real-exponential domain 1+d>0")
            blocks = [None, None]
            blocks[channel.spin] = self.evaluator._block(-1, ((channel.mode, float(d)),),
                                                         self.root_scales[j])
            self.blocks[j] = tuple(blocks)
        self.shift = [float(mu) for mu in shift]
        self.shift_changed = True

    def snapshot(self):
        result = with_partitions(self.decomposition, self.t)
        return with_shift(result, tuple(self.shift)) if self.shift_changed else result


def partition_coefficients(decomposition, source_index=None, *, loss_kind="raw"):
    """A_i,B_i,C_i for one source, or a tuple for all sources.

    Other t's are held at their CURRENT values. The sums include every
    external one-/two-body channel with its actual spin and multiplicity.
    For loss_kind='raw' no a_i weights enter; for 'weighted' every pair
    carries a_i*a_j. C is obtained from the full loss minus the two
    nonnegative current-coordinate terms, and may be negative.
    """
    workspace = _PartitionWorkspace(decomposition, loss_kind)
    if source_index is None:
        return tuple(workspace.coefficients(i) for i in range(len(workspace.t)))
    if isinstance(source_index, (bool, np.bool_)) or not isinstance(source_index, Integral):
        raise TypeError("source_index must be an integer")
    if not 0 <= source_index < len(workspace.t):
        raise IndexError("source_index is out of range")
    return workspace.coefficients(source_index)


def _tolerance(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return float(value)


def optimize_partitions(decomposition, *, max_sweeps=100,
                        gradient_atol=1e-10, gradient_rtol=1e-8,
                        amplitude_limit=None, max_abs_t=30.0, loss_kind="raw",
                        optimize_shift=False, one_body_limit=None,
                        shift_atol=0.0, shift_rtol=1e-8):
    """Sequential exact coordinate updates, at a fixed Cholesky gauge.

    loss_kind selects L_raw ('raw') or the unnormalized L_w ('weighted').
    Stop only when ||projected_gradient_t||_inf <= gradient_atol +
    gradient_rtol*initial_projected_gradient_norm, evaluated AFTER a full
    sweep with every other coordinate at its latest value. max_sweeps=0
    checks the input without updating it. Loss changes are diagnostics.

    optimize_shift=True also relaxes (mu_up, mu_down). mu is first set to its
    exact minimizer mu*(t0) (so no mu seed enters), and each sweep updates all
    t's and then each mu exactly. Since mu is then exactly optimal at the
    current t whenever the test is made, its gradient vanishes (or is
    wall-filtered at mu >= max(eps)-one_body_limit/tau1) and needs no
    threshold. The extra shift test requires the last sweep to have moved every
    mu by at most shift_atol + shift_rtol*(max(eps)-min(eps)). A nonzero shift
    requires particle numbers on the decomposition.

    A maximum-sweep exit returns converged=False, status='max_sweeps'; it
    is not silently treated as convergence. This is a reference coordinate
    solver, not an outer gauge optimizer.
    """
    if isinstance(max_sweeps, (bool, np.bool_)) or not isinstance(max_sweeps, Integral) or max_sweeps < 0:
        raise ValueError("max_sweeps must be a nonnegative integer")
    atol = _tolerance(gradient_atol, "gradient_atol")
    rtol = _tolerance(gradient_rtol, "gradient_rtol")
    loss_kind = loss_kind_name(loss_kind)
    if not isinstance(optimize_shift, (bool, np.bool_)):
        raise ValueError("optimize_shift must be boolean")
    optimize_shift = bool(optimize_shift)
    rho1 = one_body_limit_value(one_body_limit)
    lower = shift_bounds(decomposition.one_body, rho1)
    shift_tol = shift_tolerance(decomposition.one_body.eigenvalues, shift_atol, shift_rtol)
    if optimize_shift and decomposition.two_body.nelec is None:
        raise ValueError("optimize_shift requires particle numbers: pass nelec to prepare_problem")
    bounds = partition_bounds(decomposition, amplitude_limit=amplitude_limit, max_abs_t=max_abs_t)
    workspace = _PartitionWorkspace(decomposition, loss_kind)
    if np.any(workspace.t < bounds[:,0]) or np.any(workspace.t > bounds[:,1]):
        raise ValueError("Initial t lies outside the optimizer bounds; use feasible initial partitions "
                         "or increase amplitude_limit/max_abs_t")
    input_loss = workspace.loss

    def relax_shift():
        """Exact mu step at the current t; returns the largest |change|."""
        new = list(workspace.shift)
        for profile in workspace.shift_coefficients(lower):
            new[profile.spin] = profile.update
        change = max(abs(a-b) for a, b in zip(new, workspace.shift))
        workspace.update_shift(new)
        return change

    def shift_gradient():
        result = [0.0, 0.0]
        for profile in workspace.shift_coefficients(lower):
            result[profile.spin] = float(profile.projected_gradient)
        return tuple(result)

    if optimize_shift:
        # mu*(t0): the starting mu is determined by t0, never by a separate seed.
        relax_shift()
        workspace.loss = workspace.total_loss()

    def diagnostics():
        coefficients = tuple(workspace.coefficients(i) for i in range(len(workspace.t)))
        gradient = _readonly([c.gradient for c in coefficients])
        if not np.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite partition gradient")
        projected = projected_partition_gradient(workspace.t, gradient, bounds)
        norm = float(np.max(np.abs(projected), initial=0.0))
        return coefficients, gradient, projected, norm

    coefficients, gradient, projected, norm = diagnostics()
    initial_loss = workspace.loss
    threshold = atol+rtol*norm
    if not math.isfinite(threshold):
        raise ValueError("Gradient stopping threshold is not representable")
    history = [PartitionSweep(0, workspace.loss, norm, 0.0, 0.0, 0, 0.0)]
    converged = norm <= threshold
    for sweep in range(1, max_sweeps+1):
        if converged:
            break
        previous_loss = workspace.loss
        updates, max_step, shift_change = 0, 0.0, 0.0
        for i in range(len(workspace.t)):
            # Recompute after EVERY previous accepted coordinate update.
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
                continue  # Flat to working precision: retain the previous coordinate.
            workspace.update(i, optimal)
            workspace.loss = math.fsum((workspace.loss, change))
            max_step = max(max_step, abs(step))
            updates += 1
        if optimize_shift:
            # t's first, then mu: at the check below mu is exactly optimal at the latest t.
            shift_change = relax_shift()
        # Refresh the complete nonnegative sum once per sweep to prevent
        # accumulated subtraction error from affecting diagnostics/C_i.
        workspace.loss = workspace.total_loss()
        roundoff = 512*np.finfo(float).eps*max(previous_loss, workspace.loss, np.finfo(float).tiny)
        if workspace.loss > previous_loss+roundoff:
            raise FloatingPointError(f"Sequential sweep increased the full {loss_kind} loss")
        coefficients, gradient, projected, norm = diagnostics()
        history.append(PartitionSweep(sweep, workspace.loss, norm,
                                     previous_loss-workspace.loss, max_step, updates, shift_change))
        converged = norm <= threshold and (not optimize_shift or shift_change <= shift_tol)

    return PartitionOptimizationResult(
        workspace.snapshot(), initial_loss, workspace.loss, bounds, coefficients,
        gradient, projected, threshold, tuple(history), bool(converged),
        "converged" if converged else "max_sweeps",
        *_partition_limits(decomposition.two_body.amplitude_bound, amplitude_limit, max_abs_t),
        loss_kind=loss_kind, optimize_shift=optimize_shift, shift_bounds=lower,
        shift_gradient=shift_gradient() if optimize_shift else None, shift_tolerance=shift_tol, input_loss=input_loss,
        one_body_limit=one_body_limit, shift_atol=float(shift_atol), shift_rtol=float(shift_rtol))
