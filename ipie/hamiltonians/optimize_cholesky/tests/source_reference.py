"""Per-source and per-factor loops (moved out of production in step 4).

These are the step-3 scalar implementations, copied verbatim apart from
their signatures: the amplitude map, the partition intervals, the amplitude
and bound pullbacks, and the single-factor eigh pullback. Production computes
the same quantities for all sources (or factors) at once; the tests compare
the batched kernels against these loops, including which source an error
names.
"""

import math

import numpy as np

from ipie.hamiltonians.optimize_cholesky.models import GaugeFeasibilityError, GaugeRegularityError
from ipie.hamiltonians.optimize_cholesky.partitions import _partition_limits


def _source_amplitude_pair(left, right, ratio, log_tau, log_bound, partition, bound, index, label):
    """(d_left, d_right) of one source, or None for a balanced zero coupling."""
    if partition == "balanced_log":
        if left == 0 or right == 0:
            return None  # fixed formal identity channels, positive analytic weights
        base = 0.5 * (np.log(abs(left)) + np.log(abs(right)) + log_tau)
        logs = (base + ratio, base - ratio)
    else:
        logs = ((-np.inf if left == 0 else np.log(abs(left)) + 0.5*log_tau + ratio),
                (-np.inf if right == 0 else np.log(abs(right)) + 0.5*log_tau - ratio))
    if max(logs) >= log_bound:
        raise ValueError(
            f"Source {index} {label}: partition leaves abs(d)<{bound}. "
            "Choose a smaller tau2 or a different fixed t; amplitudes were not clipped."
        )
    pair = (np.sign(left)*np.exp(logs[0]), -np.sign(right)*np.exp(logs[1]))
    if any(log != -np.inf and d == 0 for log, d in zip(logs, pair)):
        raise ValueError(f"Source {index}: amplitude underflow; choose a less extreme partition")
    if not np.isfinite(pair).all() or np.max(np.abs(pair)) >= bound:
        raise ValueError(f"Source {index}: amplitudes lie outside the requested open domain")
    return pair


def amplitudes(ell, sources, tau2, t, partition="balanced_log", amplitude_bound=1.0):
    """The step-3 loop of channels.fixed_partition_amplitudes (t: one entry per source)."""
    bix, modes = sources.bix, sources.modes
    result = np.zeros((len(sources), 2), dtype=np.float64)
    log_tau = np.log(tau2)
    log_bound = np.log(amplitude_bound)
    for i, (g, p, q) in enumerate(zip(bix.tolist(), modes[:, 0].tolist(), modes[:, 1].tolist())):
        pair = _source_amplitude_pair(ell[g, p], ell[g, q], t[i], log_tau, log_bound,
                                      partition, amplitude_bound, i, f"(bix={g}, modes={(p,q)})")
        if pair is not None:
            result[i] = pair
    return result


def partition_intervals(eigenvalues, sources, tau2, partition, amplitude_bound, *,
                        amplitude_limit=None, max_abs_t=30.0):
    """The step-3 partitions._partition_intervals."""
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
    return bounds


def amplitude_pullback(two, g_amplitude, options):
    """The step-3 gradients._amplitude_pullback."""
    gell = np.zeros_like(two.gauge.eigenvalues)
    sources = two.sources
    gt = np.zeros(len(sources))
    sign = sources.channel_sign
    for i,(g,p,q,start,stop) in enumerate(zip(sources.bix.tolist(),sources.modes[:,0].tolist(),
                                               sources.modes[:,1].tolist(),sources.slices[:,0].tolist(),
                                               sources.slices[:,1].tolist())):
        signs = sign[start:stop]
        gleft,gright = np.sum(signs[:,None]*g_amplitude[start:stop],axis=0)
        left,right = two.amplitudes[i]
        gt[i] = gleft*left-gright*right
        ell = two.gauge.eigenvalues[g]
        if two.partition == "balanced_log":
            tolerance = options.zero_atol+options.zero_rtol*np.max(np.abs(ell),initial=0.)
            if abs(ell[p]) <= tolerance or abs(ell[q]) <= tolerance:
                raise GaugeRegularityError(
                    f"Source {i}, factor {g}: balanced amplitudes meet a zero/near-zero "
                    "eigenvalue; a smooth sign branch is required")
            common = .5*(gleft*left+gright*right)
            gell[g,p] += common/ell[p]
            gell[g,q] += common/ell[q]
        else:
            # Direct derivatives also cover a simple zero eigenvalue.
            gell[g,p] += gleft*math.exp(.5*math.log(two.tau2)+two.t[i])
            gell[g,q] -= gright*math.exp(.5*math.log(two.tau2)-two.t[i])
    return gell,gt


def bound_pullback(two, gt, z, amplitude_limit, max_abs_t, options):
    """The step-3 gradients._bound_pullback."""
    rho,cap = _partition_limits(two.amplitude_bound,amplitude_limit,max_abs_t)
    log_rho = math.log(rho)
    result = np.zeros_like(two.gauge.eigenvalues)
    sources = two.sources
    for i,(g,p,q) in enumerate(zip(sources.bix.tolist(),sources.modes[:,0].tolist(),
                                   sources.modes[:,1].tolist())):
        left,right = two.gauge.eigenvalues[g,p],two.gauge.eigenvalues[g,q]
        if two.partition == "balanced_log":
            base = .5*(math.log(abs(left))+math.log(abs(right))+math.log(two.tau2))
            log_left = log_right = base
        else:
            log_left = -np.inf if left == 0 else math.log(abs(left))+.5*math.log(two.tau2)
            log_right = -np.inf if right == 0 else math.log(abs(right))+.5*math.log(two.tau2)
        lower_amp,upper_amp = log_right-log_rho,log_rho-log_left
        lower_weight,upper_weight = gt[i]*(1-z[i])/2,gt[i]*(1+z[i])/2
        if lower_weight != 0:
            if abs(lower_amp+cap) <= options.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: lower amplitude bound ties the finite t cap")
            if lower_amp > -cap:
                if two.partition == "balanced_log":
                    result[g,p] += .5*lower_weight/left
                    result[g,q] += .5*lower_weight/right
                else:
                    result[g,q] += lower_weight/right
        if upper_weight != 0:
            if abs(upper_amp-cap) <= options.bound_switch_atol:
                raise GaugeRegularityError(f"Source {i}: upper amplitude bound ties the finite t cap")
            if upper_amp < cap:
                if two.partition == "balanced_log":
                    result[g,p] -= .5*upper_weight/left
                    result[g,q] -= .5*upper_weight/right
                else:
                    result[g,p] -= upper_weight/left
    return result


def eigh_pullback(eigenvalues, eigenvectors, grad_values, grad_vectors):
    """The step-3 single-factor eigh_vjp formula (no input checks)."""
    ell, X, ge, gX = eigenvalues, eigenvectors, grad_values, grad_vectors
    gap = ell[None,:]-ell[:,None]
    mask = ~np.eye(ell.size,dtype=bool)
    inverse = np.zeros_like(gap)
    np.divide(1.,gap,out=inverse,where=mask)
    term = (X.T@gX)*inverse
    result = X@(np.diag(ge)+(term+term.T)/2)@X.T
    return (result+result.T)/2
