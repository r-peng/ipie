"""Spin-resolved one-body shifts mu_sigma: bounds, exact quadratic profiles, updates.

The one-body term is written as
    sum_{k,sigma} epsilon_k n_{k,sigma}
        = sum_{k,sigma} (epsilon_k - mu_sigma) n_{k,sigma} + sum_sigma mu_sigma N_sigma,
so channel amplitudes become d[k,sigma] = (mu_sigma - epsilon_k)*tau1 and the
constant sum_sigma mu_sigma N_sigma moves into Lambda (it is a constant on a
fixed (N_up, N_down) sector). One-body channels share one basis and commute,
so each loss_kind objective is EXACTLY quadratic in every mu_sigma, and the two
spins decouple:

    L(mu_sigma) = const + tau1**2 * sum_k w_k (mu_sigma - epsilon_k)**2,
    w_k = s_k * sum_j ||[x_k x_k^T, A_j^sigma]||_F^2,

with unit one-body legs, loss-scaled two-body blocks A_j and the one-body pair
scale s_k (1 for raw, a_k=1/tau1 for weighted). The minimizer is the w-weighted
mean of the orbital energies; the only constraint is the one-sided bound
1 + d >= 1 - one_body_limit, i.e. mu_sigma >= max_k epsilon_k - one_body_limit/tau1.

Layer order: models -> problem -> channels -> loss -> shift -> partitions.
"""

from dataclasses import replace
from numbers import Real

import math
import numpy as np

from .channels import one_body_at_shift, shift_pair
from .loss import OverlapEvaluator, commutator_norm2
from .models import FixedDecomposition, ShiftCoefficients


DEFAULT_ONE_BODY_LIMIT = 1.0 - 1e-8


def one_body_limit_value(one_body_limit):
    """Closed bound rho_1 in (0,1) on negative one-body amplitudes, d >= -rho_1."""
    if one_body_limit is None:
        return DEFAULT_ONE_BODY_LIMIT
    if (isinstance(one_body_limit, (bool, np.bool_)) or not isinstance(one_body_limit, Real)
            or not np.isfinite(one_body_limit) or not 0 < one_body_limit < 1):
        raise ValueError("one_body_limit must lie strictly between 0 and 1")
    return float(one_body_limit)


def shift_bounds(one_body, one_body_limit=None):
    """(lower_up, lower_down); None for a spin without one-body channels."""
    rho = one_body_limit_value(one_body_limit)
    lower = float(np.max(one_body.eigenvalues) - rho/one_body.tau1)
    spins = one_body.spins
    return tuple(lower if s in spins else None for s in (0, 1))


def shift_scale(eigenvalues):
    """Band scale for the shift tolerance: max(eps)-min(eps), or max(1,|eps|) if flat."""
    eps = np.asarray(eigenvalues, dtype=np.float64)
    width = float(np.ptp(eps)) if eps.size else 0.0
    return width if width > 0 else max(1.0, float(np.max(np.abs(eps), initial=0.0)))


def shift_tolerance(eigenvalues, shift_atol=0.0, shift_rtol=1e-8):
    """Largest accepted change of any mu_sigma during the final sweep."""
    for value, name in ((shift_atol, "shift_atol"), (shift_rtol, "shift_rtol")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return float(shift_atol + shift_rtol*shift_scale(eigenvalues))


def with_shift(decomposition, shift):
    """Return a decomposition whose one-body channels use shift=(mu_up, mu_down).

    The h eigensystem, weights and two-body channels are reused unchanged.
    Requires particle numbers for a nonzero shift; raises ValueError when a
    one-body amplitude leaves the real-exponential domain 1+d>0.
    """
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    one_body = one_body_at_shift(decomposition.one_body, shift_pair(shift),
                                 decomposition.two_body.nelec)
    return replace(decomposition, one_body=one_body)


def check_one_body_shift(one_body, *, rtol=1e-13):
    """Verify stored one-body amplitudes against (mu_sigma - epsilon)*tau1."""
    shift = shift_pair(one_body.shift)
    for channel in one_body.channels:
        expected = (shift[channel.spin] - one_body.eigenvalues[channel.mode])*one_body.tau1
        if not math.isclose(channel.d, expected, rel_tol=rtol, abs_tol=1e-300):
            raise ValueError("One-body amplitudes disagree with the stored shift; rebuild channels")
    return shift


def shift_profiles(blocks, overlap, one_body, root_scales, lower):
    """Exact per-spin profiles from the CURRENT block list.

    blocks: per-channel (up, down) spin blocks, one-body first, with the
    two-body blocks already scaled for the loss; overlap(a,b) returns the basis
    overlap; root_scales: per-channel sqrt(s_i). Returns a tuple of
    ShiftCoefficients for each spin that carries one-body channels.
    """
    n_one = len(one_body.channels)
    shift = shift_pair(one_body.shift)
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
    result = []
    for spin in one_body.spins:
        if lower[spin] is None:
            raise ValueError(f"No shift bound for spin {spin}")
        result.append(ShiftCoefficients(spin, shift[spin], float(lower[spin]),
                                        math.fsum(weight[spin]), math.fsum(moment[spin]),
                                        float(one_body.tau1)))
    return tuple(result)


def shift_coefficients(decomposition, *, loss_kind="raw", one_body_limit=None):
    """Standalone exact mu profiles at a fixed decomposition (all t fixed)."""
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    check_one_body_shift(decomposition.one_body)
    evaluator = OverlapEvaluator(decomposition, loss_kind=loss_kind)
    return shift_profiles(evaluator.blocks, evaluator._overlap, decomposition.one_body,
                          evaluator.root_scales,
                          shift_bounds(decomposition.one_body, one_body_limit))


def optimal_shift(decomposition, *, loss_kind="raw", one_body_limit=None):
    """(mu_up, mu_down) minimizing the loss at fixed t; absent spins keep their value."""
    shift = list(shift_pair(decomposition.one_body.shift))
    for profile in shift_coefficients(decomposition, loss_kind=loss_kind,
                                      one_body_limit=one_body_limit):
        shift[profile.spin] = profile.update
    return tuple(shift)
