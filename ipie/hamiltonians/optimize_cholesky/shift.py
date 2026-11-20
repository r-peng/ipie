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
scale s_k (1 for raw, a_k=1/tau1 for weighted). sum_j ||[x_k x_k^T, A_j]||^2 is
the mode-space field dL/dq of loss.py at the h-basis mode k. The minimizer is
the w-weighted mean of the orbital energies; the only constraint is the
one-sided bound 1 + d >= 1 - one_body_limit, i.e.
mu_sigma >= max_k epsilon_k - one_body_limit/tau1.

Layer order: models -> problem -> channels -> loss -> shift -> partitions.
"""

from dataclasses import replace
from numbers import Real

import math
import numpy as np

from .channels import one_body_at_shift, shift_pair
from .loss import ModeSpace, channel_loads, check_bases, loss_kind_name
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
    """Verify stored one-body amplitudes against (mu_sigma - epsilon)*tau1 (math.isclose rule)."""
    shift = shift_pair(one_body.shift)
    table = one_body.table
    expected = (np.asarray(shift)[table.spin] - np.asarray(one_body.eigenvalues)[table.mode])*one_body.tau1
    diff = np.abs(table.d - expected)
    if np.any(diff > np.maximum(rtol*np.maximum(np.abs(table.d), np.abs(expected)), 1e-300)):
        raise ValueError("One-body amplitudes disagree with the stored shift; rebuild channels")
    return shift


def shift_profiles(one_body, field, scales, lower):
    """Exact per-spin profiles from the h-basis field.

    field: (2, n) dL/dq on the h-basis modes (loss.ModeSpace.field with
    cols=hslice) at the CURRENT loads; scales: per one-body channel pair scale
    s_k. Returns a ShiftCoefficients for each spin that carries one-body channels.
    """
    table = one_body.table
    shift = shift_pair(one_body.shift)
    eps = np.asarray(one_body.eigenvalues, dtype=np.float64)
    result = []
    for spin in one_body.spins:
        if lower[spin] is None:
            raise ValueError(f"No shift bound for spin {spin}")
        selected = table.spin == spin
        modes = table.mode[selected]
        w = np.asarray(scales)[selected]*field[spin, modes]
        result.append(ShiftCoefficients(spin, shift[spin], float(lower[spin]), math.fsum(w.tolist()),
                                        math.fsum((w*eps[modes]).tolist()), float(one_body.tau1)))
    return tuple(result)


def shift_coefficients(decomposition, *, loss_kind="weighted", one_body_limit=None):
    """Standalone exact mu profiles at a fixed decomposition (all t fixed)."""
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    loss_kind = loss_kind_name(loss_kind)
    check_one_body_shift(decomposition.one_body)
    check_bases(decomposition)
    space = ModeSpace.of(decomposition)
    q, O = channel_loads(decomposition, loss_kind, space)
    h, _ = space.offdiagonal_terms(O)
    table = decomposition.one_body.table
    scales = table.a if loss_kind == "weighted" else np.ones(len(table))
    return shift_profiles(decomposition.one_body, space.field(q, h, space.hslice), scales,
                          shift_bounds(decomposition.one_body, one_body_limit))


def optimal_shift(decomposition, *, loss_kind="weighted", one_body_limit=None):
    """(mu_up, mu_down) minimizing the loss at fixed t; absent spins keep their value."""
    shift = list(shift_pair(decomposition.one_body.shift))
    for profile in shift_coefficients(decomposition, loss_kind=loss_kind,
                                      one_body_limit=one_body_limit):
        shift[profile.spin] = profile.update
    return tuple(shift)
