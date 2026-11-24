"""Save an optimization result as a low-memory, QCSOR-ready HDF5 file.

save_decomposition(path, source) accepts an OuterOptimizationResult,
ExponentialEvaluation, GaugeGradient, PartitionOptimizationResult or a plain
FixedDecomposition. It writes the terms already laid out the way QCSOR stores
them, so ipie.hamiltonians.sor_base.QCSOR.from_file reads plain arrays and
never builds per-channel Python objects. Format version 3:

  attrs         format/version, policy, decomp_type, nelec, mu=(mu_up,mu_down),
                tau1, tau2, partition, amplitude_bound, denom and
                lambda_electronic of the retained terms, denom_all of all
                channels, drop_identity, dropped_identity, term counts,
                has_cholesky_shift, cholesky_shift (r,2; zeros without one)
                and cholesky_energy E_c (0 without one)
  h1, v0        the PHYSICAL energy integrals: h' = h - 1/2 sum_g L_g^2 and
                v0 = 1/2 sum_g L_g^2 for both spins (h1 + v0 = bare h), with the
                unshifted rotated factors L_g; local energies need no constant
  chol_X        rotated factor eigenvectors (r,n,n), columns ascending
  chol_ell      unshifted rotated factor eigenvalues (r,n); QCSOR energy
                integrals W = ell ell^T
  chol_lam      only with a Cholesky shift: the channels' shifted eigenvalues
                lambda[s,g,a] = ell[g,a] - mu[g,s], (2,r,n)
  h1_X, h1_eps  the one-body channel bases (k,n,n) and eps - mu_s (k,n): the
                eigensystem of h_s (bare h without a Cholesky shift); k=1 when
                the spins share the basis and mu_up==mu_down (or a single
                spin), else 2 (up, down)
  a             all Green weights (T,) in the final global term order
  p1, d1        one-body modes/amplitudes (T1,), global terms [0,T1)
  p2, d2        two-body modes/amplitudes (T2,2), global terms [T1,T)
  keys          (K,5) rows [bix, spin0, spin1, start, stop]; spin1=-1 marks
                one-body keys; bix<r are factors, bix>=r are h1 bases
  provenance    JSON record (statuses, losses, solver settings), user metadata,
                the gauge O, the source log-ratios t, and theta/reference/
                initial_theta when available (not read by QCSOR)

On the (N_up, N_down) sector sum_i a_i U_i = Lambda I - H with Lambda =
lambda_electronic = denom + mu.N + E_c. With a Cholesky shift the channels
are exact on the stored sector only. Version 2 files (no Cholesky shift,
h1_X of shape (n,n)) still load.

The term order is: one-body keys first, keys in first-occurrence order of the
optimizer's channel list, channel order kept within a key. (down,up) channels
are stored under SOR's (0,1) key with their legs swapped, so leg 0 is always up.

h5py is imported lazily. Pickle is deliberately not used.
"""

from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
import os

import numpy as np

from .models import (
    ExponentialEvaluation,
    FixedDecomposition,
    GaugeGradient,
    OuterOptimizationResult,
    PartitionOptimizationResult,
)


FORMAT = "lafqmc-optimized-decomposition"
FORMAT_VERSION = 3
READABLE_VERSIONS = (2, 3)

POLICY_TO_DECOMP_TYPE = {"hybrid": "all", "ab_only": "ab_only", "aa_only": "aa_only"}
_SPIN_CLASSES = ((0, 0), (1, 1), (0, 1))
_CORE = ("h1", "v0", "chol_X", "chol_ell", "h1_X", "h1_eps", "a", "p1", "d1", "p2", "d2", "keys")


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Decomposition files require h5py; install it with 'pip install h5py'") from exc
    return h5py


def _require(condition, message):
    if not condition:
        raise ValueError(message)


# --------------------------------------------------------------------------- provenance

def _jsonable(value):
    """Convert numpy scalars/arrays and tuples into JSON-compatible values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


_PARTITION_FIELDS = ("loss_kind", "loss", "initial_loss", "input_loss", "status", "converged",
                     "gradient_threshold", "amplitude_limit", "max_abs_t", "optimize_shift",
                     "shift_bounds", "shift_gradient", "shift_tolerance", "one_body_limit",
                     "shift_atol", "shift_rtol")


def _partition_record(result):
    record = {name: getattr(result, name, None) for name in _PARTITION_FIELDS}
    record["sweeps"] = len(result.history) - 1
    return record


def _unpack(source):
    """Return (FixedDecomposition, provenance record, provenance arrays)."""
    arrays = {}
    if isinstance(source, FixedDecomposition):
        return source, {"source_type": "FixedDecomposition"}, arrays
    if isinstance(source, PartitionOptimizationResult):
        record = {"source_type": "PartitionOptimizationResult", "inner": _partition_record(source)}
        return source.decomposition, record, arrays
    if isinstance(source, GaugeGradient):
        record = {"source_type": "GaugeGradient", "mode": source.mode, "loss": source.loss}
        if isinstance(source.inner_result, PartitionOptimizationResult):
            record["inner"] = _partition_record(source.inner_result)
        return source.decomposition, record, arrays
    if isinstance(source, ExponentialEvaluation):
        decomposition, record, arrays = _unpack(source.gauge_gradient)
        record["source_type"] = "ExponentialEvaluation"
        arrays["theta"] = np.asarray(source.theta, dtype=np.float64)
        arrays["gradient_theta"] = np.asarray(source.gradient_theta, dtype=np.float64)
        return decomposition, record, arrays
    if isinstance(source, OuterOptimizationResult):
        decomposition, record, arrays = _unpack(source.evaluation)
        record["source_type"] = "OuterOptimizationResult"
        record["outer"] = {
            "status": source.status, "message": source.message, "converged": source.converged,
            "loss": source.loss, "initial_loss": source.initial_loss,
            "iterations": source.iterations, "evaluations": source.evaluations,
            "failed_evaluations": source.failed_evaluations, "inner_sweeps": source.inner_sweeps,
            "gradient_norm": source.gradient_norm,
            "tangent_gradient_norm": source.tangent_gradient_norm,
            "gradient_threshold": source.gradient_threshold,
            "tangent_gradient_threshold": source.tangent_gradient_threshold,
            "history_loss": [row.loss for row in source.history],
            "history_acceptance": [getattr(row, "acceptance", "") for row in source.history],
        }
        arrays["reference"] = np.asarray(source.reference, dtype=np.float64)
        arrays["initial_theta"] = np.asarray(source.initial_theta, dtype=np.float64)
        return decomposition, record, arrays
    if hasattr(source, "decomposition") and isinstance(source.decomposition, FixedDecomposition):
        return source.decomposition, {"source_type": type(source).__name__}, arrays
    raise TypeError("source must be a FixedDecomposition or an optimization result exposing one")


# --------------------------------------------------------------------------- channel tables

def _pairs(values, dtype, count):
    return np.array(values, dtype=dtype).reshape(count, 2)


def _arrays_from_decomposition(decomposition):
    """Plain arrays of a canonical FixedDecomposition, read straight from its tables.

    The one-body eigensystem is per spin, (2, n, n) and (2, n), equal rows when
    the spins share it. h is the bare one-body matrix: the stored h_s minus its
    Cholesky-shift term (channels.one_body_matrices), or h itself without a shift.
    """
    one, two = decomposition.one_body, decomposition.two_body
    if not two.canonical:
        raise ValueError("Only canonical decompositions (from the builders or the optimizer) can be "
                         "saved; with_channels results are not")
    gauge = two.gauge
    ot, tt, src = one.table, two.table, two.sources
    nsrc = len(src)
    chol = np.asarray(gauge.chol, dtype=np.float64)
    mu = two.cholesky_shift
    spins = one.spins
    first = spins[0] if spins else 0
    X1 = np.stack([np.asarray(one.spin_X(s), dtype=np.float64) for s in (0, 1)])
    eps = np.asarray(one.mode_eigenvalues, dtype=np.float64)
    h = (X1[first]*eps[first]) @ X1[first].T
    if mu is not None:
        mu = np.asarray(mu, dtype=np.float64)
        coefficient = mu @ np.asarray(two.nelec, dtype=np.float64) - mu[:, first]
        h = h - np.einsum("g,gpq->pq", coefficient, chol)
    return {
        "policy": two.policy, "nelec": two.nelec, "denom": float(decomposition.denom),
        "h": h, "h1_X": X1, "eps": eps, "spins": tuple(spins),
        "tau1": float(one.tau1), "shift": tuple(float(x) for x in one.shift),
        "s1": np.array(ot.spin, dtype=np.int64), "a1": np.array(ot.a, dtype=np.float64),
        "m1": np.array(ot.mode, dtype=np.int64), "d1": np.array(ot.d, dtype=np.float64),
        "O": np.asarray(gauge.O, dtype=np.float64), "chol": chol,
        "ell": np.asarray(gauge.eigenvalues, dtype=np.float64), "X": np.asarray(gauge.X, dtype=np.float64),
        "cholesky_shift": mu, "cholesky_energy": float(decomposition.cholesky_energy),
        "lam": np.asarray(two.spin_eigenvalues, dtype=np.float64),
        "tau2": float(two.tau2), "partition": str(two.partition),
        "amplitude_bound": float(two.amplitude_bound), "t": np.asarray(two.t, dtype=np.float64).reshape(-1),
        "src_bix": np.array(src.bix, dtype=np.int64), "src_modes": _pairs(src.modes, np.int64, nsrc),
        "legs": np.asarray(two.leg_amplitudes, dtype=np.float64).reshape(nsrc, 2, 2),
        "slices": _pairs(src.slices, np.int64, nsrc),
        "bix2": np.array(tt.bix, dtype=np.int64), "spins2": np.array(tt.spins, dtype=np.int64),
        "a2": np.array(tt.a, dtype=np.float64), "modes2": np.array(tt.modes, dtype=np.int64),
        "d2": np.array(tt.d, dtype=np.float64),
    }


def _check_arrays(data, rtol=1e-12):
    """Vectorized consistency checks of a channel table; returns (n, r)."""
    X1, eps, chol, ell, X = data["h1_X"], data["eps"], data["chol"], data["ell"], data["X"]
    n, r = X1.shape[1], chol.shape[0]
    _require(data["policy"] in POLICY_TO_DECOMP_TYPE, f"Unknown policy {data['policy']!r}")
    _require(X1.shape == (2, n, n) and eps.shape == (2, n), "Invalid one-body eigensystem shape")
    _require(chol.shape == (r, n, n) and ell.shape == (r, n) and X.shape == (r, n, n)
             and data["O"].shape == (r, r), "Invalid Cholesky gauge shapes")
    _require(all(np.allclose(X1[s].T @ X1[s], np.eye(n), rtol=0, atol=1e-10) for s in (0, 1)),
             "One-body eigenvectors are not orthonormal")
    scale = max(1., float(np.max(np.abs(chol), initial=0.)))
    _require(np.allclose(np.einsum("gpa,ga,gqa->gpq", X, ell, X), chol, rtol=0, atol=1e-10*scale),
             "Factor eigenpairs do not reconstruct the rotated factors")
    _require(np.allclose(data["O"] @ data["O"].T, np.eye(r), rtol=0, atol=1e-10), "Gauge is not orthogonal")
    h, mu, lam = data["h"], data["cholesky_shift"], data["lam"]
    _require(h.shape == (n, n) and np.allclose(h, h.T, rtol=0, atol=1e-12*max(1., float(np.max(np.abs(h))))),
             "The bare one-body matrix is not symmetric")
    if mu is None:
        _require(np.array_equal(lam, np.stack([ell, ell])) and data["cholesky_energy"] == 0.,
                 "Unshifted decomposition with shifted eigenvalues or a Cholesky energy")
    else:
        _require(mu.shape == (r, 2) and np.all(np.isfinite(mu)), "Invalid Cholesky shift")
        _require(data["nelec"] is not None, "A Cholesky shift requires particle numbers")
        _require(np.allclose(lam, ell[None] - mu.T[:, :, None], rtol=0, atol=1e-14*scale),
                 "Shifted eigenvalues disagree with the Cholesky shift")
        counts = np.asarray(data["nelec"], dtype=np.float64)
        m = mu @ counts
        _require(math.isclose(data["cholesky_energy"], .5*float(np.sum((mu*mu) @ counts - m*m)),
                              rel_tol=1e-12, abs_tol=1e-14),
                 "Cholesky energy disagrees with the shift")
    hscale = max(1., float(np.max(np.abs(eps), initial=0.)))
    for s in data["spins"]:
        h_s = h if mu is None else h + np.einsum("g,gpq->pq", mu @ np.asarray(data["nelec"], float) - mu[:, s], chol)
        _require(np.allclose((X1[s]*eps[s]) @ X1[s].T, h_s, rtol=0, atol=1e-10*hscale),
                 f"One-body eigenpairs do not reconstruct h_sigma (spin {s})")

    s1, a1, m1, d1 = data["s1"], data["a1"], data["m1"], data["d1"]
    _require(s1.size == a1.size == m1.size == d1.size, "Inconsistent one-body channel arrays")
    _require(np.all((s1 == 0) | (s1 == 1)) and np.all((m1 >= 0) & (m1 < n)), "Invalid one-body spins or modes")
    _require(np.all(np.isfinite(a1) & (a1 > 0)), "One-body weights must be finite and positive")
    shift = data["shift"]
    expected = (np.asarray(shift)[s1] - eps[s1, m1])*data["tau1"]
    _require(np.all(np.isclose(d1, expected, rtol=1e-13, atol=1e-300)),
             "One-body amplitudes disagree with the shift")
    _require(np.all(1. + d1 > 0), "One-body amplitudes leave the real-exponential domain 1+d>0")
    _require(data["nelec"] is not None or not any(shift), "A nonzero shift requires particle numbers")

    bix2, spins2, a2, modes2, d2 = data["bix2"], data["spins2"], data["a2"], data["modes2"], data["d2"]
    nch = bix2.size
    _require(spins2.shape == modes2.shape == d2.shape == (nch, 2) and a2.size == nch,
             "Inconsistent two-body channel arrays")
    _require(np.all((bix2 >= 0) & (bix2 < r)), "Two-body basis index out of range")
    _require(np.all((spins2 == 0) | (spins2 == 1)) and np.all((modes2 >= 0) & (modes2 < n)),
             "Invalid two-body spins or modes")
    _require(np.all(np.isfinite(a2) & (a2 > 0)), "Two-body weights must be finite and positive")
    _require(np.all(np.abs(d2) < 1.), "Two-body amplitudes must satisfy |d|<1")
    # Every channel: d_L d_R = -lambda_L lambda_R tau2 with its own leg spins.
    W = lam[spins2[:, 0], bix2, modes2[:, 0]]*lam[spins2[:, 1], bix2, modes2[:, 1]]
    _require(np.all(np.isclose(d2[:, 0]*d2[:, 1], -W*data["tau2"], rtol=rtol, atol=1e-300)),
             "A channel violates d_left*d_right = -lambda_left*lambda_right*tau2")

    src_bix, src_modes, legs, slices = data["src_bix"], data["src_modes"], data["legs"], data["slices"]
    nsrc = src_bix.size
    _require(src_modes.shape == slices.shape == (nsrc, 2) and legs.shape == (nsrc, 2, 2)
             and data["t"].shape == (nsrc,), "Inconsistent source arrays")
    if nsrc == 0:
        _require(nch == 0, "Channels without sources")
    else:
        _require(slices[0, 0] == 0 and slices[-1, 1] == nch and np.all(slices[1:, 0] == slices[:-1, 1])
                 and np.all(slices[:, 1] >= slices[:, 0]), "Source channel slices must be contiguous and ordered")
        owner = np.repeat(np.arange(nsrc), slices[:, 1] - slices[:, 0])
        _require(np.all(bix2 == src_bix[owner]) and np.all(modes2 == src_modes[owner]),
                 "Channel basis index or modes disagree with their source")
        reference = np.abs(np.stack([legs[owner, 0, spins2[:, 0]], legs[owner, 1, spins2[:, 1]]], axis=1))
        _require(np.all(np.abs(np.abs(d2) - reference) <= rtol*reference),
                 "Channel amplitudes disagree with their source")
    total = math.fsum(a1) + math.fsum(a2)
    _require(math.isclose(total, data["denom"], rel_tol=rtol), "denom does not match the channel weights")
    return n, r


def _group_by_first_occurrence(key_ids):
    """Stable grouping; groups ordered by first occurrence. Returns (perm, keys, counts).

    This is exactly the order SizeGroup.add_term (one channel at a time) followed
    by parse_decomposition produces.
    """
    key_ids = np.asarray(key_ids, dtype=np.int64)
    if key_ids.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    unique, first = np.unique(key_ids, return_index=True)
    order = np.argsort(first, kind="stable")
    rank_of_unique = np.empty(unique.size, dtype=np.int64)
    rank_of_unique[order] = np.arange(unique.size, dtype=np.int64)
    rank = rank_of_unique[np.searchsorted(unique, key_ids)]
    return np.argsort(rank, kind="stable"), unique[order], np.bincount(rank, minlength=unique.size)


def _sor_layout(data, drop_identity):
    """Turn checked channel arrays into the QCSOR layout (see the module docstring)."""
    n, r = data["h1_X"].shape[1], data["chol"].shape[0]
    shift, nelec = data["shift"], data["nelec"]
    X1, eps = data["h1_X"], data["eps"]

    spins = [int(s) for s in np.unique(data["s1"])]
    shared = len(spins) == 1 or (np.array_equal(X1[0], X1[1]) and np.array_equal(eps[0], eps[1])
                                 and shift[0] == shift[1])
    if not spins:
        h1_X, h1_eps, h1_bix = np.zeros((0, n, n)), np.zeros((0, n)), {}
    elif shared:
        h1_X, h1_eps = X1[spins[0]][None], (eps[spins[0]] - shift[spins[0]])[None, :]
        h1_bix = {s: r for s in spins}
    else:
        h1_X, h1_eps = np.stack([X1[s] for s in spins]), np.stack([eps[s] - shift[s] for s in spins])
        h1_bix = {s: r + i for i, s in enumerate(spins)}

    s1, d1 = data["s1"], data["d1"]
    idx1 = np.flatnonzero(d1 != 0.) if drop_identity else np.arange(s1.size)
    perm1, keys1, counts1 = _group_by_first_occurrence(s1[idx1])
    idx1 = idx1[perm1]

    spins2 = data["spins2"]
    modes2, d2 = data["modes2"].copy(), data["d2"].copy()
    swap = (spins2[:, 0] == 1) & (spins2[:, 1] == 0)
    modes2[swap] = modes2[swap][:, ::-1]
    d2[swap] = d2[swap][:, ::-1]
    spin_class = np.where(spins2[:, 0] == spins2[:, 1], spins2[:, 0], 2)
    idx2 = (np.flatnonzero(~((d2[:, 0] == 0.) & (d2[:, 1] == 0.))) if drop_identity
            else np.arange(spin_class.size))
    perm2, keys2, counts2 = _group_by_first_occurrence(data["bix2"][idx2]*3 + spin_class[idx2])
    idx2 = idx2[perm2]

    T1, T2 = idx1.size, idx2.size
    rows, start = [], 0
    for key, count in zip(keys1, counts1):
        rows.append((h1_bix[int(key)], int(key), -1, start, start + int(count)))
        start += int(count)
    for key, count in zip(keys2, counts2):
        sp = _SPIN_CLASSES[int(key) % 3]
        rows.append((int(key)//3, sp[0], sp[1], start, start + int(count)))
        start += int(count)

    a = np.concatenate([data["a1"][idx1], data["a2"][idx2]])
    # Same reductions as parse_decomposition, for its load-time consistency check.
    denom = float(a[:T1].sum()) + (float(a[T1:].sum()) if T2 > 0 else 0.)
    L2 = np.einsum("gpr,grq->pq", data["chol"], data["chol"])
    h = data["h"]
    lam = {} if data["cholesky_shift"] is None else {"chol_lam": data["lam"]}
    return {
        "decomp_type": POLICY_TO_DECOMP_TYPE[data["policy"]],
        "h1": h - .5*L2, "v0": np.tile(.5*L2[None, :, :], (2, 1, 1)),
        "chol_X": data["X"], "chol_ell": data["ell"], **lam,
        "h1_X": h1_X.reshape(-1, n, n), "h1_eps": h1_eps.reshape(-1, n),
        "a": a, "p1": data["m1"][idx1].astype(np.int64), "d1": d1[idx1],
        "p2": modes2[idx2].astype(np.int64).reshape(-1, 2), "d2": d2[idx2].reshape(-1, 2),
        "keys": np.array(rows, dtype=np.int64).reshape(-1, 5),
        "denom": denom,
        "lambda_electronic": denom + (0. if nelec is None else shift[0]*nelec[0] + shift[1]*nelec[1])
                             + data["cholesky_energy"],
        "dropped": int(s1.size + spin_class.size - T1 - T2),
    }


def save_decomposition(path, source, *, drop_identity=False, metadata=None, overwrite=False):
    """Write source (an optimization result or a FixedDecomposition) as a low-memory QCSOR file.

    Read it back with ipie.hamiltonians.sor_base.QCSOR.from_file(path). A
    decomposition with a Cholesky shift is stored with its shifted channels,
    per-spin one-body bases, the shift and E_c; the energy integrals (h1, v0,
    chol_ell) stay the physical, unshifted ones.
    drop_identity=True omits channels whose amplitudes are all exactly zero
    (exact; Lambda shrinks by their weight). metadata: a JSON-compatible user
    dictionary stored verbatim. An existing file raises FileExistsError unless
    overwrite=True. Returns the path.
    """
    decomposition, record, provenance_arrays = _unpack(source)
    path = os.fspath(path)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    data = _arrays_from_decomposition(decomposition)
    _check_arrays(data)
    layout = _sor_layout(data, drop_identity)
    record = json.dumps(_jsonable(record), allow_nan=False)
    user_metadata = json.dumps(_jsonable(metadata or {}), allow_nan=False)

    h5py = _h5py()
    nelec, shift = data["nelec"], data["shift"]
    with h5py.File(path, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["format_version"] = FORMAT_VERSION
        f.attrs["created"] = datetime.now(timezone.utc).isoformat()
        f.attrs["policy"] = data["policy"]
        f.attrs["decomp_type"] = layout["decomp_type"]
        f.attrs["has_nelec"] = nelec is not None
        f.attrs["nelec"] = np.array(nelec if nelec is not None else (-1, -1), dtype=np.int64)
        f.attrs["nbasis"] = int(data["h1_X"].shape[1])
        f.attrs["nchol"] = int(data["chol"].shape[0])
        f.attrs["mu"] = np.array(shift, dtype=np.float64)
        f.attrs["tau1"] = data["tau1"]
        f.attrs["tau2"] = data["tau2"]
        f.attrs["partition"] = data["partition"]
        f.attrs["amplitude_bound"] = data["amplitude_bound"]
        f.attrs["denom"] = layout["denom"]
        f.attrs["denom_all"] = data["denom"]
        f.attrs["lambda_electronic"] = layout["lambda_electronic"]
        f.attrs["drop_identity"] = bool(drop_identity)
        f.attrs["dropped_identity"] = layout["dropped"]
        f.attrs["n_one_body"] = int(layout["p1"].shape[0])
        f.attrs["n_two_body"] = int(layout["p2"].shape[0])
        mu = data["cholesky_shift"]
        f.attrs["has_cholesky_shift"] = mu is not None
        f.attrs["cholesky_shift"] = np.zeros((data["chol"].shape[0], 2)) if mu is None else mu
        f.attrs["cholesky_energy"] = data["cholesky_energy"]
        for name in _CORE + (("chol_lam",) if mu is not None else ()):
            f.create_dataset(name, data=layout[name])
        g = f.create_group("provenance")
        g.attrs["record"] = record
        g.attrs["user_metadata"] = user_metadata
        g.create_dataset("O", data=data["O"])
        g.create_dataset("t", data=data["t"])
        for name, value in provenance_arrays.items():
            g.create_dataset(name, data=value)
    return path
