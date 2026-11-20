"""Save an optimization result as a low-memory, QCSOR-ready HDF5 file.

save_decomposition(path, source) accepts an OuterOptimizationResult,
ExponentialEvaluation, GaugeGradient, PartitionOptimizationResult or a plain
FixedDecomposition. It writes the terms already laid out the way QCSOR stores
them, so ipie.hamiltonians.sor_base.QCSOR.from_file reads plain arrays and
never builds per-channel Python objects:

  attrs         format/version, policy, decomp_type, nelec, mu=(mu_up,mu_down),
                tau1, tau2, partition, amplitude_bound, denom and
                lambda_electronic of the retained terms, denom_all of all
                channels, drop_identity, dropped_identity, term counts
  h1, v0        h' = h - 1/2 sum_g L_g^2 (what trial.build and local_energy use)
                and v0 = 1/2 sum_g L_g^2 for both spins (h1 + v0 = bare h)
  chol_X        rotated factor eigenvectors (r,n,n), columns ascending
  chol_ell      rotated factor eigenvalues (r,n); QCSOR integrals W = ell ell^T
  h1_X, h1_eps  h eigenvectors (n,n) and eps - mu_s for each h1 basis (k,n);
                k=1 if mu_up==mu_down or a single spin, else 2 (up, down)
  a             all Green weights (T,) in the final global term order
  p1, d1        one-body modes/amplitudes (T1,), global terms [0,T1)
  p2, d2        two-body modes/amplitudes (T2,2), global terms [T1,T)
  keys          (K,5) rows [bix, spin0, spin1, start, stop]; spin1=-1 marks
                one-body keys; bix<r are factors, bix>=r are h1 bases
  provenance    JSON record (statuses, losses, solver settings), user metadata,
                the gauge O, the source log-ratios t, and theta/reference/
                initial_theta when available (not read by QCSOR)

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
FORMAT_VERSION = 2

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
    """Plain arrays of a canonical FixedDecomposition, read straight from its tables."""
    one, two = decomposition.one_body, decomposition.two_body
    if not two.canonical:
        raise ValueError("Only canonical decompositions (from the builders or the optimizer) can be "
                         "saved; with_channels results are not")
    gauge = two.gauge
    ot, tt, src = one.table, two.table, two.sources
    nsrc = len(src)
    return {
        "policy": two.policy, "nelec": two.nelec, "denom": float(decomposition.denom),
        "h1_X": np.asarray(one.X, dtype=np.float64), "eps": np.asarray(one.eigenvalues, dtype=np.float64),
        "tau1": float(one.tau1), "shift": tuple(float(x) for x in one.shift),
        "s1": np.array(ot.spin, dtype=np.int64), "a1": np.array(ot.a, dtype=np.float64),
        "m1": np.array(ot.mode, dtype=np.int64), "d1": np.array(ot.d, dtype=np.float64),
        "O": np.asarray(gauge.O, dtype=np.float64), "chol": np.asarray(gauge.chol, dtype=np.float64),
        "ell": np.asarray(gauge.eigenvalues, dtype=np.float64), "X": np.asarray(gauge.X, dtype=np.float64),
        "tau2": float(two.tau2), "partition": str(two.partition),
        "amplitude_bound": float(two.amplitude_bound), "t": np.asarray(two.t, dtype=np.float64).reshape(-1),
        "src_bix": np.array(src.bix, dtype=np.int64), "src_modes": _pairs(src.modes, np.int64, nsrc),
        "amplitudes": np.asarray(two.amplitudes, dtype=np.float64).reshape(nsrc, 2),
        "slices": _pairs(src.slices, np.int64, nsrc),
        "bix2": np.array(tt.bix, dtype=np.int64), "spins2": np.array(tt.spins, dtype=np.int64),
        "a2": np.array(tt.a, dtype=np.float64), "modes2": np.array(tt.modes, dtype=np.int64),
        "d2": np.array(tt.d, dtype=np.float64),
    }


def _check_arrays(data, rtol=1e-12):
    """Vectorized consistency checks of a channel table; returns (n, r)."""
    X1, eps, chol, ell, X = data["h1_X"], data["eps"], data["chol"], data["ell"], data["X"]
    n, r = X1.shape[0], chol.shape[0]
    _require(data["policy"] in POLICY_TO_DECOMP_TYPE, f"Unknown policy {data['policy']!r}")
    _require(X1.shape == (n, n) and eps.shape == (n,), "Invalid one-body eigensystem shape")
    _require(chol.shape == (r, n, n) and ell.shape == (r, n) and X.shape == (r, n, n)
             and data["O"].shape == (r, r), "Invalid Cholesky gauge shapes")
    _require(np.allclose(X1.T @ X1, np.eye(n), rtol=0, atol=1e-10), "One-body eigenvectors are not orthonormal")
    scale = max(1., float(np.max(np.abs(chol), initial=0.)))
    _require(np.allclose(np.einsum("gpa,ga,gqa->gpq", X, ell, X), chol, rtol=0, atol=1e-10*scale),
             "Factor eigenpairs do not reconstruct the rotated factors")
    _require(np.allclose(data["O"] @ data["O"].T, np.eye(r), rtol=0, atol=1e-10), "Gauge is not orthogonal")

    s1, a1, m1, d1 = data["s1"], data["a1"], data["m1"], data["d1"]
    _require(s1.size == a1.size == m1.size == d1.size, "Inconsistent one-body channel arrays")
    _require(np.all((s1 == 0) | (s1 == 1)) and np.all((m1 >= 0) & (m1 < n)), "Invalid one-body spins or modes")
    _require(np.all(np.isfinite(a1) & (a1 > 0)), "One-body weights must be finite and positive")
    shift = data["shift"]
    expected = (np.asarray(shift)[s1] - eps[m1])*data["tau1"]
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

    src_bix, src_modes, amplitudes, slices = data["src_bix"], data["src_modes"], data["amplitudes"], data["slices"]
    nsrc = src_bix.size
    _require(src_modes.shape == amplitudes.shape == slices.shape == (nsrc, 2) and data["t"].shape == (nsrc,),
             "Inconsistent source arrays")
    if nsrc == 0:
        _require(nch == 0, "Channels without sources")
    else:
        _require(slices[0, 0] == 0 and slices[-1, 1] == nch and np.all(slices[1:, 0] == slices[:-1, 1])
                 and np.all(slices[:, 1] >= slices[:, 0]), "Source channel slices must be contiguous and ordered")
        W = ell[src_bix, src_modes[:, 0]]*ell[src_bix, src_modes[:, 1]]
        _require(np.all(np.isclose(amplitudes[:, 0]*amplitudes[:, 1], -W*data["tau2"], rtol=rtol, atol=1e-300)),
                 "A source violates d_left*d_right = -W*tau2")
        owner = np.repeat(np.arange(nsrc), slices[:, 1] - slices[:, 0])
        _require(np.all(bix2 == src_bix[owner]), "Channel basis index disagrees with its source")
        reference = np.abs(amplitudes[owner])
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
    n, r = data["h1_X"].shape[0], data["chol"].shape[0]
    shift, nelec = data["shift"], data["nelec"]

    spins = [int(s) for s in np.unique(data["s1"])]
    if not spins:
        h1_eps, h1_bix = np.zeros((0, n)), {}
    elif len(spins) == 1 or shift[0] == shift[1]:
        h1_eps, h1_bix = (data["eps"] - shift[spins[0]])[None, :], {s: r for s in spins}
    else:
        h1_eps = np.stack([data["eps"] - shift[s] for s in spins])
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
    h = (data["h1_X"]*data["eps"]) @ data["h1_X"].T
    return {
        "decomp_type": POLICY_TO_DECOMP_TYPE[data["policy"]],
        "h1": h - .5*L2, "v0": np.tile(.5*L2[None, :, :], (2, 1, 1)),
        "chol_X": data["X"], "chol_ell": data["ell"], "h1_X": data["h1_X"], "h1_eps": h1_eps.reshape(-1, n),
        "a": a, "p1": data["m1"][idx1].astype(np.int64), "d1": d1[idx1],
        "p2": modes2[idx2].astype(np.int64).reshape(-1, 2), "d2": d2[idx2].reshape(-1, 2),
        "keys": np.array(rows, dtype=np.int64).reshape(-1, 5),
        "denom": denom,
        "lambda_electronic": denom + (0. if nelec is None else shift[0]*nelec[0] + shift[1]*nelec[1]),
        "dropped": int(s1.size + spin_class.size - T1 - T2),
    }


def save_decomposition(path, source, *, drop_identity=False, metadata=None, overwrite=False):
    """Write source (an optimization result or a FixedDecomposition) as a low-memory QCSOR file.

    Read it back with ipie.hamiltonians.sor_base.QCSOR.from_file(path).
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
        f.attrs["nbasis"] = int(data["h1_X"].shape[0])
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
        for name in _CORE:
            f.create_dataset(name, data=layout[name])
        g = f.create_group("provenance")
        g.attrs["record"] = record
        g.attrs["user_metadata"] = user_metadata
        g.create_dataset("O", data=data["O"])
        g.create_dataset("t", data=data["t"])
        for name, value in provenance_arrays.items():
            g.create_dataset(name, data=value)
    return path
