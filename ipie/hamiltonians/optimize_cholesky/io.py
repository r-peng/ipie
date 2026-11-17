"""Save and load optimized decompositions as self-contained HDF5 files.

save_decomposition(path, source) accepts an OuterOptimizationResult,
ExponentialEvaluation, GaugeGradient, PartitionOptimizationResult or a plain
FixedDecomposition and writes two layers:

  product     everything a sampler needs, with no recomputation: the h
              eigensystem, tau1 and the spin-resolved shift; every rotated
              factor with its eigenvalues and eigenvectors; the source table
              (t, amplitudes, slices); the flat one- and two-body channel
              tables (bix, spins, modes, d, a_i); policy, nelec, tau2,
              partition map and amplitude bound.
  provenance  the gauge O (and theta/reference for outer results), loss kind,
              losses, statuses and solver settings, plus optional original
              tensors (h1, chol and optionally eri) for an exact rebuild.

load_decomposition(path) returns a SavedDecomposition whose .decomposition is
an ordinary FixedDecomposition, reconstructed from the stored arrays and
checked for internal consistency. rebuild_decomposition(saved) re-derives it
from the original tensors with build_fixed_decomposition and verifies that the
two agree. Pickle is deliberately not used: record layouts evolve, whereas this
format is versioned and independent of the Python classes.

h5py is imported lazily, as in problem.load_problem.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Optional

import numpy as np

from .channels import build_fixed_decomposition
from .models import (
    Channel,
    DensitySource,
    ExponentialEvaluation,
    FixedDecomposition,
    GaugeGradient,
    GaugeState,
    OneBodyChannel,
    OneBodyDecomposition,
    OuterOptimizationResult,
    PartitionOptimizationResult,
    TwoBodyDecomposition,
)
from .problem import build_gauge_state, prepare_problem
from .shift import check_one_body_shift


FORMAT = "lafqmc-optimized-decomposition"
FORMAT_VERSION = 1
SOURCE_KINDS = ("two_branch", "spin_quartet")


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Saving/loading decompositions requires h5py; install it with "
                          "'pip install h5py'") from exc
    return h5py


def _readonly(array, dtype=np.float64):
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


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


@dataclass(frozen=True)
class SavedDecomposition:
    """A loaded file: the decomposition, its provenance and optional input tensors."""

    decomposition: FixedDecomposition
    metadata: dict
    arrays: dict = field(default_factory=dict)
    tensors: Optional[dict] = None
    path: Optional[str] = None

    @property
    def nelec(self):
        return self.decomposition.two_body.nelec

    @property
    def policy(self):
        return self.decomposition.two_body.policy

    @property
    def shift(self):
        return self.decomposition.shift

    @property
    def loss_kind(self):
        inner = self.metadata.get("provenance", {}).get("inner", {})
        return inner.get("loss_kind")

    def problem(self, *, options=None):
        """PreparedProblem from the stored original tensors (eri rebuilt from chol if absent)."""
        if self.tensors is None:
            raise ValueError("This file stores no original tensors; save with problem=...")
        h1, chol = self.tensors["h1"], self.tensors["chol"]
        eri = self.tensors.get("eri")
        if eri is None:
            eri = np.einsum("gpr,gqs->prqs", chol, chol)
        return prepare_problem(h1, eri, chol, nelec=self.nelec, options=options)


def _string_array(values):
    return np.array([v.encode("utf-8") for v in values], dtype="S")


def save_decomposition(path, source, *, problem=None, store_eri=False, metadata=None,
                       overwrite=False):
    """Write source (a result or a FixedDecomposition) to an HDF5 file; return the path.

    problem: the PreparedProblem used for the optimization. When given, the
    original h1 and chol (and eri if store_eri=True) are stored so that
    rebuild_decomposition can re-derive and verify the decomposition.
    metadata: optional JSON-compatible user dictionary stored verbatim.
    An existing file raises FileExistsError unless overwrite=True.
    """
    h5py = _h5py()
    decomposition, record, provenance_arrays = _unpack(source)
    one, two = decomposition.one_body, decomposition.two_body
    gauge = two.gauge
    check_one_body_shift(one)
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    if problem is not None:
        if problem.data.nbasis != one.X.shape[0] or problem.data.nchol != gauge.chol.shape[0]:
            raise ValueError("problem dimensions do not match the decomposition")
        if problem.policy != two.policy or problem.nelec != two.nelec:
            raise ValueError("problem policy/particle numbers do not match the decomposition")

    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = FORMAT
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["created"] = datetime.now(timezone.utc).isoformat()
        handle.attrs["policy"] = two.policy
        handle.attrs["has_nelec"] = two.nelec is not None
        handle.attrs["nelec"] = np.array(two.nelec if two.nelec is not None else (-1, -1), dtype=np.int64)
        handle.attrs["nbasis"] = int(one.X.shape[0])
        handle.attrs["nchol"] = int(gauge.chol.shape[0])
        handle.attrs["denom"] = float(decomposition.denom)
        handle.attrs["lambda_electronic"] = float(decomposition.lambda_electronic) \
            if (two.nelec is not None or not any(decomposition.shift)) else float("nan")

        g = handle.create_group("one_body")
        g.create_dataset("X", data=np.asarray(one.X, dtype=np.float64))
        g.create_dataset("eigenvalues", data=np.asarray(one.eigenvalues, dtype=np.float64))
        g.attrs["tau1"] = float(one.tau1)
        g.attrs["shift"] = np.asarray(one.shift, dtype=np.float64)
        c = g.create_group("channels")
        c.create_dataset("spin", data=np.array([ch.spin for ch in one.channels], dtype=np.int64))
        c.create_dataset("a_i", data=np.array([ch.a_i for ch in one.channels], dtype=np.float64))
        c.create_dataset("mode", data=np.array([ch.mode for ch in one.channels], dtype=np.int64))
        c.create_dataset("d", data=np.array([ch.d for ch in one.channels], dtype=np.float64))

        g = handle.create_group("gauge")
        for name in ("O", "chol", "eigenvalues", "X"):
            g.create_dataset(name, data=np.asarray(getattr(gauge, name), dtype=np.float64))

        g = handle.create_group("two_body")
        g.attrs["tau2"] = float(two.tau2)
        g.attrs["partition"] = two.partition
        g.attrs["amplitude_bound"] = float(two.amplitude_bound)
        nsrc, nch = len(two.sources), len(two.channels)
        s = g.create_group("sources")
        s.create_dataset("bix", data=np.array([x.bix for x in two.sources], dtype=np.int64))
        s.create_dataset("modes", data=np.array([x.modes for x in two.sources], dtype=np.int64).reshape(nsrc, 2))
        s.create_dataset("kind", data=np.array([SOURCE_KINDS.index(x.kind) for x in two.sources], dtype=np.int64))
        s.create_dataset("spins", data=np.array([(-1, -1) if x.spins is None else x.spins
                                                 for x in two.sources], dtype=np.int64).reshape(nsrc, 2))
        g.create_dataset("t", data=np.asarray(two.t, dtype=np.float64))
        g.create_dataset("amplitudes", data=np.asarray(two.amplitudes, dtype=np.float64).reshape(nsrc, 2))
        g.create_dataset("source_slices", data=np.array(two.source_slices, dtype=np.int64).reshape(nsrc, 2))
        c = g.create_group("channels")
        c.create_dataset("bix", data=np.array([x.bix for x in two.channels], dtype=np.int64))
        c.create_dataset("spins", data=np.array([x.spins for x in two.channels], dtype=np.int64).reshape(nch, 2))
        c.create_dataset("a_i", data=np.array([x.a_i for x in two.channels], dtype=np.float64))
        c.create_dataset("modes", data=np.array([x.modes for x in two.channels], dtype=np.int64).reshape(nch, 2))
        c.create_dataset("d", data=np.array([x.d for x in two.channels], dtype=np.float64).reshape(nch, 2))

        g = handle.create_group("provenance")
        g.attrs["record"] = json.dumps(_jsonable(record), allow_nan=False)
        g.attrs["user_metadata"] = json.dumps(_jsonable(metadata or {}), allow_nan=False)
        for name, value in provenance_arrays.items():
            g.create_dataset(name, data=value)

        if problem is not None:
            t = handle.create_group("tensors")
            t.create_dataset("h1", data=np.asarray(problem.data.h1, dtype=np.float64))
            t.create_dataset("chol", data=np.asarray(problem.data.chol, dtype=np.float64))
            if store_eri:
                t.create_dataset("eri", data=np.asarray(problem.data.eri, dtype=np.float64))
    return str(path)


def _check_loaded(decomposition, stored_denom, *, rtol=1e-12):
    """Internal-consistency checks; raise ValueError for a corrupt or edited file."""
    one, two = decomposition.one_body, decomposition.two_body
    gauge = two.gauge
    check_one_body_shift(one)
    if decomposition.denom != stored_denom:
        raise ValueError("Stored denom does not match the stored channel weights")
    n = one.X.shape[0]
    scale = max(1.0, float(np.max(np.abs(gauge.chol), initial=0.)))
    rebuilt = np.einsum("gpa,ga,gqa->gpq", gauge.X, gauge.eigenvalues, gauge.X)
    if not np.allclose(rebuilt, gauge.chol, rtol=0, atol=1e-10*scale):
        raise ValueError("Stored factor eigenpairs do not reconstruct the stored factors")
    if not np.allclose(gauge.O @ gauge.O.T, np.eye(gauge.O.shape[0]), rtol=0, atol=1e-10):
        raise ValueError("Stored gauge is not orthogonal")
    h = (one.X*one.eigenvalues) @ one.X.T
    if not np.allclose(one.X.T @ one.X, np.eye(n), rtol=0, atol=1e-10) or not np.allclose(h, h.T):
        raise ValueError("Stored one-body eigensystem is not orthonormal")
    for i, (source, d, (start, stop)) in enumerate(zip(two.sources, two.amplitudes, two.source_slices)):
        W = gauge.eigenvalues[source.bix, source.modes[0]]*gauge.eigenvalues[source.bix, source.modes[1]]
        if not math.isclose(d[0]*d[1], -W*two.tau2, rel_tol=rtol, abs_tol=1e-300):
            raise ValueError(f"Source {i} violates d_left*d_right = -W*tau2")
        if not 0 <= start <= stop <= len(two.channels):
            raise ValueError(f"Source {i} has an invalid channel slice")
        for channel in two.channels[start:stop]:
            if channel.bix != source.bix or abs(abs(channel.d[0]) - abs(d[0])) > rtol*abs(d[0]) \
                    or abs(abs(channel.d[1]) - abs(d[1])) > rtol*abs(d[1]):
                raise ValueError(f"Source {i} channels disagree with its amplitudes")


def load_decomposition(path):
    """Read a file written by save_decomposition; return a SavedDecomposition."""
    h5py = _h5py()
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        version = int(handle.attrs.get("format_version", -1))
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported format_version {version}; expected {FORMAT_VERSION}")
        policy = str(handle.attrs["policy"])
        nelec = tuple(int(x) for x in handle.attrs["nelec"]) if bool(handle.attrs["has_nelec"]) else None
        stored_denom = float(handle.attrs["denom"])

        g = handle["one_body"]
        shift = tuple(float(x) for x in g.attrs["shift"])
        c = g["channels"]
        one_channels = tuple(OneBodyChannel(int(s), float(a), int(m), float(d))
                             for s, a, m, d in zip(c["spin"][()], c["a_i"][()], c["mode"][()], c["d"][()]))
        one = OneBodyDecomposition(_readonly(g["X"][()]), _readonly(g["eigenvalues"][()]),
                                   float(g.attrs["tau1"]), one_channels, shift)

        g = handle["gauge"]
        gauge = GaugeState(*(_readonly(g[name][()]) for name in ("O", "chol", "eigenvalues", "X")))

        g = handle["two_body"]
        s = g["sources"]
        sources = tuple(
            DensitySource(int(b), (int(m[0]), int(m[1])), SOURCE_KINDS[int(k)],
                          None if int(sp[0]) < 0 else (int(sp[0]), int(sp[1])))
            for b, m, k, sp in zip(s["bix"][()], s["modes"][()], s["kind"][()], s["spins"][()]))
        c = g["channels"]
        channels = tuple(Channel(int(b), (int(sp[0]), int(sp[1])), float(a), (int(m[0]), int(m[1])),
                                 (float(d[0]), float(d[1])))
                         for b, sp, a, m, d in zip(c["bix"][()], c["spins"][()], c["a_i"][()],
                                                   c["modes"][()], c["d"][()]))
        two = TwoBodyDecomposition(gauge, policy, nelec, float(g.attrs["tau2"]),
                                   str(g.attrs["partition"]), float(g.attrs["amplitude_bound"]),
                                   sources, _readonly(g["t"][()]), _readonly(g["amplitudes"][()]),
                                   tuple((int(a), int(b)) for a, b in g["source_slices"][()]), channels)

        p = handle["provenance"]
        metadata = {"format": FORMAT, "format_version": version,
                    "created": str(handle.attrs.get("created", "")),
                    "provenance": json.loads(p.attrs["record"]),
                    "user": json.loads(p.attrs["user_metadata"])}
        arrays = {name: _readonly(p[name][()]) for name in p.keys()}
        tensors = None
        if "tensors" in handle:
            t = handle["tensors"]
            tensors = {name: _readonly(t[name][()]) for name in t.keys()}

    decomposition = FixedDecomposition(one, two)
    _check_loaded(decomposition, stored_denom)
    return SavedDecomposition(decomposition, metadata, arrays, tensors, str(path))


def rebuild_decomposition(saved, *, problem=None, rtol=1e-12, atol=1e-13):
    """Re-derive the saved decomposition from the original tensors and verify agreement.

    Uses the stored gauge O, partitions t, shift, scales and map. Projectors
    (not eigenvector signs), eigenvalues, amplitudes and the channel table are
    compared. Returns the rebuilt FixedDecomposition; raises ValueError on a
    mismatch.
    """
    if not isinstance(saved, SavedDecomposition):
        raise TypeError("saved must be a SavedDecomposition")
    problem = saved.problem() if problem is None else problem
    stored = saved.decomposition
    one, two = stored.one_body, stored.two_body
    gauge = build_gauge_state(problem.data, two.gauge.O)
    rebuilt = build_fixed_decomposition(problem, gauge, one.tau1, two.tau2, np.asarray(two.t),
                                        partition=two.partition, amplitude_bound=two.amplitude_bound,
                                        shift=one.shift)

    def close(a, b, what):
        if not np.allclose(a, b, rtol=rtol, atol=atol):
            raise ValueError(f"Rebuilt {what} disagrees with the saved file")

    close(rebuilt.two_body.gauge.eigenvalues, two.gauge.eigenvalues, "factor eigenvalues")
    projector = lambda X: np.einsum("...pa,...qa->...apq", X, X)
    close(projector(rebuilt.two_body.gauge.X), projector(two.gauge.X), "factor projectors")
    close(rebuilt.one_body.eigenvalues, one.eigenvalues, "one-body eigenvalues")
    close(projector(rebuilt.one_body.X), projector(one.X), "one-body projectors")
    close(rebuilt.two_body.amplitudes, two.amplitudes, "source amplitudes")
    if rebuilt.two_body.sources != two.sources or rebuilt.two_body.source_slices != two.source_slices:
        raise ValueError("Rebuilt source topology disagrees with the saved file")
    for left, right in ((rebuilt.one_body.channels, one.channels),
                        (rebuilt.two_body.channels, two.channels)):
        if len(left) != len(right):
            raise ValueError("Rebuilt channel count disagrees with the saved file")
    close([c.d for c in rebuilt.one_body.channels], [c.d for c in one.channels], "one-body amplitudes")
    close(np.array([c.d for c in rebuilt.two_body.channels]).reshape(-1, 2),
          np.array([c.d for c in two.channels]).reshape(-1, 2), "two-body amplitudes")
    for a, b in zip(rebuilt.two_body.channels, two.channels):
        if (a.bix, a.spins, a.modes, a.a_i) != (b.bix, b.spins, b.modes, b.a_i):
            raise ValueError("Rebuilt channel metadata disagrees with the saved file")
    return rebuilt
