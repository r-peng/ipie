"""Inputs, HDF5 loading, validation, and exact orthogonal Cholesky gauges."""

import numpy as np
from ipie.hamiltonians.optimize_cholesky.models import (
    ErrorMetric,
    ValidationError,
    ValidationOptions,
    ValidationReport,
)
from numbers import Integral, Real
from pathlib import Path
from scipy.linalg import expm
from .models import PreparedProblem, TensorData, GaugeState


# Validation and tensor checks

def _options(options):
    if options is None:
        return ValidationOptions()
    if not isinstance(options, ValidationOptions):
        raise TypeError("options must be a ValidationOptions instance")
    return options


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValidationError(f"{name} must have a real numeric dtype; got {array.dtype}")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValidationError(f"{name} contains NaN, infinity, or float64 overflow")
    return array


def _metric(candidate, reference, options):
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    if candidate.shape != reference.shape:
        raise ValidationError(f"Cannot compare shapes {candidate.shape} and {reference.shape}")
    diff = candidate - reference
    err = _frobenius(diff)
    norm = _frobenius(reference)
    maxerr = float(np.max(np.abs(diff), initial=0.0))
    maxref = float(np.max(np.abs(reference), initial=0.0))
    return _make_metric(maxerr, maxref, err, norm, options)


def _frobenius(array):
    """Scaled norm avoids squaring unscaled very large/small entries."""
    scale = float(np.max(np.abs(array), initial=0.0))
    return 0.0 if scale == 0 else float(scale * np.linalg.norm((array / scale).ravel()))


def _make_metric(maxerr, maxref, err, norm, options):
    values = (maxerr, maxref, err, norm)
    if not all(np.isfinite(value) for value in values):
        raise ValidationError("Numerical overflow while evaluating a validation residual")
    threshold = float(options.atol + options.rtol * maxref)
    if not np.isfinite(threshold):
        raise ValidationError("Validation tolerances overflow at this reference scale")
    relative = err / norm if norm != 0 else (0.0 if err == 0 else None)
    if relative is not None and not np.isfinite(relative):
        relative = None
    return ErrorMetric(maxerr, maxref, err, norm, relative, threshold, bool(maxerr <= threshold))


def _require_report(report):
    failures = [f"{key}: max_abs={m.max_abs_error:.3e} > {m.tolerance:.3e}"
                for key, m in report.checks.items() if not m.passed]
    if failures:
        raise ValidationError("Validation failed: " + "; ".join(failures), report)
    return report


def _factor_array(chol, options):
    chol = _real_array(chol, "chol")
    if chol.ndim != 3 or chol.shape[0] < 1 or chol.shape[1] < 1 or chol.shape[1] != chol.shape[2]:
        raise ValidationError("chol must have shape (r,n,n) with r>=1 and n>=1")
    checks = {f"chol_symmetry[{g}]": _metric(L, L.T, options)
              for g, L in enumerate(chol)}
    _require_report(ValidationReport(checks, options))
    return chol


def _gauge_array(O, nchol, options):
    O = _real_array(O, "O")
    if O.shape != (nchol, nchol):
        raise ValidationError(f"O must have shape {(nchol, nchol)}; got {O.shape}")
    metric = _metric(O.T @ O, np.eye(nchol), options)
    _require_report(ValidationReport({"gauge_orthogonality": metric}, options))
    return O


def _tensor_arrays(data, options):
    h1 = _real_array(data.h1, "h1")
    if h1.ndim != 2 or h1.shape[0] < 1 or h1.shape[0] != h1.shape[1]:
        raise ValidationError("h1 must have shape (n,n) with n>=1")
    n = h1.shape[0]
    eri = _real_array(data.eri, "eri")
    if eri.shape != (n, n, n, n):
        raise ValidationError(f"eri must have shape {(n,n,n,n)} in (p,r,q,s) order; got {eri.shape}")
    chol = _factor_array(data.chol, options)
    if chol.shape[1:] != (n, n):
        raise ValidationError("h1 and chol have inconsistent spatial dimensions")
    return h1, eri, chol


def _eri_metric(candidate_chol, options, *, eri=None, reference_chol=None):
    """Compare every pair-matrix entry using only block_size*n^2 temporaries."""
    C = candidate_chol.reshape(candidate_chol.shape[0], -1)
    size = C.shape[1]
    R = None if reference_chol is None else reference_chol.reshape(reference_chol.shape[0], -1)
    # A non-C tensor must be indexed by row block; reshape would copy all n^4.
    V = eri.reshape(size, size) if eri is not None and eri.flags.c_contiguous else None
    n = candidate_chol.shape[1]
    maxerr = maxref = err = norm = 0.0
    for first in range(0, size, options.block_size):
        last = min(first + options.block_size, size)
        actual = C[:, first:last].T @ C
        if R is not None:
            target = R[:, first:last].T @ R
        elif V is not None:
            target = V[first:last]
        else:
            rows = np.arange(first, last)
            target = eri[rows // n, rows % n, :, :].reshape(last-first, size)
        delta = actual - target
        maxerr = max(maxerr, float(np.max(np.abs(delta), initial=0.0)))
        maxref = max(maxref, float(np.max(np.abs(target), initial=0.0)))
        err = float(np.hypot(err, _frobenius(delta)))
        norm = float(np.hypot(norm, _frobenius(target)))
    return _make_metric(maxerr, maxref, err, norm, options)


def validate_inputs(data, options=None):
    """Check real finite shapes, per-factor symmetry, and exact ERI recovery.

    The reconstruction checks every eri[p,r,q,s] entry. All physical ERI
    permutation symmetries follow (within numerical tolerance) from equality
    to symmetric-factor reconstruction. No symmetrization is performed.
    Raises ValidationError (a ValueError subclass) on failure.
    """
    options = _options(options)
    h1, eri, chol = _tensor_arrays(data, options)
    checks = {"h1_symmetry": _metric(h1, h1.T, options)}
    checks.update({f"chol_symmetry[{g}]": _metric(L, L.T, options)
                   for g, L in enumerate(chol)})
    checks["input_eri_reconstruction"] = _eri_metric(chol, options, eri=eri)
    return _require_report(ValidationReport(checks, options))


def validate_gauge_state(data, state, options=None):
    """Check a state against original tensors, independently of its diagnostics.

    Verifies O.T@O, the specified rotation, eigen reconstruction and X.T@X,
    rotated ERIs versus supplied ERIs, direct ERI gauge invariance, and
    invariance of sum_gamma L_gamma@L_gamma. Degeneracies are allowed.
    """
    options = _options(options)
    original_report = validate_inputs(data, options)
    _, eri, original = _tensor_arrays(data, options)
    r, n, _ = original.shape
    O = _gauge_array(state.O, r, options)
    rotated = _factor_array(state.chol, options)
    if rotated.shape != original.shape:
        raise ValidationError("Rotated chol must retain the original (r,n,n) shape")
    X = _real_array(state.X, "X")
    ell = _real_array(state.eigenvalues, "eigenvalues")
    if X.shape != (r,n,n) or ell.shape != (r,n):
        raise ValidationError(f"Expected X.shape={(r,n,n)} and eigenvalues.shape={(r,n)}")
    checks = dict(original_report.checks)
    checks["gauge_orthogonality"] = _metric(O.T @ O, np.eye(r), options)
    expected = (O @ original.reshape(r, -1)).reshape(r, n, n)
    for g in range(r):
        checks[f"specified_rotation[{g}]"] = _metric(rotated[g], expected[g], options)
        checks[f"eigenvector_orthogonality[{g}]"] = _metric(X[g].T @ X[g], np.eye(n), options)
        checks[f"eigenpair_reconstruction[{g}]"] = _metric(
            (X[g] * ell[g][None, :]) @ X[g].T, rotated[g], options)
    checks["rotated_eri_reconstruction"] = _eri_metric(rotated, options, eri=eri)
    checks["eri_gauge_invariance"] = _eri_metric(rotated, options, reference_chol=original)
    original_square_sum = np.einsum("gpr,grs->ps", original, original)
    rotated_square_sum = np.einsum("gpr,grs->ps", rotated, rotated)
    checks["normal_order_invariance"] = _metric(rotated_square_sum, original_square_sum, options)
    return _require_report(ValidationReport(checks, options))


# Problem preparation, gauge construction, and factor diagonalization

def select_sector_policy(nelec, nbasis):
    """Return canonical particle counts and hybrid/ab_only/aa_only policy.

    Counts are optional in step 1; None selects the generic hybrid policy.
    A down-only sector is left generic rather than silently relabeling spins.
    """
    if nelec is None:
        return None, "hybrid"
    try:
        nelec = tuple(nelec)
    except TypeError as exc:
        raise ValueError("nelec must be (N_up,N_down)") from exc
    if len(nelec) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral)
        or not 0 <= v <= nbasis for v in nelec
    ):
        raise ValueError(f"nelec must contain two integers between 0 and {nbasis}")
    counts = tuple(int(v) for v in nelec)
    if counts[1] == 0:
        return counts, "aa_only"
    if counts == (1, 1):
        return counts, "ab_only"
    return counts, "hybrid"


def prepare_problem(h1, eri, chol, nelec=None, options=None):
    """Validate exact tensors and return owned, read-only float64 arrays.

    h1 is the original physical h, never h'. No normal-order correction is
    applied. No tensors are symmetrized or refactorized. No caller arrays
    are modified. eri uses the explicit axes (p,r,q,s), not (p,q,r,s).
    """
    options = _options(options)
    raw = TensorData(np.asarray(h1), np.asarray(eri), np.asarray(chol))
    report = validate_inputs(raw, options)
    counts, policy = select_sector_policy(nelec, raw.nbasis)
    arrays = [np.array(a, dtype=np.float64, order="C", copy=True)
              for a in (raw.h1, raw.eri, raw.chol)]
    for array in arrays:
        array.setflags(write=False)
    return PreparedProblem(TensorData(*arrays), report, counts, policy)


def _read_dataset(path, key):
    # Import lazily so array-only use does not require the optional I/O module.
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("HDF5 loading requires h5py; install it with 'pip install h5py'") from exc
    path = Path(path)
    if not isinstance(key, str) or not key:
        raise ValueError("HDF5 dataset keys must be nonempty strings")
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Dataset {key!r} not found in {str(path)!r}")
        dataset = handle[key]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{key!r} in {str(path)!r} is not an HDF5 dataset")
        return dataset[()]


def load_problem(h1_path, eri_path=None, chol_path=None, *, h1_key="h1", eri_key="eri",
                 chol_key="chol", nelec=None, options=None):
    """Read one or three HDF5 files, close them, and call prepare_problem.

    If eri_path/chol_path are omitted, read those datasets from h1_path.
    Keys may include HDF5 groups, e.g. 'integrals/h1'. Full arrays are loaded
    into memory; this is not an out-of-core loader. No disk writes occur.
    """
    eri_path = h1_path if eri_path is None else eri_path
    chol_path = h1_path if chol_path is None else chol_path
    h1 = _read_dataset(h1_path, h1_key)
    eri = _read_dataset(eri_path, eri_key)
    chol = _read_dataset(chol_path, chol_key)
    return prepare_problem(h1, eri, chol, nelec=nelec, options=options)

def _dimension(nchol):
    if isinstance(nchol, (bool, np.bool_)) or not isinstance(nchol, Integral) or nchol < 1:
        raise ValueError("nchol must be a positive integer")
    return int(nchol)


def skew_from_parameters(theta, nchol):
    """Pack theta into upper triangle: (0,1),(0,2),...,(1,2),... .

    S[p,q]=theta[k] for p<q, S[q,p]=-theta[k], and diagonal zero.
    """
    r = _dimension(nchol)
    theta = _real_array(theta, "theta")
    count = r * (r - 1) // 2
    if theta.shape != (count,):
        raise ValueError(f"theta must be a length-{count} vector for {r} factors")
    S = np.zeros((r, r), dtype=np.float64)
    rows, cols = np.triu_indices(r, 1)
    S[rows, cols] = theta
    S[cols, rows] = -theta
    return S


def exponential_gauge(theta, nchol, reference=None):
    """Return O=expm(S(theta)) @ reference (reference defaults to identity).

    Exponential coordinates stay in the determinant component of reference.
    A reflection reference is permitted, including for r=1.
    """
    r = _dimension(nchol)
    reference = np.eye(r) if reference is None else _gauge_array(reference, r, _options(None))
    result = expm(skew_from_parameters(theta, r)) @ reference
    return _gauge_array(result, r, _options(None))


def givens_gauge(nchol, rotations, reference=None):
    """Left-multiply reference by successive (p,q,angle) plane rotations.

    Require 0<=p<q<r. The active block is [[cos,sin],[-sin,cos]].
    In list order, O <- G(p,q,angle) @ O; the last rotation is leftmost.
    Only the two affected rows are updated; no dense G is constructed.
    """
    r = _dimension(nchol)
    result = np.eye(r) if reference is None else _gauge_array(reference, r, _options(None)).copy()
    for rotation in rotations:
        try:
            p, q, angle = rotation
        except (TypeError, ValueError) as exc:
            raise ValueError("Each Givens rotation must be (p,q,angle)") from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in (p,q)):
            raise ValueError("Givens indices must be integers")
        if not 0 <= p < q < r:
            raise ValueError(f"Require 0 <= p < q < {r}; got {(p,q)}")
        if not isinstance(angle, Real) or not np.isfinite(angle):
            raise ValueError("Givens angle must be finite and real (radians)")
        cosine, sine = np.cos(angle), np.sin(angle)
        row_p, row_q = result[p].copy(), result[q].copy()
        result[p] = cosine * row_p + sine * row_q
        result[q] = -sine * row_p + cosine * row_q
    return _gauge_array(result, r, _options(None))


def rotate_cholesky(chol, O, options=None):
    """Return rotated[eta,p,r] = sum_gamma O[eta,gamma]*chol[gamma,p,r]."""
    options = _options(options)
    chol = _factor_array(chol, options)
    O = _gauge_array(O, chol.shape[0], options)
    return (O @ chol.reshape(chol.shape[0], -1)).reshape(chol.shape)


def diagonalize_factors(chol, options=None):
    """Return (ell[gamma,alpha], X[gamma,p,alpha]) with ascending ell.

    All modes are retained. No clipping, low-rank truncation, symmetrization,
    or perturbation of degeneracies is performed. Sign/basis choices within
    degenerate eigenspaces are those of numpy.linalg.eigh.
    """
    chol = _factor_array(chol, _options(options))
    return np.linalg.eigh(chol)


def near_zero_eigenvalues(state, *, rtol=1e-6, atol=0.0):
    """Return ((factor, mode, eigenvalue), ...) with |ell| <= atol+rtol*max_mode|ell_factor|.

    The relaxed reduced loss depends on |ell_{gamma,alpha}| (the inner optimum
    over a source's partition is 2*sqrt(A*B), proportional to |ell_alpha*ell_beta|),
    so it has a kink on every surface ell_{gamma,alpha}(O)=0. A minimizer that
    makes a rotated factor rank-deficient sits on such a kink, where gradient
    tests cannot pass. Use this after a line-search termination to see which
    kinks are active. The scale is per factor, so it detects rank deficiency.

    state may be a GaugeState, a FixedDecomposition, or any result with a
    .decomposition attribute (outer or partition optimization results).
    """
    if not isinstance(rtol, Real) or not isinstance(atol, Real) or rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be nonnegative real numbers")
    gauge = state
    if hasattr(gauge, "decomposition") and not hasattr(gauge, "eigenvalues"):
        gauge = gauge.decomposition
    if hasattr(gauge, "two_body"):
        gauge = gauge.two_body.gauge
    if not hasattr(gauge, "eigenvalues"):
        raise TypeError("state must be a GaugeState, FixedDecomposition, or a result with .decomposition")
    ell = np.asarray(gauge.eigenvalues, dtype=np.float64)
    found = []
    for g, values in enumerate(ell):
        tolerance = atol+rtol*np.max(np.abs(values), initial=0.0)
        found.extend((g, int(a), float(values[a])) for a in np.flatnonzero(np.abs(values) <= tolerance))
    return tuple(found)


def build_gauge_state(data, O=None, options=None):
    """Build and fully validate a step-1 state against supplied exact ERIs.

    This full-validation function is intended for initial/reference checks,
    not for every future optimizer loss evaluation. Low-level rotation/eigh
    functions above do not reconstruct the ERI tensor.
    """
    options = _options(options)
    chol = _factor_array(data.chol, options)
    O = np.eye(chol.shape[0]) if O is None else O
    O = _gauge_array(O, chol.shape[0], options)
    rotated = rotate_cholesky(chol, O, options)
    ell, X = diagonalize_factors(rotated, options)
    state = GaugeState(O.copy(), rotated, ell, X)
    report = validate_gauge_state(data, state, options)
    # Own immutable snapshots; never mutate the caller's supplied arrays.
    for array in (state.O, state.chol, state.eigenvalues, state.X):
        array.setflags(write=False)
    return GaugeState(state.O, state.chol, state.eigenvalues, state.X, report)
