"""Deterministic checks of all ERI entries; no fitting or tensor repair."""

import numpy as np

from ipie.hamiltonians.optimize_cholesky.models import ErrorMetric, ValidationError, ValidationOptions, ValidationReport


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



