"""The equivalence standard for comparisons between code paths (decision 4).

Two computations of the same quantity agree when max|actual - expected| <=
RTOL * scale, with scale = max|expected| by default. Tests use this instead
of bitwise identity, so that a legitimate change of operation order (a
batched kernel, a reused intermediate) does not count as a regression. Exact
comparisons remain for integers, layouts, flags, and values that are exact by
construction (structural zeros, copies, inputs that must not change).
"""

import numpy as np

RTOL = 1e-12

GRADIENT_FIELDS = ("gradient_O", "fixed_gradient_O", "bound_correction_O", "tangent_gradient",
                   "generator_gradient", "partition_gradient", "eigenvalue_gradient")


def assert_close(actual, expected, *, rtol=RTOL, scale=None, label=""):
    """max|actual-expected| <= rtol*scale; scale defaults to max|expected| (exact match if that is 0)."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != {expected.shape}"
    if expected.size == 0:
        return
    diff = float(np.max(np.abs(actual - expected)))
    scale = float(np.max(np.abs(expected))) if scale is None else float(scale)
    assert diff <= rtol*scale, f"{label}: max|diff|={diff:.3e} > {rtol:.0e}*{scale:.3e}"


def _field(result, name):
    return result[name] if isinstance(result, dict) else getattr(result, name)


def assert_same_evaluation(actual, expected, *, rtol=RTOL, label=""):
    """Loss and every gauge-gradient field of a GaugeGradient against another (or an oracle dict).

    Each gradient field is scaled by max(max|field|, |loss|): at a relaxed
    solution partition_gradient and bound_correction_O are inner residuals,
    whose own size is not a meaningful scale.
    """
    loss = float(_field(expected, "loss"))
    assert_close(_field(actual, "loss"), loss, rtol=rtol, label=f"{label}loss")
    for name in GRADIENT_FIELDS:
        reference = np.asarray(_field(expected, name))
        scale = max(float(np.max(np.abs(reference), initial=0.0)), abs(loss))
        assert_close(_field(actual, name), reference, rtol=rtol, scale=scale, label=f"{label}{name}")
