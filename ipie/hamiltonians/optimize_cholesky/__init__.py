"""Step 1: data models, HDF5 inputs, exact Cholesky gauges and validation."""

from .models import (
    Channel, ErrorMetric, GaugeState, PreparedProblem, TensorData,
    ValidationError, ValidationOptions, ValidationReport,
)
from .prepare import load_problem, prepare_problem, select_sector_policy
from .gauge import (
    build_gauge_state, diagonalize_factors, exponential_gauge, givens_gauge,
    rotate_cholesky, skew_from_parameters,
)
from .validation import validate_gauge_state, validate_inputs

__all__ = [
    "Channel", "ErrorMetric", "GaugeState", "PreparedProblem", "TensorData",
    "ValidationError", "ValidationOptions", "ValidationReport", "load_problem",
    "prepare_problem", "select_sector_policy", "build_gauge_state",
    "diagonalize_factors", "exponential_gauge", "givens_gauge", "rotate_cholesky",
    "skew_from_parameters", "validate_gauge_state", "validate_inputs",
]
