"""Cholesky gauges and exact, fixed-partition density-channel construction."""

from .models import (
    Channel, ErrorMetric, GaugeState, PreparedProblem, TensorData,
    ValidationError, ValidationOptions, ValidationReport,
    DensitySource, FixedDecomposition, OneBodyChannel, OneBodyDecomposition,
    TwoBodyDecomposition,
)
from .prepare import (
    load_problem, prepare_problem, select_sector_policy,
    build_gauge_state, diagonalize_factors, exponential_gauge, givens_gauge,
    rotate_cholesky, skew_from_parameters,
)
from .validation import validate_gauge_state, validate_inputs
from .channels import (
    fixed_partition_amplitudes,
    enumerate_sources, build_two_body_channels, build_one_body_channels,
    build_fixed_decomposition, two_body_rotation_matrix, one_body_rotation_matrix,
    get_rotation_matrix,
)

__all__ = [
    "Channel", "ErrorMetric", "GaugeState", "PreparedProblem", "TensorData",
    "ValidationError", "ValidationOptions", "ValidationReport", "load_problem",
    "prepare_problem", "select_sector_policy", "build_gauge_state",
    "diagonalize_factors", "exponential_gauge", "givens_gauge", "rotate_cholesky",
    "skew_from_parameters", "validate_gauge_state", "validate_inputs",
    "DensitySource", "FixedDecomposition", "OneBodyChannel", "OneBodyDecomposition",
    "TwoBodyDecomposition", "fixed_partition_amplitudes", "enumerate_sources",
    "build_two_body_channels", "build_one_body_channels", "build_fixed_decomposition",
    "two_body_rotation_matrix", "one_body_rotation_matrix", "get_rotation_matrix",
]

