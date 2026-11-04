"""Cholesky gauges, density channels, commutator loss and partition updates."""

from .models import (
    Channel, ErrorMetric, GaugeState, PreparedProblem, TensorData,
    ValidationError, ValidationOptions, ValidationReport,
    DensitySource, FixedDecomposition, OneBodyChannel, OneBodyDecomposition,
    TwoBodyDecomposition, CommutatorLoss, PartitionCoefficients,
    PartitionSweep, PartitionOptimizationResult,
    GaugeRegularityError, InnerConvergenceError, GradientOptions, GaugeGradient,
    GaugeDifference, GaugeGradientCheck,
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
from .objective import (
    OverlapEvaluator, commutator_norm2, dense_commutator_loss,
    overlap_commutator_loss, commutator_loss, raw_loss,
)
from .partitions import (
    analytic_partition_update, optimize_partitions, partition_bounds,
    partition_coefficients, projected_partition_gradient, with_partitions,
)
from .derivatives import (
    commutator_norm2_derivatives, eigh_vjp, exponential_gauge_vjp, givens_gauge_vjp,
)
from .gradients import (
    GaugeEvaluator, fixed_partition_gradient, normalized_partition_gradient,
    relaxed_gauge_gradient, check_gauge_direction,
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
    "CommutatorLoss", "OverlapEvaluator", "commutator_norm2", "dense_commutator_loss",
    "overlap_commutator_loss", "commutator_loss", "raw_loss",
    "PartitionCoefficients", "PartitionSweep", "PartitionOptimizationResult",
    "analytic_partition_update", "optimize_partitions", "partition_bounds",
    "partition_coefficients", "projected_partition_gradient", "with_partitions",
    "GaugeRegularityError", "InnerConvergenceError", "GradientOptions", "GaugeGradient",
    "GaugeDifference", "GaugeGradientCheck", "commutator_norm2_derivatives", "eigh_vjp",
    "exponential_gauge_vjp", "givens_gauge_vjp", "GaugeEvaluator", "fixed_partition_gradient",
    "normalized_partition_gradient", "relaxed_gauge_gradient", "check_gauge_direction",
]




