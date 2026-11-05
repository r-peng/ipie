"""Reserved geometric-direct-minimization child of the common outer driver."""

from .base import OuterOptimizer


class GDMOptimizer(OuterOptimizer):
    """Extension point for GDM; geometric steps are not implemented yet.

    A future implementation supplies a manifold objective/point, its own
    retraction or exponential path, direction/history transport, acceptance
    procedure and result builder. None of those choices belong in the parent.
    """

    @staticmethod
    def _unimplemented():
        raise NotImplementedError(
            "GDMOptimizer is not implemented yet; its geometric update and solver state "
            "must be supplied in this subclass."
        )

    def _initialize(self, initial):
        self._unimplemented()

    def _step(self, point, run, iteration):
        self._unimplemented()

    def _build_result(self, point, summary):
        self._unimplemented()
