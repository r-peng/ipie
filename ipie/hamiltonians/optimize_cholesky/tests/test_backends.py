"""Public backend selection, compatibility and dependency boundaries."""

from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator, exponential_gauge_vjp, fixed_partition_gradient,
    givens_gauge_vjp, normalized_partition_gradient, relaxed_gauge_gradient,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import make_decomposition


class DerivativeBackendTests(unittest.TestCase):
    def assert_same_evaluation(self, left, right):
        self.assertEqual(left.mode, right.mode)
        self.assertEqual(left.loss, right.loss)
        for name in ("gradient_O", "fixed_gradient_O", "bound_correction_O",
                     "tangent_gradient", "generator_gradient", "partition_gradient",
                     "eigenvalue_gradient", "eigenvector_gradient", "rotated_cholesky_gradient"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)
        np.testing.assert_array_equal(left.decomposition.two_body.amplitudes,
                                      right.decomposition.two_body.amplitudes)

    def test_default_and_explicit_numpy_engines_agree_in_all_three_modes(self):
        problem, state = make_decomposition((2, 0))
        O = state.two_body.gauge.O
        implicit = GaugeEvaluator(problem, tau1=.2, tau2=.04)
        explicit = GaugeEvaluator(problem, tau1=.2, tau2=.04, backend="numpy")
        self.assertEqual(implicit.backend, "numpy")
        self.assertEqual(explicit.backend, "numpy")
        with self.assertRaises(AttributeError):
            explicit.backend = "pytorch"
        for method, kwargs in (("fixed", {"t": .12}),
                               ("fixed_z", {"z": .2, "max_abs_t": .4}),
                               ("relaxed", {"gradient_atol": 1e-12, "gradient_rtol": 0})):
            with self.subTest(method=method):
                self.assert_same_evaluation(getattr(implicit, method)(O, **kwargs),
                                            getattr(explicit, method)(O, **kwargs))

    def test_standalone_numpy_entrypoints_and_coordinate_pullbacks_agree(self):
        problem, state = make_decomposition((2, 0))
        for function in (fixed_partition_gradient, normalized_partition_gradient):
            with self.subTest(function=function.__name__):
                self.assert_same_evaluation(function(problem, state),
                                            function(problem, state, backend="numpy"))
        evaluator = GaugeEvaluator(problem, tau1=.2, tau2=.04, backend="numpy")
        inner = evaluator.relaxed(state.two_body.gauge.O).inner_result
        self.assert_same_evaluation(relaxed_gauge_gradient(problem, inner),
                                    relaxed_gauge_gradient(problem, inner, backend="numpy"))
        E = np.array([[.2, -.3], [.4, .1]])
        reference = np.diag([-1., 1.])
        np.testing.assert_array_equal(
            exponential_gauge_vjp([.2], 2, E, reference),
            exponential_gauge_vjp([.2], 2, E, reference, backend="numpy"))
        rotations = ((0, 1, .2), (0, 1, -.1))
        np.testing.assert_array_equal(
            givens_gauge_vjp(2, rotations, E, reference),
            givens_gauge_vjp(2, rotations, E, reference, backend="numpy"))

    def test_pytorch_engine_fails_before_any_initialization_work(self):
        problem, _ = make_decomposition((2, 0))
        with patch("ipie.hamiltonians.optimize_cholesky.gradients.build_one_body_channels",
                   side_effect=AssertionError("one-body setup must not run")), \
             patch("ipie.hamiltonians.optimize_cholesky.gradients.optimize_partitions",
                   side_effect=AssertionError("inner solver must not run")), \
             patch("numpy.linalg.eigh", side_effect=AssertionError("eigh must not run")):
            with self.assertRaisesRegex(NotImplementedError, "pytorch.*not implemented"):
                GaugeEvaluator(problem, tau1=.2, tau2=.04, backend="pytorch")

    def test_standalone_rejections_are_early_and_unknown_backends_do_not_fall_back(self):
        # Invalid numerical inputs ensure backend selection precedes any work.
        calls = (
            lambda **kw: fixed_partition_gradient(None, None, **kw),
            lambda **kw: normalized_partition_gradient(None, None, **kw),
            lambda **kw: relaxed_gauge_gradient(None, None, **kw),
            lambda **kw: exponential_gauge_vjp(None, 0, None, **kw),
            lambda **kw: givens_gauge_vjp(0, None, None, **kw),
            lambda **kw: GaugeEvaluator(None, tau1=.2, tau2=.04, **kw),
        )
        for call in calls:
            with self.assertRaises(NotImplementedError):
                call(backend="pytorch")
            for name in ("", "torch", "NUMPY", "cupy", "pytorch "):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    call(backend=name)
            for value in (None, False, 1, {}, ["numpy"]):
                with self.subTest(value=value), self.assertRaises(TypeError):
                    call(backend=value)

    def test_package_import_numpy_evaluation_and_stub_require_no_torch(self):
        # Use the real package in a fresh interpreter while blocking all torch
        # imports. Namespace scaffolding bypasses unrelated ipie/MPI startup.
        source_root = Path(__file__).resolve().parents[4]
        script = r'''
import importlib.abc
from pathlib import Path
import sys
import types

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("Unexpected PyTorch import: " + fullname)
        return None

sys.meta_path.insert(0, NoTorch())
root = Path(sys.argv[1])
for name, relative in (("ipie", "ipie"), ("ipie.hamiltonians", "ipie/hamiltonians")):
    module = types.ModuleType(name)
    module.__path__ = [str(root/relative)]
    sys.modules[name] = module
    if "." in name:
        parent, attribute = name.rsplit(".", 1)
        setattr(sys.modules[parent], attribute, module)

from ipie.hamiltonians.optimize_cholesky import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import make_decomposition
problem, state = make_decomposition((2, 0))
evaluator = GaugeEvaluator(problem, tau1=.2, tau2=.04, backend="numpy")
assert evaluator.fixed(state.two_body.gauge.O).loss >= 0
try:
    GaugeEvaluator(problem, tau1=.2, tau2=.04, backend="pytorch")
except NotImplementedError:
    pass
else:
    raise AssertionError("pytorch was silently accepted")
assert "torch" not in sys.modules
print("No torch dependency required")
'''
        result = subprocess.run([sys.executable, "-c", script, str(source_root)],
                                capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("No torch dependency required", result.stdout)


if __name__ == "__main__":
    unittest.main()
