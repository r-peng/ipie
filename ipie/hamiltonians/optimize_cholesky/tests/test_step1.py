"""Independent reference checks for input preparation and exact Cholesky gauges.

The ERI reference uses explicit ``eri[p,r,q,s]`` contractions.  Eigenvectors
are checked by reconstruction, so degeneracies do not impose an arbitrary
choice of eigenbasis on the implementation.
"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    import h5py
except ImportError:
    h5py = None
import numpy as np

from ipie.hamiltonians.optimize_cholesky.gauge import (
    build_gauge_state,
    diagonalize_factors,
    exponential_gauge,
    givens_gauge,
    rotate_cholesky,
)
from ipie.hamiltonians.optimize_cholesky.models import TensorData, ValidationOptions
from ipie.hamiltonians.optimize_cholesky.prepare import load_problem, prepare_problem
from ipie.hamiltonians.optimize_cholesky.validation import validate_gauge_state, validate_inputs


def reference_eri(chol):
    return np.einsum("gpr,gqs->prqs", chol, chol)


def reference_square_sum(chol):
    return np.einsum("gpr,grs->ps", chol, chol)


def example_tensors():
    h1 = np.array([[-1.2, 0.1, -0.2], [0.1, 0.3, 0.15], [-0.2, 0.15, 0.8]])
    chol = np.array(
        [
            [[0.6, 0.2, 0.0], [0.2, -0.3, 0.1], [0.0, 0.1, 0.2]],
            [[-0.1, 0.4, 0.2], [0.4, 0.3, -0.2], [0.2, -0.2, 0.5]],
            [[0.2, -0.15, 0.1], [-0.15, 0.4, 0.25], [0.1, 0.25, -0.2]],
        ],
        dtype=np.float64,
    )
    return h1, reference_eri(chol), chol


class TensorValidationTests(unittest.TestCase):
    def setUp(self):
        self.h1, self.eri, self.chol = example_tensors()
        self.options = ValidationOptions(atol=1e-11, rtol=1e-10, block_size=2)

    def data(self, **changes):
        values = {"h1": self.h1, "eri": self.eri, "chol": self.chol}
        values.update(changes)
        return TensorData(**values)

    def test_valid_noncommuting_factors(self):
        commutator = self.chol[0] @ self.chol[1] - self.chol[1] @ self.chol[0]
        self.assertGreater(np.linalg.norm(commutator), 0.1)
        self.assertIsNotNone(validate_inputs(self.data(), self.options))

    def test_zero_rank_deficient_and_degenerate_factors(self):
        repeated = np.diag([0.7, 0.7, -0.2])
        scalar = 0.3 * np.eye(3)
        chol = np.stack([np.zeros((3, 3)), scalar, repeated, repeated - scalar])
        data = self.data(chol=chol, eri=reference_eri(chol))
        self.assertIsNotNone(validate_inputs(data, self.options))
        eigenvalues, vectors = diagonalize_factors(chol)
        self.assertEqual(eigenvalues.shape, (4, 3))
        self.assertEqual(vectors.shape, (4, 3, 3))
        reconstructed = np.einsum("gpa,ga,gra->gpr", vectors, eigenvalues, vectors)
        np.testing.assert_allclose(reconstructed, chol, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            np.einsum("gpa,gpb->gab", vectors, vectors),
            np.broadcast_to(np.eye(3), (4, 3, 3)),
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertIsNotNone(validate_gauge_state(data, build_gauge_state(data), self.options))

    def test_identically_zero_interaction_is_valid(self):
        chol = np.zeros_like(self.chol)
        data = self.data(chol=chol, eri=reference_eri(chol))
        self.assertIsNotNone(validate_inputs(data, self.options))
        state = build_gauge_state(data, exponential_gauge(np.array([0.2, -0.3, 0.4]), 3))
        np.testing.assert_array_equal(state.chol, chol)
        np.testing.assert_array_equal(state.eigenvalues, np.zeros((3, 3)))
        self.assertIsNotNone(validate_gauge_state(data, state, self.options))

    def test_malformed_shapes_are_rejected(self):
        malformed = [
            {"h1": np.zeros((3, 2))},
            {"h1": self.h1[None, :, :]},
            {"eri": np.zeros((3, 3, 3))},
            {"eri": np.zeros((3, 3, 3, 2))},
            {"chol": self.chol[0]},
            {"chol": np.zeros((3, 3, 2))},
            {"h1": np.zeros((0, 0)), "eri": np.zeros((0, 0, 0, 0)), "chol": np.zeros((1, 0, 0))},
        ]
        for changes in malformed:
            with self.subTest(shapes={key: value.shape for key, value in changes.items()}):
                with self.assertRaises(ValueError):
                    validate_inputs(self.data(**changes), self.options)

    def test_nonfinite_tensors_are_rejected(self):
        for name, original in [("h1", self.h1), ("eri", self.eri), ("chol", self.chol)]:
            for bad in [np.nan, np.inf, -np.inf]:
                modified = original.copy()
                modified.flat[0] = bad
                with self.subTest(tensor=name, value=bad):
                    with self.assertRaises(ValueError):
                        validate_inputs(self.data(**{name: modified}), self.options)

    def test_nonreal_tensors_are_rejected(self):
        for name, original in [("h1", self.h1), ("eri", self.eri), ("chol", self.chol)]:
            modified = original.astype(np.complex128)
            modified.flat[0] += 0.1j
            with self.subTest(tensor=name):
                with self.assertRaises((ValueError, TypeError)):
                    validate_inputs(self.data(**{name: modified}), self.options)

    def test_nonsymmetric_tensors_are_rejected(self):
        changes = []
        h1 = self.h1.copy()
        h1[0, 1] += 0.1
        changes.append({"h1": h1})
        chol = self.chol.copy()
        chol[0, 0, 1] += 0.1
        changes.append({"chol": chol, "eri": reference_eri(chol)})
        eri = self.eri.copy()
        eri[0, 1, 2, 2] += 0.1
        changes.append({"eri": eri})
        for change in changes:
            with self.subTest(tensors=list(change)):
                with self.assertRaises(ValueError):
                    validate_inputs(self.data(**change), self.options)

    def test_symmetric_but_inexact_eri_is_rejected(self):
        extra = np.diag([0.2, -0.1, 0.3])[None, :, :]
        wrong_eri = self.eri + reference_eri(extra)
        with self.assertRaises(ValueError):
            validate_inputs(self.data(eri=wrong_eri), self.options)

    def test_wrong_eri_axis_order_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_inputs(self.data(eri=self.eri.transpose(0, 2, 1, 3)), self.options)


class GaugeTests(unittest.TestCase):
    def setUp(self):
        h1, eri, chol = example_tensors()
        self.data = TensorData(h1=h1, eri=eri, chol=chol)
        self.chol = chol
        self.options = ValidationOptions(atol=1e-11, rtol=1e-10, block_size=2)

    def assert_invariants(self, O):
        np.testing.assert_allclose(O.T @ O, np.eye(3), atol=1e-12, rtol=1e-12)
        rotated = rotate_cholesky(self.chol, O)
        np.testing.assert_allclose(rotated, np.einsum("eg,gpr->epr", O, self.chol), atol=1e-12)
        np.testing.assert_allclose(reference_eri(rotated), self.data.eri, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(
            reference_square_sum(rotated), reference_square_sum(self.chol), atol=1e-12, rtol=1e-12
        )
        state = build_gauge_state(self.data, O, self.options)
        np.testing.assert_allclose(state.O, O, atol=1e-12)
        np.testing.assert_allclose(state.chol, rotated, atol=1e-12)
        reconstructed = np.einsum("gpa,ga,gra->gpr", state.X, state.eigenvalues, state.X)
        np.testing.assert_allclose(reconstructed, rotated, atol=1e-12, rtol=1e-12)
        self.assertIsNotNone(validate_gauge_state(self.data, state, self.options))

    def test_initial_state_is_identity_gauge(self):
        state = build_gauge_state(self.data)
        np.testing.assert_allclose(state.O, np.eye(3), atol=1e-14)
        np.testing.assert_allclose(state.chol, self.chol, atol=1e-14)

    def test_single_factor_single_orbital_has_no_rotation_parameters(self):
        chol = np.array([[[0.4]]])
        data = TensorData(h1=np.array([[-0.3]]), eri=reference_eri(chol), chol=chol)
        for O in [exponential_gauge(np.array([]), 1), givens_gauge(1, [])]:
            np.testing.assert_array_equal(O, np.eye(1))
            state = build_gauge_state(data, O)
            np.testing.assert_allclose(state.chol, chol, atol=1e-14)
            self.assertIsNotNone(validate_gauge_state(data, state))

    def test_dense_exponential_gauge(self):
        O = exponential_gauge(np.array([0.31, -0.47, 0.29]), 3)
        self.assertGreater(np.count_nonzero(np.abs(O) > 1e-4), 6)
        self.assertAlmostEqual(np.linalg.det(O), 1.0, places=12)
        self.assert_invariants(O)

    def test_givens_sequence(self):
        O = givens_gauge(3, [(0, 1, 0.23), (1, 2, -0.51), (0, 2, 0.37)])
        self.assertAlmostEqual(np.linalg.det(O), 1.0, places=12)
        self.assert_invariants(O)

    def test_both_maps_support_reflection_reference(self):
        reference = np.diag([-1.0, 1.0, 1.0])
        gauges = [
            exponential_gauge(np.array([0.2, -0.1, 0.3]), 3, reference=reference),
            givens_gauge(3, [(0, 1, 0.3), (1, 2, -0.2)], reference=reference),
        ]
        for O in gauges:
            self.assertAlmostEqual(np.linalg.det(O), -1.0, places=12)
            self.assert_invariants(O)
        np.testing.assert_allclose(exponential_gauge(np.zeros(3), 3, reference=reference), reference)
        np.testing.assert_allclose(givens_gauge(3, [], reference=reference), reference)

    def test_composition_matches_sequential_factor_rotation(self):
        first = exponential_gauge(np.array([0.31, -0.2, 0.07]), 3)
        second = givens_gauge(3, [(0, 2, -0.3), (0, 1, 0.5)])
        sequential = rotate_cholesky(rotate_cholesky(self.chol, first), second)
        np.testing.assert_allclose(sequential, rotate_cholesky(self.chol, second @ first), atol=1e-12)
        self.assert_invariants(second @ first)

    def test_nonorthogonal_or_wrong_shape_gauge_is_rejected(self):
        bad_gauges = [np.diag([1.0, 1.0, 1.02]), np.eye(2), np.full((3, 3), np.nan)]
        for O in bad_gauges:
            with self.subTest(gauge=O):
                with self.assertRaises(ValueError):
                    build_gauge_state(self.data, O, self.options)

    def test_state_validation_rejects_tampered_fields(self):
        state = build_gauge_state(self.data)
        for name, original in [
            ("O", state.O),
            ("chol", state.chol),
            ("eigenvalues", state.eigenvalues),
            ("X", state.X),
        ]:
            modified = original.copy()
            modified.flat[0] += 0.03
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    validate_gauge_state(self.data, replace(state, **{name: modified}), self.options)


class ProblemPreparationTests(unittest.TestCase):
    def setUp(self):
        self.h1, self.eri, self.chol = example_tensors()

    def assert_loaded_tensors(self, prepared):
        np.testing.assert_array_equal(prepared.data.h1, self.h1)
        np.testing.assert_array_equal(prepared.data.eri, self.eri)
        np.testing.assert_array_equal(prepared.data.chol, self.chol)
        self.assertIsNotNone(prepared.report)

    def test_prepare_preserves_original_physical_h1(self):
        prepared = prepare_problem(self.h1, self.eri, self.chol, nelec=(2, 1))
        self.assert_loaded_tensors(prepared)
        np.testing.assert_array_equal(prepared.nelec, (2, 1))
        # This catches accidental conversion of the physical h into h'.
        self.assertGreater(np.linalg.norm(reference_square_sum(self.chol)), 0.1)

    def test_policy_selection_uses_conserved_particle_sector(self):
        cases = [(None, "hybrid"), ((2, 1), "hybrid"), ((1, 1), "ab_only"), ((2, 0), "aa_only")]
        for nelec, policy in cases:
            with self.subTest(nelec=nelec):
                prepared = prepare_problem(self.h1, self.eri, self.chol, nelec=nelec)
                self.assertEqual(prepared.policy, policy)

    def test_invalid_particle_counts_are_rejected(self):
        for nelec in [(-1, 1), (4, 0), (1,), (1, 1, 0), (1.5, 1)]:
            with self.subTest(nelec=nelec):
                with self.assertRaises(ValueError):
                    prepare_problem(self.h1, self.eri, self.chol, nelec=nelec)

    @unittest.skipIf(h5py is None, "Install h5py to run the real HDF5 round-trip tests")
    def test_same_hdf5_file_default_dataset_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "integrals.h5"
            with h5py.File(path, "w") as handle:
                handle["h1"] = self.h1
                handle["eri"] = self.eri
                handle["chol"] = self.chol
            prepared = load_problem(path, nelec=(1, 1))
            self.assert_loaded_tensors(prepared)
            np.testing.assert_array_equal(prepared.nelec, (1, 1))

    @unittest.skipIf(h5py is None, "Install h5py to run the real HDF5 round-trip tests")
    def test_separate_hdf5_files_with_custom_dataset_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ["one.h5", "two.h5", "factors.h5"]]
            keys = ["integrals/physical_h", "interaction/prqs", "factorization/L"]
            for path, key, tensor in zip(paths, keys, [self.h1, self.eri, self.chol]):
                with h5py.File(path, "w") as handle:
                    handle[key] = tensor
            prepared = load_problem(
                *paths,
                h1_key=keys[0],
                eri_key=keys[1],
                chol_key=keys[2],
                nelec=(2, 0),
                options=ValidationOptions(block_size=2),
            )
            self.assert_loaded_tensors(prepared)
            np.testing.assert_array_equal(prepared.nelec, (2, 0))

    @unittest.skipIf(h5py is None, "Install h5py to run the real HDF5 round-trip tests")
    def test_loader_rejects_inexact_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.h5"
            with h5py.File(path, "w") as handle:
                handle["h1"] = self.h1
                handle["eri"] = self.eri * 1.1
                handle["chol"] = self.chol
            with self.assertRaises(ValueError):
                load_problem(path)


if __name__ == "__main__":
    unittest.main()
