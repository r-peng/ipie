"""Additional contract checks for channel records and numerical diagnostics."""

import json
import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky import (
    Channel, TensorData, ValidationOptions, build_gauge_state, prepare_problem,
    validate_inputs,
)


class ContractTests(unittest.TestCase):
    def test_requested_channel_fields_and_all_spin_orientations(self):
        for spins in ((0,0), (0,1), (1,0), (1,1)):
            channel = Channel(2, spins, 5.0, (0,1), (0.2,-0.3))
            self.assertEqual((channel.bix, channel.spins, channel.a_i, channel.modes, channel.d),
                             (2, spins, 5.0, (0,1), (0.2,-0.3)))
        for changes in ({"a_i": 0.0}, {"a_i": -1.0}, {"d": (np.nan,0)},
                        {"spins": (0,2)}, {"bix": -1}, {"modes": (-1,0)}):
            kwargs = dict(bix=0, spins=(0,1), a_i=2.0, modes=(0,0), d=(0.1,-0.1))
            kwargs.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Channel(**kwargs)

    def test_prepared_arrays_are_owned_readonly_copies(self):
        h = np.eye(2)
        chol = np.array([[[0.2,0.1],[0.1,-0.3]]])
        eri = np.einsum("gpr,gqs->prqs", chol, chol)
        prepared = prepare_problem(h, eri, chol)
        h[0,0] = 7
        chol[0,0,0] = 8
        eri.flat[0] = 9
        np.testing.assert_array_equal(prepared.data.h1, np.eye(2))
        for arr in (prepared.data.h1, prepared.data.eri, prepared.data.chol):
            self.assertFalse(arr.flags.writeable)
        state = build_gauge_state(prepared.data)
        for arr in (state.O, state.chol, state.eigenvalues, state.X):
            self.assertFalse(arr.flags.writeable)

    def test_fortran_eri_and_zero_diagnostics(self):
        L = np.array([[[.2,.1],[.1,-.3]], [[.4,-.2],[-.2,.1]]])
        eri = np.asfortranarray(np.einsum("gpr,gqs->prqs",L,L))
        self.assertFalse(eri.flags.c_contiguous)
        report = validate_inputs(TensorData(np.eye(2),eri,L),ValidationOptions(block_size=1))
        self.assertTrue(report.passed)
        zero = TensorData(np.zeros((1,1)),np.zeros((1,1,1,1)),np.zeros((1,1,1)))
        json.dumps(build_gauge_state(zero).diagnostics.as_dict(), allow_nan=False)

    def test_small_factor_asymmetry_cannot_be_hidden_by_large_factor(self):
        chol = np.array([np.eye(2)*1e6, [[1e-3,1e-5],[0,2e-3]]])
        eri = np.einsum("gpr,gqs->prqs",chol,chol)
        with self.assertRaises(ValueError):
            validate_inputs(TensorData(np.eye(2),eri,chol))


if __name__ == "__main__":
    unittest.main()
