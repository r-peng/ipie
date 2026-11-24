"""Step 8: QCSOR.make_shift, the one-shot Cholesky shift in sor_base.

With L'_g = L_g - c_g I and N = N_up + N_down, the exact identity on the
(N_up, N_down) sector is

    H = bare(h + (N-1) sum_g c_g L'_g, L' x L') + 1/2 N(N-1) sum_g c_g^2.

make_shift hands that bare form to compute_h1e_shift, which picks the one-body
shift mu for it, and returns the h' form with mu removed:

    h_t = h + (N-1) sum_g c_g L'_g - 1/2 sum_g L'_g^2 - mu,
    E0 += 1/2 N(N-1) sum_g c_g^2 + mu.N,

so that H = sum_s h_t,s n_s + 1/2 sum_g rho'_g^2 + E0 on the sector.
Both forms are checked in Fock space.
"""

import itertools
import unittest
from unittest.mock import patch

import numpy as np

from ipie.hamiltonians.bitstring_utils import get_all_configs_u11
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_norm,
    eri2MB,
    physical2MB,
    quadratic2MB,
)
from ipie.hamiltonians.sor_base import QCSOR, compute_cholesky_shift, compute_h1e_shift

H1E_SHIFT = "ipie.hamiltonians.sor_base.compute_h1e_shift"


def tensors(n=4, r=3, seed=5):
    """(h, chol) with nonzero medians, so every factor is actually shifted."""
    rng = np.random.default_rng(seed)
    chol = []
    for g in range(r):
        X, _ = np.linalg.qr(rng.normal(size=(n, n)))
        chol.append((X*(.3*rng.normal(size=n) + (-1)**g*.4*(g + 1))) @ X.T)
    A = rng.normal(size=(n, n))
    return .3*(A + A.T), np.array(chol)


def sector(n, nelec):
    basis = get_all_configs_u11((n, n), nelec)
    return basis, {cf: i for i, cf in enumerate(basis)}


class MakeShiftTests(unittest.TestCase):
    SECTORS = ((2, 1), (1, 1), (2, 2))

    def run_make_shift(self, h, chol, nelec):
        """make_shift, plus copies of the arguments compute_h1e_shift received.

        Copies are taken at call time: make_shift updates its h_t in place afterwards."""
        ham = QCSOR(h.shape[0])
        received = []

        def spy(*args):
            received.append(tuple(np.array(x, copy=True) if isinstance(x, np.ndarray) else x for x in args))
            return compute_h1e_shift(*args)

        with patch(H1E_SHIFT, side_effect=spy):
            h_t, Lp = ham.make_shift(h.copy(), chol.copy(), nelec, .03, .04)
        self.assertEqual(len(received), 1)
        return ham, h_t, Lp, received[0]

    def test_factors_are_shifted_by_their_medians(self):
        h, chol = tensors()
        _, c = compute_cholesky_shift(chol)
        self.assertTrue(np.all(np.abs(c) > .1))
        _, _, Lp, _ = self.run_make_shift(h, chol, (2, 1))
        np.testing.assert_allclose(Lp, chol - c[:, None, None]*np.eye(h.shape[0]), atol=1e-15)
        np.testing.assert_allclose(np.median(np.linalg.eigvalsh(Lp), axis=1), 0., atol=1e-14)

    def test_mu_is_chosen_for_the_shifted_hamiltonian(self):
        """compute_h1e_shift receives the bare form of H for L' (N-1, not N)."""
        h, chol = tensors()
        n = h.shape[0]
        _, c = compute_cholesky_shift(chol)
        for nelec in self.SECTORS:
            with self.subTest(nelec=nelec):
                _, _, Lp, (h_bare, eri_bare, chol_bare, nelec_bare, dt1, dt2) = self.run_make_shift(h, chol, nelec)
                N = sum(nelec)
                np.testing.assert_array_equal(chol_bare, Lp)
                np.testing.assert_allclose(eri_bare, np.einsum("gpr,gqs->prqs", Lp, Lp), atol=1e-15)
                self.assertEqual((tuple(nelec_bare), dt1, dt2), (nelec, .03, .04))
                basis, basis_map = sector(n, nelec)
                H = physical2MB(h, np.einsum("gpr,gqs->prqs", chol, chol), basis, basis_map)
                const = .5*N*(N - 1)*np.sum(c**2)
                check_norm(physical2MB(h_bare, eri_bare, basis, basis_map) + const*np.eye(len(basis)), H)

    def test_returned_h_and_E0_reproduce_H(self):
        """H = sum_s h_t,s n_s + 1/2 sum_g rho'_g^2 + E0 on the sector (h' form, mu removed)."""
        h, chol = tensors()
        n = h.shape[0]
        for nelec in self.SECTORS:
            with self.subTest(nelec=nelec):
                ham, h_t, Lp, _ = self.run_make_shift(h, chol, nelec)
                h_spin = h_t if isinstance(h_t, tuple) else (h_t, h_t)
                basis, basis_map = sector(n, nelec)
                H = physical2MB(h, np.einsum("gpr,gqs->prqs", chol, chol), basis, basis_map)
                # 1/2 sum_g rho'_g^2 = one-body 1/2 sum_g L'_g^2 + the normal-ordered two-body part.
                half_square = .5*np.einsum("gpr,grs->ps", Lp, Lp)
                rebuilt = sum(quadratic2MB(hs + half_square, basis, basis_map, s) for s, hs in enumerate(h_spin))
                eri_p = np.einsum("gpr,gqs->prqs", Lp, Lp)
                for s, t in itertools.product((0, 1), repeat=2):
                    rebuilt = rebuilt + .5*eri2MB(eri_p, basis, basis_map, s, t)
                check_norm(rebuilt + float(ham.E0)*np.eye(len(basis)), H)

    def test_E0_is_added_to_the_existing_value(self):
        h, chol = tensors()
        first, *_ = self.run_make_shift(h, chol, (2, 1))
        ham = QCSOR(h.shape[0])
        ham.E0 = 1.25
        ham.make_shift(h.copy(), chol.copy(), (2, 1), .03, .04)
        self.assertAlmostEqual(float(ham.E0), 1.25 + float(first.E0), places=13)


if __name__ == "__main__":
    unittest.main()
