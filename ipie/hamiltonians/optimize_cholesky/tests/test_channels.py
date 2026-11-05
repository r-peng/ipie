"""Small-system Hamiltonian/Slater checks in the style of tests/sor_test.py.

Uses the repository's bitstring signs, interleaved spin orbitals, exterior
minors for propagators, and direct original quartic ERIs for the reference H.
No normal-order correction is subtracted from h in that quartic reference.
Run with unittest, pytest, or `python -m ...tests.test_channels`.
"""

from dataclasses import replace
import itertools
import unittest

import numpy as np

from ipie.hamiltonians.bitstring_utils import get_all_configs_u11, string_act
from ipie.hamiltonians.optimize_cholesky import (
    Channel, build_fixed_decomposition, build_gauge_state, build_one_body_channels,
    build_two_body_channels, enumerate_sources, exponential_gauge, get_rotation_matrix,
    givens_gauge, prepare_problem, two_body_rotation_matrix,
)


def get_occupied_orbitals(cf, norb):
    return [p for p in range(norb) if (cf >> p) & 1]


def embed_spin_rotation(U, nsite):
    U_full = np.eye(2*nsite)
    for spin, block in enumerate(U):
        if block is not None:
            U_full[spin::2, spin::2] = block
    return U_full


def rotation2MB(U, basis, basis_map, nsite):
    """Exterior-power lift; also handles the full Fock basis and vacuum."""
    U_full = embed_spin_rotation(U, nsite)
    occupied = [get_occupied_orbitals(cf, 2*nsite) for cf in basis]
    matrix = np.zeros((len(basis), len(basis)))
    for ix1, occ1 in enumerate(occupied):
        for ix2, occ2 in enumerate(occupied):
            if len(occ1) == len(occ2):
                matrix[ix2, ix1] = np.linalg.det(U_full[np.ix_(occ2, occ1)])
    return matrix


def slater2MB(B, basis):
    n_orb, n_occ = B.shape
    psi = np.zeros(len(basis))
    for ix, cf in enumerate(basis):
        occ = get_occupied_orbitals(cf, n_orb)
        if len(occ) == n_occ:
            psi[ix] = np.linalg.det(B[occ, :])
    return psi


def quadratic2MB(M, basis, basis_map, spin):
    H = np.zeros((len(basis), len(basis)))
    for p, q in itertools.product(range(M.shape[0]), repeat=2):
        if M[p,q] == 0:
            continue
        ops = ((2*p+spin, "cre"), (2*q+spin, "des"))
        for ix1, cf in enumerate(basis):
            cf2, sign = string_act(cf, ops)
            if cf2 is not None:
                H[basis_map[cf2], ix1] += M[p,q] * sign
    return H


def eri2MB(eri, basis, basis_map, spin1, spin2):
    H = np.zeros((len(basis), len(basis)))
    for p, r, q, s in itertools.product(range(eri.shape[0]), repeat=4):
        if eri[p,r,q,s] == 0:
            continue
        ops = ((2*p+spin1, "cre"), (2*q+spin2, "cre"),
               (2*s+spin2, "des"), (2*r+spin1, "des"))
        for ix1, cf in enumerate(basis):
            cf2, sign = string_act(cf, ops)
            if cf2 is not None:
                H[basis_map[cf2], ix1] += eri[p,r,q,s] * sign
    return H


def physical2MB(h1, eri, basis, basis_map):
    """Independent bare-h plus canonical quartic Hamiltonian."""
    H = sum(quadratic2MB(h1, basis, basis_map, spin) for spin in (0,1))
    for s, t in itertools.product((0,1), repeat=2):
        H += 0.5 * eri2MB(eri, basis, basis_map, s, t)
    return H


def get_MB_gf(decomposition, basis, basis_map, nsite):
    """UNNORMALIZED sum a_i U_i; normalize only after combining both lists."""
    G = np.zeros((len(basis), len(basis)))
    for ix, a_i in enumerate(decomposition.coefficients):
        G += a_i * rotation2MB(get_rotation_matrix(decomposition, ix), basis, basis_map, nsite)
    return G


def check_norm(actual, expected, tol=1e-10):
    residual = np.linalg.norm(actual-expected)
    scale = max(1.0, np.linalg.norm(expected))
    if residual > tol*scale:
        raise AssertionError(f"norm mismatch: {residual:.3e} (scale {scale:.3e})")
    return residual / scale


def check_hamiltonian(decomposition, h1, eri, nelecs, *, full_fock=False, seed=11, ecore=0.0):
    """H, shifted G, normalized G, and action on a general Slater determinant."""
    nsite = h1.shape[0]
    basis = list(range(1 << (2*nsite))) if full_fock else get_all_configs_u11((nsite,nsite), nelecs)
    basis_map = {cf: i for i, cf in enumerate(basis)}
    identity = np.eye(len(basis))
    H = physical2MB(h1, eri, basis, basis_map) + ecore*identity
    G = get_MB_gf(decomposition, basis, basis_map, nsite)
    B = decomposition.denom
    err = check_norm((ecore+B)*identity-G, H)
    check_norm(G, (ecore+B)*identity-H)
    check_norm(G/B, identity-(H-ecore*identity)/B)
    rng = np.random.default_rng(seed)
    Bdet = rng.normal(size=(2*nsite, sum(nelecs)))
    psi = slater2MB(Bdet, basis)
    from_updates = np.zeros(len(basis))
    for ix, a_i in enumerate(decomposition.coefficients):
        U = embed_spin_rotation(get_rotation_matrix(decomposition, ix), nsite)
        from_updates += (a_i/B) * slater2MB(U @ Bdet, basis)
    check_norm(from_updates, (G/B) @ psi)
    return err


def example_tensors(n=3, r=2, seed=23):
    rng = np.random.default_rng(seed)
    # Every factor has positive and negative eigenvalues, with different bases.
    chol = []
    for g in range(r):
        X, _ = np.linalg.qr(rng.normal(size=(n,n)))
        ell = np.linspace(-0.45, 0.65, n) + 0.07*g
        chol.append((X*ell) @ X.T)
    chol = np.asarray(chol)
    eri = np.einsum("gpr,gqs->prqs", chol, chol)
    h = rng.normal(size=(n,n))*0.2
    h = (h+h.T)/2
    return h, eri, chol


def make_decomposition(nelec, *, partition="balanced_log", rotated=True, seed=23):
    h, eri, chol = example_tensors(seed=seed)
    problem = prepare_problem(h, eri, chol, nelec=nelec)
    O = exponential_gauge([0.37], 2) if rotated else np.eye(2)
    gauge = build_gauge_state(problem.data, O)
    count = len(enumerate_sources(problem))
    t = np.linspace(-0.23, 0.31, count)
    result = build_fixed_decomposition(problem, gauge, 0.03, 0.04, t, partition=partition)
    return problem, result


class ReconstructionTests(unittest.TestCase):
    def test_three_policies_both_partition_maps_and_gauges(self):
        self.max_error = 0.0
        for nelec in ((2,1), (1,1), (2,0)):
            for partition in ("balanced_log", "anchored_log"):
                for rotated in (False, True):
                    with self.subTest(nelec=nelec, partition=partition, rotated=rotated):
                        problem, result = make_decomposition(nelec, partition=partition, rotated=rotated)
                        err = check_hamiltonian(result, problem.data.h1, problem.data.eri, nelec, ecore=0.37)
                        self.max_error = max(self.max_error, err)

    def test_hybrid_is_exact_on_full_fock_space(self):
        for partition in ("balanced_log", "anchored_log"):
            problem, result = make_decomposition(None, partition=partition)
            check_hamiltonian(result, problem.data.h1, problem.data.eri, (1,1), full_fock=True)

    def test_reduced_policies_are_not_full_fock_identities(self):
        for nelec in ((1,1), (2,0)):
            problem, result = make_decomposition(nelec)
            basis = list(range(64))
            basis_map = {cf:i for i,cf in enumerate(basis)}
            H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
            generated = result.denom*np.eye(64)-get_MB_gf(result, basis, basis_map, 3)
            self.assertGreater(np.linalg.norm(generated-H), 0.1)

    def test_counts_weights_products_and_shared_branches(self):
        r, n, tau2 = 2, 3, 0.04
        for nelec, ns, nc, B2 in (
            ((2,1), r*n*(n+1)//2, 2*r*n*n, r*(2*n*n-n)/tau2),
            ((1,1), r*n*n, 2*r*n*n, r*n*n/tau2),
            ((2,0), r*n*(n-1)//2, r*n*(n-1), r*n*(n-1)/(2*tau2)),
        ):
            problem, result = make_decomposition(nelec)
            two = result.two_body
            self.assertEqual(len(two.sources), ns)
            self.assertEqual(len(two.channels), nc)
            self.assertAlmostEqual(two.weight_sum, B2)
            n1 = n if nelec[1] == 0 else 2*n
            self.assertEqual(len(result.one_body.channels), n1)
            self.assertAlmostEqual(result.denom, B2+n1/0.03)
            self.assertTrue(np.all(result.coefficients > 0))
            for source, ds, (start,stop) in zip(two.sources, two.amplitudes, two.source_slices):
                p, q = source.modes
                W = np.prod(two.gauge.eigenvalues[source.bix, [p,q]])
                np.testing.assert_allclose(ds[0]*ds[1], -W*tau2, atol=0, rtol=1e-13)
                rows = two.channels[start:stop]
                if source.kind == "two_branch":
                    self.assertEqual(len(rows), 2)
                    self.assertTrue(all(c.a_i == 1/(2*tau2) for c in rows))
                    np.testing.assert_array_equal(rows[0].d, ds)
                    np.testing.assert_array_equal(rows[1].d, -ds)
                else:
                    self.assertEqual([c.spins for c in rows], [(0,0),(0,1),(1,0),(1,1)])
                    self.assertTrue(all(c.a_i == 1/tau2 for c in rows))
                    for c, sign in zip(rows, (1,-1,-1,1)):
                        np.testing.assert_array_equal(c.d, sign*ds)

    def test_each_channel_matches_independent_fock_density_product(self):
        problem, result = make_decomposition((2,1))
        n = 3
        basis = get_all_configs_u11((n,n), (2,1))
        basis_map = {cf:i for i,cf in enumerate(basis)}
        identity = np.eye(len(basis))
        for channel in result.two_body.channels:
            factors = []
            for spin, mode, d in zip(channel.spins, channel.modes, channel.d):
                x = result.two_body.gauge.X[channel.bix, :, mode]
                factors.append(identity+d*quadratic2MB(np.outer(x,x), basis, basis_map, spin))
            lifted = rotation2MB(two_body_rotation_matrix(channel, result.two_body.gauge),
                                  basis, basis_map, n)
            check_norm(lifted, factors[0] @ factors[1])

    def test_zero_couplings_and_anchored_surviving_leg(self):
        h = np.array([[0.0,0.2],[0.2,0.1]])
        chol = np.array([np.diag([0.,0.6]), np.zeros((2,2))])
        eri = np.einsum("gpr,gqs->prqs",chol,chol)
        for nelec in ((1,0), (1,1), (2,0), (2,1)):
            problem = prepare_problem(h,eri,chol,nelec=nelec)
            gauge = build_gauge_state(problem.data)
            for partition in ("balanced_log", "anchored_log"):
                result = build_fixed_decomposition(problem,gauge,.03,.04,.13,partition=partition)
                check_hamiltonian(result,h,eri,nelec)
                for source, d in zip(result.two_body.sources, result.two_body.amplitudes):
                    vals = gauge.eigenvalues[source.bix, list(source.modes)]
                    if np.prod(vals) == 0 and partition == "balanced_log":
                        np.testing.assert_array_equal(d,[0.,0.])
                if partition == "anchored_log" and nelec == (1,1):
                    self.assertTrue(any((d[0] == 0) != (d[1] == 0) for d in result.two_body.amplitudes))

    def test_empty_two_body_list_and_zero_one_body(self):
        h=np.zeros((1,1));chol=np.zeros((1,1,1));eri=np.zeros((1,1,1,1))
        problem=prepare_problem(h,eri,chol,nelec=(1,0))
        result=build_fixed_decomposition(problem,build_gauge_state(problem.data),.1,.1)
        self.assertEqual(result.two_body.channels, ())
        self.assertEqual(len(result.one_body.channels), 1)
        self.assertGreater(result.denom,0)
        check_hamiltonian(result,h,eri,(1,0))

    def test_one_body_positive_amplitude_above_one_is_allowed(self):
        h=np.array([[-6.]])
        problem=prepare_problem(h,np.zeros((1,1,1,1)),np.zeros((1,1,1)),nelec=(1,0))
        result=build_fixed_decomposition(problem,build_gauge_state(problem.data),.4,.1)
        self.assertGreater(result.one_body.channels[0].d,1)
        check_hamiltonian(result,h,problem.data.eri,(1,0))

    def test_domain_and_policy_failures_are_explicit(self):
        problem, result = make_decomposition((2,1))
        gauge=result.two_body.gauge
        for kwargs in ({"t":1000.}, {"t":np.nan}, {"t":[0.,1.]},
                       {"amplitude_bound":1.1}, {"partition":"unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_two_body_channels(problem,gauge,.04,**kwargs)
        for tau in (0,-.1,np.inf):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                build_two_body_channels(problem,gauge,tau)
        with self.assertRaises(ValueError):
            build_one_body_channels(problem,100.)
        with self.assertRaises(ValueError):
            build_two_body_channels(replace(problem,policy="ab_only"),gauge,.04)

    def test_literal_repeated_mode_rotation_keeps_cross_term(self):
        _, result = make_decomposition((2,1))
        channel=Channel(0,(0,0),1.,(0,0),(.2,.3))
        U=two_body_rotation_matrix(channel,result.two_body.gauge)
        x=result.two_body.gauge.X[0,:,0]
        np.testing.assert_allclose(U[0],np.eye(3)+(.2+.3+.2*.3)*np.outer(x,x),atol=1e-14)
        self.assertIsNone(U[1])


def run_reference_checks():
    """Readable direct-run output, following sor_test.py's demonstration style."""
    for nelec in ((2,1),(1,1),(2,0)):
        for partition in ("balanced_log","anchored_log"):
            problem, result = make_decomposition(nelec,partition=partition)
            error=check_hamiltonian(result,problem.data.h1,problem.data.eri,nelec)
            print(f"{problem.policy:8s} {partition:12s} N={nelec} "
                  f"n1={len(result.one_body.channels):2d} n2={len(result.two_body.channels):2d} "
                  f"B={result.denom:.6f} relative_H_error={error:.3e}; G/Slater checked")


if __name__ == "__main__":
    run_reference_checks()
    unittest.main()


