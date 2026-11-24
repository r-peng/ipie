"""Step 12: saving Cholesky-shifted decompositions (format version 3) and loading them with QCSOR.

A shifted decomposition is stored with its shifted channels, per-spin one-body
bases (the eigenbases of h_s = h + sum_g (m_g - mu[g,s]) L_g), the shift, the
shifted channel eigenvalues chol_lam and E_c. The energy integrals stay the
physical ones (h1 = h - 1/2 sum L^2 per spin, v0, W = ell ell^T), so ham.E0 = 0
and the walker local energy is the exact one. On the stored sector

    sum_i a_i U_i = Lambda I - H,  Lambda = ham.denom + mu.N + ham.cholesky_energy.

The tests check:
  * the file attributes and the QCSOR against a per-channel reference;
  * exact reconstruction of the sector Hamiltonian, and that a shifted file
    loads only on its sector;
  * the walker local energy against the exact mixed estimator, and same-seed
    propagation against the reference;
  * truncation by each channel's own |lambda lambda|;
  * version 2 files still load, and the reader's checks;
  * QCSOR.optimize_from_cholesky's cholesky_shift and optimize_cholesky_shift.
"""

from contextlib import ExitStack
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ipie.hamiltonians.optimize_cholesky.channels import with_channels
from ipie.hamiltonians.optimize_cholesky.optimizers import median_cholesky_shift
from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, prepare_problem
from ipie.hamiltonians.optimize_cholesky.shift import with_shift
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import check_norm, example_tensors, physical2MB, slater2MB
from ipie.hamiltonians.optimize_cholesky.tests.test_cholesky_shift import MAPS, build, fixture, shift_of
from ipie.hamiltonians.optimize_cholesky.tests.test_io_sor import (
    HAVE_H5PY,
    HAVE_SOR,
    HAVE_WALKERS,
    assert_same_qcsor,
    assert_same_terms_per_key,
    file_attrs,
    random_walkers,
    reference_qcsor,
    sector_basis,
    sector_lambda,
    sor_green_function,
    threshold_between,
    uhf_spin_orbitals,
)
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import assert_close

if HAVE_H5PY:
    import h5py
    from ipie.hamiltonians.optimize_cholesky.io import FORMAT, FORMAT_VERSION, save_decomposition
if HAVE_SOR:
    from ipie.hamiltonians.sor_base import QCSOR
if HAVE_WALKERS:
    from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
    from ipie.walkers.lafqmc_uhf_walkers import UHFWalkers

SECTORS = ((2, 1), (1, 1), (2, 0), (2, 2))
KINDS = ("tied", "split")


def cases(partitions=MAPS):
    """(label, problem, decomposition) for every sector, map and shift kind."""
    for nelec in SECTORS:
        problem, gauge = fixture(nelec)
        for partition in partitions:
            for kind in KINDS:
                yield f"{nelec} {partition[:3]} {kind}", problem, build(problem, gauge, shift_of(kind, problem),
                                                                     partition)


def channel_values(decomposition):
    """|lambda_left lambda_right| of every two-body channel (its own leg spins), |eps - mu_s| of every one-body one."""
    one, two = decomposition.one_body, decomposition.two_body
    lam, tt, ot = two.spin_eigenvalues, two.table, one.table
    W = np.abs(lam[tt.spins[:, 0], tt.bix, tt.modes[:, 0]]*lam[tt.spins[:, 1], tt.bix, tt.modes[:, 1]])
    eps = np.abs(one.mode_eigenvalues[ot.spin, ot.mode] - np.asarray(one.shift)[ot.spin])
    return eps, W


def split_shifted(decomposition, thresh):
    """(kept, dropped) under from_file's rule, selected from the decomposition's own channels."""
    eps, W = channel_values(decomposition)
    one, two = decomposition.one_body.channels, decomposition.two_body.channels
    pick = lambda keep1, keep2: with_channels(decomposition, one_body=[c for c, k in zip(one, keep1) if k],
                                              two_body=[c for c, k in zip(two, keep2) if k])
    return pick(eps >= thresh, W >= thresh), pick(eps < thresh, W < thresh)


@unittest.skipUnless(HAVE_H5PY and HAVE_SOR, "h5py and ipie.hamiltonians.sor_base are required")
class ShiftedFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def saved(self, source, name="shifted.h5", **kwargs):
        return save_decomposition(os.path.join(self.tmp.name, name), source, overwrite=True, **kwargs)

    def test_attributes_and_physical_integrals(self):
        for label, problem, decomposition in cases():
            with self.subTest(case=label):
                path = self.saved(decomposition)
                attrs = file_attrs(path)
                self.assertEqual((attrs["format"], int(attrs["format_version"])), (FORMAT, FORMAT_VERSION))
                self.assertTrue(bool(attrs["has_cholesky_shift"]))
                np.testing.assert_array_equal(attrs["cholesky_shift"], decomposition.cholesky_shift)
                self.assertEqual(float(attrs["cholesky_energy"]), decomposition.cholesky_energy)
                assert_close(float(attrs["lambda_electronic"]), decomposition.lambda_electronic)
                with h5py.File(path, "r") as f:
                    np.testing.assert_array_equal(f["chol_lam"][()], decomposition.two_body.spin_eigenvalues)
                    np.testing.assert_array_equal(f["chol_ell"][()], decomposition.two_body.gauge.eigenvalues)
                    k = f["h1_X"].shape[0]
                one = decomposition.one_body
                shared = len(one.spins) == 1 or (not one.split and one.shift[0] == one.shift[1])
                self.assertEqual(k, 1 if shared else 2)
                ham = QCSOR.from_file(path, thresh=None)
                np.testing.assert_array_equal(ham.cholesky_shift, decomposition.cholesky_shift)
                self.assertEqual(ham.cholesky_energy, decomposition.cholesky_energy)
                self.assertEqual(float(ham.E0), 0.)
                h, chol = problem.data.h1, problem.data.chol
                L2 = np.einsum("gpr,grq->pq", chol, chol)
                for spin in (0, 1):
                    assert_close(np.asarray(ham.h1)[spin], h - .5*L2, label="h1")
                    assert_close(np.asarray(ham.h1)[spin] + ham.v0[spin], h, label="h1 + v0")

    def test_file_matches_the_per_channel_reference(self):
        for label, problem, decomposition in cases():
            for drop in (False, True):
                with self.subTest(case=label, drop_identity=drop):
                    ham = QCSOR.from_file(self.saved(decomposition, drop_identity=drop), thresh=None)
                    reference = reference_qcsor(decomposition, drop_identity=drop, h=problem.data.h1)
                    assert_same_qcsor(self, reference, ham)

    def test_sector_hamiltonian_is_reconstructed_exactly(self):
        for label, problem, decomposition in cases():
            with self.subTest(case=label):
                nelec = problem.nelec
                ham = QCSOR.from_file(self.saved(decomposition), thresh=None)
                n = problem.data.nbasis
                basis, basis_map = sector_basis(n, nelec)
                H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
                Lam = sector_lambda(ham, ham.mu, nelec)
                assert_close(Lam, float(file_attrs(self.saved(decomposition))["lambda_electronic"]))
                check_norm(Lam*np.eye(len(basis)) - sor_green_function(ham, basis, basis_map, n), H)

    def test_a_shifted_file_loads_only_on_its_sector(self):
        problem, gauge = fixture((2, 1))
        path = self.saved(build(problem, gauge, shift_of("tied", problem)))
        QCSOR.from_file(path, nelec=(2, 1))
        with self.assertRaisesRegex(ValueError, "only on the sector"):
            QCSOR.from_file(path, nelec=(1, 2))
        QCSOR.from_file(path, nelec=(1, 2), parse=False)        # unparsed: no sector yet
        unshifted = self.saved(build(problem, gauge, None), name="plain.h5")
        QCSOR.from_file(unshifted, nelec=(1, 2))                 # hybrid without a shift: any sector

    def test_truncation_uses_each_channel_s_shifted_magnitude(self):
        for label, problem, decomposition in cases(("balanced_log",)):
            path = self.saved(decomposition)
            eps, W = channel_values(decomposition)
            for fraction in (.3, .6):
                thresh = threshold_between(np.concatenate([eps, W]), fraction)
                with self.subTest(case=label, thresh=thresh):
                    kept, dropped = split_shifted(decomposition, thresh)
                    self.assertGreater(dropped.nterms, 0)
                    ham = QCSOR.from_file(path, thresh=thresh)
                    assert_same_terms_per_key(self, reference_qcsor(kept, h=problem.data.h1), ham)
                    np.testing.assert_allclose(float(ham.denom), decomposition.denom - dropped.denom,
                                               rtol=1e-13, atol=0)
                    # Lambda_kept I - sum_kept a U = H - sum_dropped a (I - U) on the sector.
                    n, nelec = problem.data.nbasis, problem.nelec
                    basis, basis_map = sector_basis(n, nelec)
                    H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
                    removed = reference_qcsor(dropped, h=problem.data.h1)
                    H = H - (float(removed.denom)*np.eye(len(basis)) - sor_green_function(removed, basis, basis_map, n))
                    check_norm(sector_lambda(ham, ham.mu, nelec)*np.eye(len(basis))
                               - sor_green_function(ham, basis, basis_map, n), H)


@unittest.skipUnless(HAVE_H5PY and HAVE_SOR and HAVE_WALKERS, "h5py, sor_base and the LAFQMC walkers are required")
class ShiftedWalkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_local_energy_matches_the_exact_mixed_estimator(self):
        rng = np.random.default_rng(41)
        for label, problem, decomposition in cases(("balanced_log",)):
            if problem.policy == "aa_only":
                continue                                   # the UHF walker needs both spins occupied
            with self.subTest(case=label):
                path = save_decomposition(os.path.join(self.tmp.name, "w.h5"), decomposition, overwrite=True)
                ham = QCSOR.from_file(path, thresh=None)
                h, eri, nelec = problem.data.h1, problem.data.eri, problem.nelec
                n, (na, nb), nw = h.shape[0], nelec, 3
                psi, phi = random_walkers(rng, n, nelec, nw)
                trial = SingleDet(psi, nelec, n)
                trial.build(ham)
                walkers = UHFWalkers(phi[0], na, nb, n, nw, None)
                walkers.phi = phi.copy()
                walkers.compute_S(trial, set_buff=True)
                eloc, _, _ = walkers.local_energy(ham, trial)
                basis, basis_map = sector_basis(n, nelec)
                H = physical2MB(h, eri, basis, basis_map)
                left = slater2MB(uhf_spin_orbitals(psi, na, n), basis)
                for w in range(nw):
                    right = slater2MB(uhf_spin_orbitals(phi[w], na, n), basis)
                    np.testing.assert_allclose(float(np.real(eloc[w])), (left @ H @ right)/(left @ right),
                                               rtol=1e-10, atol=1e-12)

    def test_same_seed_propagation_matches_the_reference(self):
        for kind in KINDS:
            with self.subTest(shift=kind):
                problem, gauge = fixture((2, 1))
                decomposition = build(problem, gauge, shift_of(kind, problem))
                n, nelec, nw = problem.data.nbasis, (2, 1), 6
                psi, phi0 = random_walkers(np.random.default_rng(1), n, nelec, nw)
                path = save_decomposition(os.path.join(self.tmp.name, "p.h5"), decomposition, overwrite=True)
                out = []
                for ham in (reference_qcsor(decomposition, h=problem.data.h1), QCSOR.from_file(path, thresh=None)):
                    trial = SingleDet(psi, nelec, n)
                    trial.build(ham)
                    walkers = UHFWalkers(phi0[0], *nelec, n, nw, None)
                    walkers.phi = phi0.copy()
                    walkers.compute_S(trial, set_buff=True)
                    sampler, b = np.random.default_rng(7), np.ones(nw)
                    for _ in range(20):
                        ham.parse_samples(sampler.choice(ham.nterms, size=nw, p=np.asarray(ham.a)))
                        b = walkers.update_walkers(ham, trial, b=b)
                    out.append((walkers.phi.copy(), b.copy(), walkers.local_energy(ham, trial)[0]))
                for x, y in zip(*out):
                    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-14)


@unittest.skipUnless(HAVE_H5PY and HAVE_SOR, "h5py and ipie.hamiltonians.sor_base are required")
class VersionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def as_version_2(path):
        """Rewrite an unshifted version 3 file in the version 2 layout: one (n,n) h1_X, no shift attributes."""
        with h5py.File(path, "r+") as f:
            X = f["h1_X"][()]
            assert np.all(X == X[0]) and not bool(f.attrs["has_cholesky_shift"])
            del f["h1_X"]
            f.create_dataset("h1_X", data=X[0])
            for name in ("has_cholesky_shift", "cholesky_shift", "cholesky_energy"):
                del f.attrs[name]
            f.attrs["format_version"] = 2

    def test_version_2_files_load(self):
        for nelec, shift in (((2, 1), (.05, -.03)), ((2, 1), (.05, .05)), ((1, 1), (.1, .02)), ((2, 0), (.04, 0.))):
            with self.subTest(nelec=nelec, shift=shift):
                problem, gauge = fixture(nelec)
                decomposition = with_shift(build(problem, gauge, None), shift)
                v3 = save_decomposition(os.path.join(self.tmp.name, "v3.h5"), decomposition, overwrite=True)
                v2 = save_decomposition(os.path.join(self.tmp.name, "v2.h5"), decomposition, overwrite=True)
                self.as_version_2(v2)
                a, b = QCSOR.from_file(v3, thresh=None), QCSOR.from_file(v2, thresh=None)
                assert_same_qcsor(self, a, b)
                self.assertIsNone(b.cholesky_shift)
                self.assertEqual(b.cholesky_energy, 0.)
                self.assertEqual(np.shape(b.h1), (2, problem.data.nbasis, problem.data.nbasis))

    def test_reader_checks(self):
        problem, gauge = fixture((2, 1))
        path = save_decomposition(os.path.join(self.tmp.name, "bad.h5"),
                                  build(problem, gauge, shift_of("split", problem)), overwrite=True)
        with h5py.File(path, "r+") as f:
            lam = f["chol_lam"][()]
            del f["chol_lam"]
            f.create_dataset("chol_lam", data=lam[:1])
        with self.assertRaisesRegex(ValueError, "Cholesky shift arrays"):
            QCSOR.from_file(path)
        with h5py.File(path, "r+") as f:
            f.attrs["format_version"] = 1
        with self.assertRaisesRegex(ValueError, "format_version"):
            QCSOR.from_file(path)


OPTIMIZE = "ipie.hamiltonians.optimize_cholesky.optimizers.optimize_gauge"
SAVE = "ipie.hamiltonians.optimize_cholesky.io.save_decomposition"
PREPARE = "ipie.hamiltonians.optimize_cholesky.problem.prepare_problem"


@unittest.skipUnless(HAVE_SOR, "ipie.hamiltonians.sor_base is required")
class WrapperTests(unittest.TestCase):
    @staticmethod
    def captured(nelec=(2, 1), **kwargs):
        h, _, chol = example_tensors(n=3, r=2)
        with ExitStack() as stack:
            optimize = stack.enter_context(patch(OPTIMIZE, return_value="result"))
            stack.enter_context(patch(SAVE))
            QCSOR.optimize_from_cholesky(h, chol, nelec, .03, .04, "unused.h5", **kwargs)
        return optimize.call_args.kwargs

    def test_options_reach_optimize_gauge(self):
        default = self.captured()
        self.assertEqual((default["optimize_cholesky_shift"], default["cholesky_shift0"]), ("tied", None))
        mu = np.array([.1, -.2])
        for kwargs in (dict(cholesky_shift="median", optimize_cholesky_shift=False), dict(optimize_cholesky_shift=False),
                       dict(cholesky_shift=mu, optimize_cholesky_shift="split"), dict(cholesky_shift="zero")):
            with self.subTest(kwargs=kwargs):
                call = self.captured(**kwargs)
                self.assertEqual(call["optimize_cholesky_shift"], kwargs.get("optimize_cholesky_shift", "tied"))
                self.assertIs(call["cholesky_shift0"], kwargs.get("cholesky_shift"))

    def test_invalid_shift_options_fail_before_numerical_work(self):
        for kwargs, message in ((dict(cholesky_shift="mean"), "'median'"),
                                (dict(cholesky_shift=np.zeros(3)), "cholesky_shift"),
                                (dict(optimize_cholesky_shift="both"), "optimize_cholesky_shift"),
                                (dict(optimize_cholesky_shift=True), "optimize_cholesky_shift")):
            with self.subTest(kwargs=kwargs), patch(PREPARE, side_effect=AssertionError("prepare_problem ran")), \
                    self.assertRaisesRegex(ValueError, message):
                self.captured(**kwargs)
        for kwargs in (dict(), dict(cholesky_shift="median", optimize_cholesky_shift=False)):
            with self.subTest(nelec=None, **kwargs), patch(PREPARE, side_effect=AssertionError("prepare_problem ran")), \
                    self.assertRaisesRegex(ValueError, "particle numbers"):
                self.captured(nelec=None, **kwargs)
        self.captured(nelec=None, optimize_cholesky_shift=False)                 # no shift: no particle numbers needed

    @unittest.skipUnless(HAVE_H5PY, "h5py is required")
    def test_end_to_end(self):
        h, eri, chol = example_tensors(n=3, r=2)
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        n = problem.data.nbasis
        basis, basis_map = sector_basis(n, (2, 1))
        H = physical2MB(h, eri, basis, basis_map)
        theta0 = .1*(np.random.RandomState(4).rand(1)*2 - 1)
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in (dict(cholesky_shift="median", optimize_cholesky_shift=False), dict(),
                           dict(optimize_cholesky_shift="split")):
                with self.subTest(kwargs=kwargs):
                    path = os.path.join(folder, "run.h5")
                    result = QCSOR.optimize_from_cholesky(h, chol, (2, 1), .03, .04, path, seed=4, maxiter=3,
                                                          verbosity=0, overwrite=True, **kwargs)
                    if not kwargs:                          # the default: a tied shift, optimized from the median
                        np.testing.assert_array_equal(result.cholesky_shift[:, 0], result.cholesky_shift[:, 1])
                        median = median_cholesky_shift(problem, exponential_gauge(theta0, 2))
                        np.testing.assert_array_equal(result.initial_cholesky_shift, np.stack([median, median], 1))
                    if kwargs.get("optimize_cholesky_shift", "tied") is False:
                        median = median_cholesky_shift(problem, exponential_gauge(theta0, 2))
                        np.testing.assert_array_equal(result.cholesky_shift, np.stack([median, median], 1))
                    ham = QCSOR.from_file(path, thresh=None)
                    np.testing.assert_array_equal(ham.cholesky_shift, result.cholesky_shift)
                    check_norm(sector_lambda(ham, ham.mu, (2, 1))*np.eye(len(basis))
                               - sor_green_function(ham, basis, basis_map, n), H)


if __name__ == "__main__":
    unittest.main()
