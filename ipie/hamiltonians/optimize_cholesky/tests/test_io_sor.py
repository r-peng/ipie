"""Saving/loading optimized decompositions and building QCSOR from them.

The file-format tests need h5py; the QCSOR tests need ipie.hamiltonians.sor_base
(ipie.utils); the walker local-energy test also needs the LAFQMC trial and walker
modules (plum). Each group is skipped when its imports are unavailable.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io as stdio
import os
import tempfile
import unittest

import numpy as np

from ipie.hamiltonians.bitstring_utils import get_all_configs_u11
from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator,
    InnerOptions,
    LBFGSOptions,
    build_fixed_decomposition,
    build_gauge_state,
    commutator_loss,
    exponential_gauge,
    objective_loss,
    optimize_gauge,
    optimize_partitions,
    prepare_problem,
    with_shift,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_norm,
    example_tensors,
    physical2MB,
    rotation2MB,
    slater2MB,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_weighted import fixture

try:
    import h5py  # noqa: F401
    from ipie.hamiltonians.optimize_cholesky.io import (
        FORMAT_VERSION,
        SavedDecomposition,
        load_decomposition,
        rebuild_decomposition,
        save_decomposition,
    )
    HAVE_H5PY = True
except ImportError:  # pragma: no cover
    HAVE_H5PY = False

try:
    from ipie.hamiltonians.sor_base import QCSOR, compute_total_commutativity
    HAVE_SOR = True
except ImportError:  # pragma: no cover
    HAVE_SOR = False

try:
    from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
    from ipie.walkers.lafqmc_uhf_walkers import UHFWalkers
    HAVE_WALKERS = True
except ImportError:  # pragma: no cover
    HAVE_WALKERS = False


SECTORS = (((2, 1), (.05, -.03)), ((1, 1), (.1, .02)), ((2, 0), (.04, 0.)))


def shifted_fixture(nelec, shift, partition="balanced_log"):
    problem, decomposition = fixture(nelec, partition=partition)
    return problem, with_shift(decomposition, shift)


def zero_coupling_fixture(nelec=(2, 1)):
    """A factor with a zero eigenvalue and an all-zero factor: exact identity channels."""
    h = np.array([[0.0, 0.2], [0.2, 0.1]])
    chol = np.array([np.diag([0., .6]), np.zeros((2, 2))])
    eri = np.einsum("gpr,gqs->prqs", chol, chol)
    problem = prepare_problem(h, eri, chol, nelec=nelec)
    decomposition = build_fixed_decomposition(problem, build_gauge_state(problem.data), .03, .04, .13)
    return problem, decomposition


def sector_basis(n, nelec):
    basis = get_all_configs_u11((n, n), nelec)
    return basis, {cf: i for i, cf in enumerate(basis)}


def sor_green_function(ham, basis, basis_map, n):
    """Unnormalized sum_i a_i U_i from the PARSED QCSOR (a is stored normalized)."""
    G = np.zeros((len(basis), len(basis)))
    for ix in range(ham.nterms):
        G += float(ham.a[ix])*float(ham.denom)*rotation2MB(ham.get_rotation_matrix(ix), basis, basis_map, n)
    return G


def uhf_spin_orbitals(phi, nup, n):
    """(n, Nup+Ndown) UHF orbitals -> (2n, N) spin-interleaved determinant (index 2p+s)."""
    full = np.zeros((2*n, phi.shape[1]))
    full[0::2, :nup] = phi[:, :nup]
    full[1::2, nup:] = phi[:, nup:]
    return full


def tensors(problem):
    return problem.data.h1, problem.data.eri, problem.data.chol


@unittest.skipUnless(HAVE_H5PY, "h5py is required for the decomposition file format")
class FileFormatTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def assert_same_decomposition(self, loaded, original):
        one, two = loaded.one_body, loaded.two_body
        for name in ("X", "eigenvalues"):
            np.testing.assert_array_equal(getattr(one, name), getattr(original.one_body, name))
        self.assertEqual(one.tau1, original.one_body.tau1)
        self.assertEqual(one.shift, original.one_body.shift)
        self.assertEqual(one.channels, original.one_body.channels)
        for name in ("O", "chol", "eigenvalues", "X"):
            np.testing.assert_array_equal(getattr(two.gauge, name), getattr(original.two_body.gauge, name))
        for name in ("policy", "nelec", "tau2", "partition", "amplitude_bound", "sources",
                     "source_slices", "channels"):
            self.assertEqual(getattr(two, name), getattr(original.two_body, name), name)
        np.testing.assert_array_equal(two.t, original.two_body.t)
        np.testing.assert_array_equal(two.amplitudes, original.two_body.amplitudes)
        self.assertEqual(loaded.denom, original.denom)
        for loss_kind in ("raw", "weighted"):
            self.assertEqual(objective_loss(loaded, loss_kind=loss_kind),
                             objective_loss(original, loss_kind=loss_kind))
        self.assertFalse(two.t.flags.writeable)
        self.assertFalse(two.gauge.X.flags.writeable)

    def test_round_trip_reproduces_every_field_bitwise(self):
        for nelec, shift in SECTORS:
            for partition in ("balanced_log", "anchored_log"):
                with self.subTest(nelec=nelec, partition=partition):
                    problem, decomposition = shifted_fixture(nelec, shift, partition)
                    path = save_decomposition(self.path(f"{nelec}{partition}.h5"), decomposition,
                                              problem=problem)
                    saved = load_decomposition(path)
                    self.assertIsInstance(saved, SavedDecomposition)
                    self.assert_same_decomposition(saved.decomposition, decomposition)
                    self.assertEqual(saved.metadata["format_version"], FORMAT_VERSION)
                    self.assertEqual(saved.metadata["provenance"]["source_type"], "FixedDecomposition")
                    self.assertAlmostEqual(saved.decomposition.lambda_electronic,
                                           decomposition.lambda_electronic, places=12)
                    rebuild_decomposition(saved)

    def test_full_fock_hybrid_without_particle_numbers(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol)
        decomposition = build_fixed_decomposition(problem, build_gauge_state(problem.data), .03, .04, .1)
        saved = load_decomposition(save_decomposition(self.path("fock.h5"), decomposition))
        self.assertIsNone(saved.nelec)
        self.assert_same_decomposition(saved.decomposition, decomposition)
        with self.assertRaises(ValueError):  # no stored tensors
            saved.problem()

    def test_outer_result_provenance_tensors_and_rebuild(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        result = optimize_gauge(engine, options=LBFGSOptions(max_iterations=2),
                                inner_options=InnerOptions(optimize_shift=True))
        path = save_decomposition(self.path("outer.h5"), result, problem=problem, store_eri=True,
                                  metadata={"molecule": "test", "R": 1.5})
        saved = load_decomposition(path)
        self.assert_same_decomposition(saved.decomposition, result.decomposition)
        record = saved.metadata["provenance"]
        self.assertEqual(record["source_type"], "OuterOptimizationResult")
        self.assertEqual(record["outer"]["status"], result.status)
        self.assertEqual(record["outer"]["loss"], result.loss)
        self.assertEqual(record["inner"]["loss_kind"], "weighted")
        self.assertTrue(record["inner"]["optimize_shift"])
        self.assertEqual(saved.loss_kind, "weighted")
        self.assertEqual(saved.metadata["user"], {"molecule": "test", "R": 1.5})
        for name, value in (("theta", result.theta), ("reference", result.reference),
                            ("initial_theta", result.initial_theta)):
            np.testing.assert_array_equal(saved.arrays[name], value)
        np.testing.assert_allclose(objective_loss(saved.decomposition, loss_kind="weighted"), result.loss,
                                   rtol=1e-12)
        for name in ("h1", "chol", "eri"):
            np.testing.assert_array_equal(saved.tensors[name], getattr(problem.data, name))
        rebuilt = rebuild_decomposition(saved)
        self.assertEqual(rebuilt.shift, result.decomposition.shift)

    def test_partition_result_records_its_settings(self):
        problem, decomposition = fixture((2, 1))
        inner = optimize_partitions(decomposition, loss_kind="weighted", optimize_shift=True, max_sweeps=500)
        saved = load_decomposition(save_decomposition(self.path("inner.h5"), inner))
        record = saved.metadata["provenance"]
        self.assertEqual(record["source_type"], "PartitionOptimizationResult")
        self.assertEqual(record["inner"]["status"], inner.status)
        self.assertEqual(record["inner"]["sweeps"], inner.sweeps)
        self.assertEqual(saved.shift, inner.shift)
        self.assert_same_decomposition(saved.decomposition, inner.decomposition)

    def test_overwrite_format_and_corruption_are_rejected(self):
        problem, decomposition = shifted_fixture((2, 1), (.05, -.03))
        path = save_decomposition(self.path("x.h5"), decomposition)
        with self.assertRaises(FileExistsError):
            save_decomposition(path, decomposition)
        save_decomposition(path, decomposition, overwrite=True)
        with self.assertRaises(TypeError):
            save_decomposition(self.path("y.h5"), object())
        other_problem, _ = fixture((1, 1))
        with self.assertRaises(ValueError):  # policy / nelec mismatch
            save_decomposition(self.path("z.h5"), decomposition, problem=other_problem)
        with h5py.File(path, "r+") as handle:
            handle["two_body/channels/d"][0, 0] *= 1.01
        with self.assertRaises(ValueError):
            load_decomposition(path)
        save_decomposition(path, decomposition, overwrite=True)
        with h5py.File(path, "r+") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION + 1
        with self.assertRaises(ValueError):
            load_decomposition(path)
        with h5py.File(path, "r+") as handle:
            handle.attrs["format"] = "something-else"
        with self.assertRaises(ValueError):
            load_decomposition(path)


@unittest.skipUnless(HAVE_SOR, "ipie.hamiltonians.sor_base is not importable")
class QCSORFromOptimizedTests(unittest.TestCase):
    def test_terms_weights_and_shift_match_the_optimizer(self):
        for nelec, shift in SECTORS:
            with self.subTest(nelec=nelec):
                _, decomposition = shifted_fixture(nelec, shift)
                ham = QCSOR.from_optimized(decomposition, parse=True)
                self.assertEqual(ham.decomp_type, {"hybrid": "all"}.get(decomposition.two_body.policy,
                                                                        decomposition.two_body.policy))
                self.assertEqual(ham.nterms, decomposition.nterms)
                self.assertAlmostEqual(float(ham.denom), decomposition.denom, delta=1e-12*decomposition.denom)
                self.assertAlmostEqual(float(ham.E0), decomposition.shift_energy, places=14)
                self.assertAlmostEqual(float(ham.denom+ham.E0), decomposition.lambda_electronic,
                                       delta=1e-12*decomposition.denom)
                tags = ham.size_groups[decomposition.one_body.X.shape[0]].integral_tags
                r = decomposition.two_body.gauge.chol.shape[0]
                self.assertEqual(tags[:r], ["thc"]*r)
                expected_h1 = 1 if (nelec[1] == 0 or shift[0] == shift[1]) else 2
                self.assertEqual(tags[r:], ["h1"]*expected_h1)

    def test_sector_hamiltonian_is_reconstructed_exactly(self):
        for nelec, shift in SECTORS:
            for partition in ("balanced_log", "anchored_log"):
                with self.subTest(nelec=nelec, partition=partition):
                    problem, decomposition = shifted_fixture(nelec, shift, partition)
                    ham = QCSOR.from_optimized(decomposition, parse=True)
                    h, eri, _ = tensors(problem)
                    n = h.shape[0]
                    basis, basis_map = sector_basis(n, nelec)
                    H = physical2MB(h, eri, basis, basis_map)
                    G = sor_green_function(ham, basis, basis_map, n)
                    check_norm(float(ham.denom+ham.E0)*np.eye(len(basis))-G, H)

    def test_losses_match_sor_diagnostics(self):
        for nelec, shift in SECTORS:
            with self.subTest(nelec=nelec):
                _, decomposition = shifted_fixture(nelec, shift)
                ham = QCSOR.from_optimized(decomposition, parse=True)
                buffer = stdio.StringIO()
                with redirect_stdout(buffer):
                    compute_total_commutativity(ham)
                printed = dict(line.split("=", 1) for line in buffer.getvalue().split("\n") if "=" in line)
                reference = commutator_loss(decomposition, loss_kind="weighted")
                np.testing.assert_allclose(float(printed["comm2"]), reference.comm2, rtol=1e-11)
                np.testing.assert_allclose(float(printed["weighted_comm2"]), reference.objective, rtol=1e-11)

    def test_local_energy_inputs(self):
        problem, decomposition = shifted_fixture((2, 1), (.05, -.03))
        ham = QCSOR.from_optimized(decomposition)
        h, _, chol = tensors(problem)
        L2 = np.einsum("gpr,grq->pq", chol, chol)  # gauge invariant
        np.testing.assert_allclose(np.asarray(ham.h1), h-.5*L2, atol=1e-13)
        np.testing.assert_allclose(np.asarray(ham.h1)+ham.v0[0], h, atol=1e-13)
        sg = ham.size_groups[h.shape[0]]
        gauge = decomposition.two_body.gauge
        for g in range(gauge.chol.shape[0]):
            np.testing.assert_array_equal(sg.integrals[g], np.outer(gauge.eigenvalues[g], gauge.eigenvalues[g]))
            self.assertIsNone(sg.overlaps[g])
            np.testing.assert_allclose((sg.basis[g]*gauge.eigenvalues[g]) @ sg.basis[g].T, gauge.chol[g],
                                       atol=1e-13)

    def test_spin_resolved_shift_uses_one_h1_basis_per_spin(self):
        for shift, expected in (((.05, .05), 1), ((.05, -.03), 2)):
            _, decomposition = shifted_fixture((2, 1), shift)
            ham = QCSOR.from_optimized(decomposition)
            sg = ham.size_groups[decomposition.one_body.X.shape[0]]
            h1 = [i for i, tag in enumerate(sg.integral_tags) if tag == "h1"]
            self.assertEqual(len(h1), expected)
            for i, s in zip(h1, (0, 1) if expected == 2 else (0,)):
                np.testing.assert_allclose(sg.integrals[i], decomposition.one_body.eigenvalues-shift[s])
            np.testing.assert_array_equal(ham.mu, shift)

    def test_drop_identity_is_exact_and_lowers_lambda(self):
        for nelec in ((2, 1), (1, 1)):
            with self.subTest(nelec=nelec):
                problem, decomposition = zero_coupling_fixture(nelec)
                identity = [c.a_i for c in decomposition.two_body.channels if c.d == (0., 0.)]
                self.assertTrue(identity)
                full = QCSOR.from_optimized(decomposition, parse=True)
                dropped = QCSOR.from_optimized(decomposition, parse=True, drop_identity=True)
                self.assertEqual(full.nterms, decomposition.nterms)
                self.assertEqual(dropped.nterms, decomposition.nterms-len(identity))
                self.assertEqual(dropped.dropped_identity, len(identity))
                self.assertAlmostEqual(float(dropped.denom), decomposition.denom-sum(identity), places=10)
                h, eri, _ = tensors(problem)
                basis, basis_map = sector_basis(h.shape[0], nelec)
                H = physical2MB(h, eri, basis, basis_map)
                for ham in (full, dropped):
                    G = sor_green_function(ham, basis, basis_map, h.shape[0])
                    check_norm(float(ham.denom+ham.E0)*np.eye(len(basis))-G, H)

    def test_sources_validation_and_sector_checks(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        inner = optimize_partitions(decomposition, max_sweeps=0)
        self.assertEqual(QCSOR.from_optimized(inner, parse=True).nterms, decomposition.nterms)
        with self.assertRaises(TypeError):
            QCSOR.from_optimized(object())
        ham = QCSOR.from_optimized(decomposition)
        with self.assertRaises(ValueError):  # not empty any more
            ham.load_optimized(decomposition)
        with self.assertRaises(ValueError):
            QCSOR(4).load_optimized(decomposition)  # wrong nbasis
        with self.assertRaises(ValueError):
            QCSOR(3, decomp_type="ab_only").load_optimized(decomposition)  # wrong policy
        _, ab = shifted_fixture((1, 1), (.1, .02))
        with self.assertRaises(ValueError):
            QCSOR.from_optimized(ab, parse=True, nelec=(2, 1))
        _, aa = shifted_fixture((2, 0), (.04, 0.))
        with self.assertRaises(ValueError):
            QCSOR.from_optimized(aa, parse=True, nelec=(2, 1))
        hybrid = QCSOR.from_optimized(decomposition, parse=True, nelec=(1, 2))  # hybrid: any sector
        self.assertAlmostEqual(float(hybrid.E0), .05*1-.03*2, places=14)

    @unittest.skipUnless(HAVE_H5PY, "h5py is required for the decomposition file format")
    def test_from_file_matches_in_memory(self):
        problem, decomposition = shifted_fixture((2, 1), (.05, -.03))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_decomposition(os.path.join(tmp, "opt.h5"), decomposition, problem=problem,
                                      metadata={"tag": 1})
            from_file = QCSOR.from_optimized(path, parse=True)
        in_memory = QCSOR.from_optimized(decomposition, parse=True)
        self.assertEqual(from_file.optimized_metadata["user"], {"tag": 1})
        np.testing.assert_array_equal(from_file.a, in_memory.a)
        self.assertEqual(float(from_file.denom), float(in_memory.denom))
        self.assertEqual(float(from_file.E0), float(in_memory.E0))
        np.testing.assert_array_equal(np.asarray(from_file.h1), np.asarray(in_memory.h1))
        for ix in range(in_memory.nterms):
            for left, right in zip(from_file.get_rotation_matrix(ix), in_memory.get_rotation_matrix(ix)):
                if left is None or right is None:
                    self.assertIs(left, right)
                else:
                    np.testing.assert_array_equal(left, right)

    @unittest.skipUnless(HAVE_WALKERS, "LAFQMC trial/walker modules are not importable")
    def test_walker_local_energy_matches_the_exact_mixed_estimator(self):
        rng = np.random.default_rng(31)
        for nelec, shift in SECTORS[:2]:
            with self.subTest(nelec=nelec):
                problem, decomposition = shifted_fixture(nelec, shift)
                ham = QCSOR.from_optimized(decomposition, parse=True)
                h, eri, _ = tensors(problem)
                n, (na, nb) = h.shape[0], nelec
                psi = np.concatenate([np.linalg.qr(rng.normal(size=(n, n)))[0][:, :na],
                                      np.linalg.qr(rng.normal(size=(n, n)))[0][:, :nb]], axis=1)
                trial = SingleDet(psi, nelec, n)
                trial.build(ham)
                nw = 3
                phi = np.array([np.concatenate([np.linalg.qr(rng.normal(size=(n, n)))[0][:, :na],
                                                np.linalg.qr(rng.normal(size=(n, n)))[0][:, :nb]], axis=1)
                                for _ in range(nw)])
                walkers = UHFWalkers(phi[0], na, nb, n, nw, None)
                walkers.phi = phi.copy()
                walkers.compute_S(trial, set_buff=True)
                eloc, _, _ = walkers.local_energy(ham, trial)
                basis, basis_map = sector_basis(n, nelec)
                H = physical2MB(h, eri, basis, basis_map)
                left = slater2MB(uhf_spin_orbitals(psi, na, n), basis)
                for w in range(nw):
                    right = slater2MB(uhf_spin_orbitals(phi[w], na, n), basis)
                    exact = (left @ H @ right)/(left @ right)
                    np.testing.assert_allclose(float(np.real(eloc[w])), exact, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
