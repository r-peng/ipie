"""Saving optimized decompositions (optimize_cholesky.io) and loading them with QCSOR.from_file.

QCSOR.from_file(path, *, nelec=None, parse=True, thresh=1e-6, iprint=0) drops
one-body channels with |eps_k - mu_s| < thresh and two-body channels with
|W_ab| = |ell_a ell_b| < thresh; thresh=None loads every term (exact).

The tests need h5py and ipie.hamiltonians.sor_base (ipie.utils); the walker tests
also need the LAFQMC trial and walker modules (plum). Each group is skipped when
its imports are unavailable. File metadata is read from the .h5 file directly.

Two independent references are used:
  * reference_qcsor rebuilds a QCSOR channel by channel with SizeGroup.add_term
    (the previous per-channel loader), so the array-only file path is compared
    with a construction that shares no layout code with io.py or from_file;
  * filtered_decomposition removes channels from the optimizer's own channel
    objects, so truncation is checked against a selection made independently of
    the file arrays, and the truncated Hamiltonian is checked in Fock space.

The structural comparisons (assert_same_qcsor, assert_same_terms_per_key) cover
the terms, bases and h1/v0. Lambda = denom + E0 with E0 = mu.N is tested
separately (QCSORFromFileTests.test_lambda_includes_the_shift_energy and the
Fock-space reconstructions), so a wrong shift energy shows up there by name.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io as stdio
import json
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
    import h5py
    from ipie.hamiltonians.optimize_cholesky.io import FORMAT, FORMAT_VERSION, save_decomposition
    HAVE_H5PY = True
except ImportError:  # pragma: no cover
    HAVE_H5PY = False

try:
    from ipie.hamiltonians.sor_base import QCSOR, SizeGroup, compute_total_commutativity
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
POLICY_TO_DECOMP_TYPE = {"hybrid": "all", "ab_only": "ab_only", "aa_only": "aa_only"}
DEFAULT_THRESH = 1e-6


# --------------------------------------------------------------------------- fixtures

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


def unequal_shift_fixture():
    h, eri, chol = example_tensors()
    problem = prepare_problem(h, eri, chol, nelec=(2, 1))
    return problem, build_fixed_decomposition(problem, build_gauge_state(problem.data), .03, .04, .1,
                                              shift=(.9, -.4))


def all_cases():
    """(label, problem, decomposition, nelec) over policies, maps, shifts and zero couplings."""
    cases = [(f"{nelec}{partition}", *shifted_fixture(nelec, shift, partition), nelec)
             for nelec, shift in SECTORS for partition in ("balanced_log", "anchored_log")]
    cases += [(f"zero{nelec}", *zero_coupling_fixture(nelec), nelec) for nelec in ((2, 1), (1, 1))]
    cases.append(("unequal mu", *unequal_shift_fixture(), (2, 1)))
    return cases


# --------------------------------------------------------------------------- helpers

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


def random_walkers(rng, n, nelec, nw):
    na, nb = nelec
    det = lambda: np.concatenate([np.linalg.qr(rng.normal(size=(n, n)))[0][:, :na],
                                  np.linalg.qr(rng.normal(size=(n, n)))[0][:, :nb]], axis=1)
    return det(), np.array([det() for _ in range(nw)])


def shift_energy(shift, nelec):
    return shift[0]*nelec[0] + shift[1]*nelec[1]


def file_attrs(path):
    """The file's attributes plus decoded provenance record and user metadata."""
    with h5py.File(path, "r") as f:
        attrs = {key: f.attrs[key] for key in f.attrs.keys()}
        attrs["provenance"] = json.loads(str(f["provenance"].attrs["record"]))
        attrs["user"] = json.loads(str(f["provenance"].attrs["user_metadata"]))
    return attrs


def reference_qcsor(decomposition, *, drop_identity=False, nelec=None):
    """Per-channel QCSOR construction (the previous load_optimized), parsed."""
    one, two = decomposition.one_body, decomposition.two_body
    gauge = two.gauge
    n = one.X.shape[0]
    ham = QCSOR(n, decomp_type=POLICY_TO_DECOMP_TYPE[two.policy])
    sg = SizeGroup()
    for g in range(gauge.chol.shape[0]):
        ell = np.array(gauge.eigenvalues[g])
        sg.add_basis(np.array(gauge.X[g]), np.outer(ell, ell), "thc", isometry=True)
    shift, eps, X1 = tuple(one.shift), np.array(one.eigenvalues), np.array(one.X)
    # h1 bases follow the policy (as in the file), even if every one-body channel was filtered out.
    spins = (0,) if two.policy == "aa_only" else (0, 1)
    if len(spins) == 1 or shift[0] == shift[1]:
        sg.add_basis(X1, eps - shift[spins[0]], "h1")
        h1_bix = {s: sg.get_basis_index() for s in spins}
    else:
        h1_bix = {}
        for s in spins:
            sg.add_basis(X1.copy(), eps - shift[s], "h1")
            h1_bix[s] = sg.get_basis_index()
    for c in one.channels:
        if not (drop_identity and c.d == 0.):
            sg.add_term(h1_bix[c.spin], [c.spin], float(c.a_i), [int(c.mode)], [float(c.d)])
    for c in two.channels:
        if drop_identity and c.d[0] == 0. and c.d[1] == 0.:
            continue
        (sl, sr), (pl, pr), (dl, dr) = c.spins, c.modes, c.d
        if sl == sr or (sl, sr) == (0, 1):
            sg.add_2body_term(int(c.bix), (sl, sr), float(c.a_i), [int(pl), int(pr)], [float(dl), float(dr)])
        else:
            sg.add_2body_term(int(c.bix), (0, 1), float(c.a_i), [int(pr), int(pl)], [float(dr), float(dl)])
    ham.size_groups[n] = sg
    L2 = np.einsum("gpr,grq->pq", np.asarray(gauge.chol), np.asarray(gauge.chol))
    ham.h1 = (X1*eps) @ X1.T - .5*L2
    ham.v0 = np.tile(.5*L2[None], (2, 1, 1))
    ham.mu = np.array(shift)
    ham.parse_decomposition(two.nelec if nelec is None else nelec)
    return ham


def filtered_decomposition(decomposition, keep_one, keep_two):
    """The decomposition with channel lists filtered by predicates on the channel objects."""
    one, two = decomposition.one_body, decomposition.two_body
    return replace(decomposition,
                   one_body=replace(one, channels=tuple(c for c in one.channels if keep_one(c))),
                   two_body=replace(two, channels=tuple(c for c in two.channels if keep_two(c))))


def channel_magnitudes(decomposition):
    """|eps_k - mu_s| per one-body channel and |W_ab| per two-body channel, from the objects."""
    one, two = decomposition.one_body, decomposition.two_body
    ell = two.gauge.eigenvalues
    eps = lambda c: abs(float(one.eigenvalues[c.mode]) - float(one.shift[c.spin]))
    W = lambda c: abs(float(ell[c.bix, c.modes[0]])*float(ell[c.bix, c.modes[1]]))
    return eps, W


def split_by_threshold(decomposition, thresh):
    """(kept, dropped) decompositions under from_file's rule, applied to the channel objects."""
    eps, W = channel_magnitudes(decomposition)
    kept = filtered_decomposition(decomposition, lambda c: eps(c) >= thresh, lambda c: W(c) >= thresh)
    dropped = filtered_decomposition(decomposition, lambda c: eps(c) < thresh, lambda c: W(c) < thresh)
    return kept, dropped


def threshold_between(values, fraction):
    """A threshold strictly between two distinct values (no ties); None if impossible."""
    unique = np.unique(np.asarray(values, dtype=np.float64))
    if unique.size < 2:
        return None
    i = min(max(int(fraction*(unique.size - 1)), 0), unique.size - 2)
    return 0.5*(unique[i] + unique[i + 1])


def all_magnitudes(decomposition):
    eps, W = channel_magnitudes(decomposition)
    return ([eps(c) for c in decomposition.one_body.channels],
            [W(c) for c in decomposition.two_body.channels])


class TemporaryFiles:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def saved(self, decomposition, name="opt.h5", **kwargs):
        return save_decomposition(self.path(name), decomposition, overwrite=True, **kwargs)


def assert_same_qcsor(test, ref, new):
    """Identical terms in identical global order, bases, h1 and v0 (Lambda is tested separately)."""
    test.assertEqual(ref.nterms, new.nterms)
    np.testing.assert_array_equal(ref.a, new.a)
    test.assertEqual(float(ref.denom), float(new.denom))
    np.testing.assert_array_equal(ref.E1_ixs, new.E1_ixs)
    np.testing.assert_array_equal(ref.E2_ixs, new.E2_ixs)
    test.assertEqual(ref.ix2key, new.ix2key)
    np.testing.assert_array_equal(np.asarray(ref.h1), np.asarray(new.h1))
    np.testing.assert_array_equal(ref.v0, new.v0)
    test.assertEqual(list(ref.size_groups), list(new.size_groups))
    for key in ref.size_groups:
        a, b = ref.size_groups[key], new.size_groups[key]
        np.testing.assert_array_equal(a.basis, b.basis)
        test.assertEqual(a.integral_tags, b.integral_tags)
        test.assertEqual(a.overlaps, b.overlaps)
        for x, y in zip(a.integrals, b.integrals):
            np.testing.assert_array_equal(x, y)
        test.assertEqual(list(a.term_info), list(b.term_info))
        for term_key in a.term_info:
            for name in ("p", "d", "ix"):
                x, y = a.term_info[term_key][name], b.term_info[term_key][name]
                test.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
                np.testing.assert_array_equal(x, y)
    for ix in range(ref.nterms):
        for u, v in zip(ref.get_rotation_matrix(ix), new.get_rotation_matrix(ix)):
            test.assertEqual(u is None, v is None)
            if u is not None:
                np.testing.assert_array_equal(u, v)


def assert_same_terms_per_key(test, ref, new):
    """Same bases and the same terms under every key, in the same order within a key.

    The order of the keys themselves may differ: truncation keeps the file's key
    order, whereas a reference built from filtered channels orders keys by their
    first kept channel.
    """
    test.assertEqual(ref.nterms, new.nterms)
    np.testing.assert_allclose(float(new.denom), float(ref.denom), rtol=1e-14, atol=0)
    np.testing.assert_array_equal(np.asarray(ref.h1), np.asarray(new.h1))
    (size_key,) = ref.size_groups
    test.assertEqual(list(new.size_groups), [size_key])
    sr, sn = ref.size_groups[size_key], new.size_groups[size_key]
    np.testing.assert_array_equal(sr.basis, sn.basis)
    test.assertEqual(sr.integral_tags, sn.integral_tags)
    test.assertEqual(set(sr.term_info), set(sn.term_info))
    for key in sr.term_info:
        for name in ("p", "d"):
            x, y = sr.term_info[key][name], sn.term_info[key][name]
            test.assertEqual((x.dtype, x.shape), (y.dtype, y.shape))
            np.testing.assert_array_equal(x, y)
        np.testing.assert_allclose(np.asarray(new.a)[sn.term_info[key]["ix"]],
                                   np.asarray(ref.a)[sr.term_info[key]["ix"]], rtol=1e-14, atol=0)
    for ix in range(new.nterms):  # every global index points back to itself
        _, term_key, i = new.ix2key[ix]
        test.assertEqual(int(sn.term_info[term_key]["ix"][i]), ix)


def check_truncated_hamiltonian(ham, problem, kept, dropped, nelec):
    """Lambda_kept I - sum_kept a U == H - sum_dropped a (I - U) on the (N_up, N_down) sector."""
    n = problem.data.h1.shape[0]
    basis, basis_map = sector_basis(n, nelec)
    H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
    if dropped.nterms:
        removed = reference_qcsor(dropped, nelec=nelec)
        H = H - (float(removed.denom)*np.eye(len(basis)) - sor_green_function(removed, basis, basis_map, n))
    check_norm(float(ham.denom + ham.E0)*np.eye(len(basis)) - sor_green_function(ham, basis, basis_map, n), H)


# --------------------------------------------------------------------------- the file

@unittest.skipUnless(HAVE_H5PY and HAVE_SOR, "h5py and ipie.hamiltonians.sor_base are required")
class FileFormatTests(TemporaryFiles, unittest.TestCase):
    def test_save_and_metadata(self):
        for nelec, shift in SECTORS:
            for partition in ("balanced_log", "anchored_log"):
                with self.subTest(nelec=nelec, partition=partition):
                    _, decomposition = shifted_fixture(nelec, shift, partition)
                    path = self.saved(decomposition)
                    attrs = file_attrs(path)
                    self.assertEqual((attrs["format"], int(attrs["format_version"])), (FORMAT, FORMAT_VERSION))
                    self.assertEqual((attrs["policy"], tuple(int(x) for x in attrs["nelec"]), str(attrs["partition"])),
                                     (decomposition.two_body.policy, nelec, partition))
                    np.testing.assert_array_equal(attrs["mu"], shift)
                    self.assertEqual(int(attrs["n_one_body"]) + int(attrs["n_two_body"]), decomposition.nterms)
                    self.assertAlmostEqual(float(attrs["denom"]), decomposition.denom, delta=1e-12*decomposition.denom)
                    self.assertAlmostEqual(float(attrs["lambda_electronic"]), decomposition.lambda_electronic,
                                           delta=1e-12*decomposition.denom)
                    self.assertEqual(attrs["provenance"]["source_type"], "FixedDecomposition")
                    with h5py.File(path, "r") as f:
                        np.testing.assert_array_equal(f["provenance/t"][()], decomposition.two_body.t)
                    ham = QCSOR.from_file(path, parse=False, thresh=None)
                    n = decomposition.one_body.X.shape[0]
                    stored = sum(len(info["a"]) for info in ham.size_groups[n].term_info.values())
                    self.assertEqual(stored, decomposition.nterms)

    def test_outer_result_provenance(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        result = optimize_gauge(engine, options=LBFGSOptions(max_iterations=2),
                                inner_options=InnerOptions(optimize_shift=True))
        path = save_decomposition(self.path("outer.h5"), result, metadata={"molecule": "test", "R": 1.5})
        attrs = file_attrs(path)
        record = attrs["provenance"]
        self.assertEqual(record["source_type"], "OuterOptimizationResult")
        self.assertEqual(record["outer"]["status"], result.status)
        self.assertEqual(record["outer"]["loss"], result.loss)
        self.assertEqual(record["inner"]["loss_kind"], "weighted")
        self.assertTrue(record["inner"]["optimize_shift"])
        self.assertEqual(attrs["user"], {"molecule": "test", "R": 1.5})
        np.testing.assert_array_equal(attrs["mu"], result.decomposition.shift)
        with h5py.File(path, "r") as f:
            for name, value in (("theta", result.theta), ("reference", result.reference),
                                ("initial_theta", result.initial_theta), ("O", result.O)):
                np.testing.assert_array_equal(f["provenance"][name][()], value)
        ham = QCSOR.from_file(path, nelec=(2, 1), thresh=None)
        assert_same_qcsor(self, reference_qcsor(result.decomposition), ham)

    def test_partition_result_records_its_settings(self):
        _, decomposition = fixture((2, 1))
        inner = optimize_partitions(decomposition, loss_kind="weighted", optimize_shift=True, max_sweeps=500)
        attrs = file_attrs(self.saved(inner, name="inner.h5"))
        self.assertEqual(attrs["provenance"]["source_type"], "PartitionOptimizationResult")
        self.assertEqual(attrs["provenance"]["inner"]["status"], inner.status)
        self.assertEqual(attrs["provenance"]["inner"]["sweeps"], inner.sweeps)
        np.testing.assert_array_equal(attrs["mu"], inner.shift)

    def test_drop_identity_is_a_save_option(self):
        _, decomposition = zero_coupling_fixture()
        identity = [c.a_i for c in decomposition.two_body.channels if c.d == (0., 0.)]
        self.assertTrue(identity)
        full_path = self.saved(decomposition, name="full.h5")
        drop_path = self.saved(decomposition, name="drop.h5", drop_identity=True)
        full = QCSOR.from_file(full_path, nelec=(2, 1), thresh=None)
        dropped = QCSOR.from_file(drop_path, nelec=(2, 1), thresh=None)
        self.assertEqual(full.nterms, decomposition.nterms)
        self.assertEqual(dropped.nterms, decomposition.nterms - len(identity))
        self.assertAlmostEqual(float(dropped.denom), decomposition.denom - sum(identity), places=10)
        attrs = file_attrs(drop_path)
        self.assertEqual(int(attrs["dropped_identity"]), len(identity))
        self.assertEqual(float(attrs["denom_all"]), decomposition.denom)
        self.assertEqual(int(file_attrs(full_path)["dropped_identity"]), 0)

    def test_overwrite_type_and_corruption(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        path = self.saved(decomposition, name="x.h5")
        with self.assertRaises(FileExistsError):
            save_decomposition(path, decomposition)
        save_decomposition(path, decomposition, overwrite=True)
        with self.assertRaises(TypeError):
            save_decomposition(self.path("y.h5"), object())
        with h5py.File(path, "r+") as handle:
            handle["keys"][0, 4] += 1
        with self.assertRaises(ValueError):
            QCSOR.from_file(path, nelec=(2, 1), thresh=None)
        save_decomposition(path, decomposition, overwrite=True)
        with h5py.File(path, "r+") as handle:
            handle.attrs["format_version"] = FORMAT_VERSION + 1
        with self.assertRaises(ValueError):
            QCSOR.from_file(path, nelec=(2, 1), thresh=None)
        save_decomposition(path, decomposition, overwrite=True)
        with h5py.File(path, "r+") as handle:
            handle.attrs["denom"] = handle.attrs["denom"]*1.001
        with self.assertRaises(ValueError):
            QCSOR.from_file(path, nelec=(2, 1), thresh=None)


# --------------------------------------------------------------------------- loading, untruncated

@unittest.skipUnless(HAVE_H5PY and HAVE_SOR, "h5py and ipie.hamiltonians.sor_base are required")
class QCSORFromFileTests(TemporaryFiles, unittest.TestCase):
    def test_file_path_matches_per_channel_reference(self):
        cases = [case for case in all_cases() if case[0] != "unequal mu"]
        h, eri, chol = example_tensors()
        fock = prepare_problem(h, eri, chol)
        cases.append(("fock", fock, build_fixed_decomposition(fock, build_gauge_state(fock.data), .03, .04, .1), (2, 1)))
        cases.append(("unequal mu", *unequal_shift_fixture(), (2, 1)))
        for label, _, decomposition, nelec in cases:
            for drop in (False, True):
                with self.subTest(case=label, drop_identity=drop):
                    ham = QCSOR.from_file(self.saved(decomposition, drop_identity=drop), nelec=nelec, thresh=None)
                    assert_same_qcsor(self, reference_qcsor(decomposition, drop_identity=drop, nelec=nelec), ham)

    def test_lambda_includes_the_shift_energy(self):
        """E0 = mu.N, so Lambda = denom + E0 = lambda_electronic stored in the file."""
        for nelec, shift in SECTORS + (((2, 1), (.9, -.4)),):
            with self.subTest(nelec=nelec, shift=shift):
                if shift == (.9, -.4):
                    _, decomposition = unequal_shift_fixture()
                else:
                    _, decomposition = shifted_fixture(nelec, shift)
                path = self.saved(decomposition)
                ham = QCSOR.from_file(path, nelec=nelec, thresh=None)
                self.assertAlmostEqual(float(ham.E0), shift_energy(shift, nelec), places=14)
                self.assertAlmostEqual(float(ham.denom + ham.E0), float(file_attrs(path)["lambda_electronic"]),
                                       delta=1e-12*float(ham.denom))
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        other = QCSOR.from_file(self.saved(decomposition), nelec=(1, 2), thresh=None)  # hybrid: any sector
        self.assertAlmostEqual(float(other.E0), shift_energy((.05, -.03), (1, 2)), places=14)

    def test_default_call_uses_the_stored_particle_numbers(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        path = self.saved(decomposition)
        default = QCSOR.from_file(path)
        explicit = QCSOR.from_file(path, nelec=(2, 1), thresh=DEFAULT_THRESH)
        assert_same_qcsor(self, explicit, default)
        self.assertEqual(float(default.E0), float(explicit.E0))
        h, eri, chol = example_tensors()
        fock = prepare_problem(h, eri, chol)  # saved without particle numbers
        fock_path = self.saved(build_fixed_decomposition(fock, build_gauge_state(fock.data), .03, .04, .1),
                               name="fock.h5")
        with self.assertRaisesRegex(ValueError, "nelec"):
            QCSOR.from_file(fock_path)
        QCSOR.from_file(fock_path, nelec=(2, 1))

    def test_sector_hamiltonian_is_reconstructed_exactly(self):
        for label, problem, decomposition, nelec in all_cases():
            with self.subTest(case=label):
                ham = QCSOR.from_file(self.saved(decomposition), nelec=nelec, thresh=None)
                n = problem.data.h1.shape[0]
                basis, basis_map = sector_basis(n, nelec)
                H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
                check_norm(float(ham.denom + ham.E0)*np.eye(len(basis)) - sor_green_function(ham, basis, basis_map, n), H)

    def test_losses_match_sor_diagnostics(self):
        for nelec, shift in SECTORS:
            with self.subTest(nelec=nelec):
                _, decomposition = shifted_fixture(nelec, shift)
                ham = QCSOR.from_file(self.saved(decomposition), nelec=nelec, thresh=None)
                buffer = stdio.StringIO()
                with redirect_stdout(buffer):
                    compute_total_commutativity(ham)
                printed = dict(line.split("=", 1) for line in buffer.getvalue().split("\n") if "=" in line)
                reference = commutator_loss(decomposition, loss_kind="weighted")
                np.testing.assert_allclose(float(printed["comm2"]), reference.comm2, rtol=1e-11)
                np.testing.assert_allclose(float(printed["weighted_comm2"]), reference.objective, rtol=1e-11)

    def test_local_energy_inputs_and_spin_resolved_bases(self):
        for shift, expected in (((.05, .05), 1), ((.05, -.03), 2)):
            with self.subTest(shift=shift):
                problem, decomposition = shifted_fixture((2, 1), shift)
                ham = QCSOR.from_file(self.saved(decomposition), parse=False, thresh=None)
                h, chol = problem.data.h1, problem.data.chol
                L2 = np.einsum("gpr,grq->pq", chol, chol)
                np.testing.assert_allclose(np.asarray(ham.h1), h - .5*L2, atol=1e-13)
                np.testing.assert_allclose(np.asarray(ham.h1) + ham.v0[0], h, atol=1e-13)
                sg = ham.size_groups[h.shape[0]]
                gauge = decomposition.two_body.gauge
                r = gauge.chol.shape[0]
                for g in range(r):
                    np.testing.assert_array_equal(sg.integrals[g], np.outer(gauge.eigenvalues[g], gauge.eigenvalues[g]))
                    self.assertIsNone(sg.overlaps[g])
                self.assertEqual(sg.integral_tags, ["thc"]*r + ["h1"]*expected)
                for i, s in zip(range(r, r + expected), (0, 1)):
                    np.testing.assert_allclose(sg.integrals[i], decomposition.one_body.eigenvalues - shift[s])

    def test_sector_checks(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        unparsed = QCSOR.from_file(self.saved(decomposition), parse=False, thresh=None)
        self.assertFalse(hasattr(unparsed, "a"))
        _, ab = shifted_fixture((1, 1), (.1, .02))
        with self.assertRaises(ValueError):
            QCSOR.from_file(self.saved(ab, name="ab.h5"), nelec=(2, 1), thresh=None)
        _, aa = shifted_fixture((2, 0), (.04, 0.))
        with self.assertRaises(ValueError):
            QCSOR.from_file(self.saved(aa, name="aa.h5"), nelec=(2, 1), thresh=None)

    @unittest.skipUnless(HAVE_WALKERS, "LAFQMC trial/walker modules are not importable")
    def test_walker_local_energy_matches_the_exact_mixed_estimator(self):
        rng = np.random.default_rng(31)
        for nelec, shift in SECTORS[:2]:
            with self.subTest(nelec=nelec):
                problem, decomposition = shifted_fixture(nelec, shift)
                ham = QCSOR.from_file(self.saved(decomposition), nelec=nelec, thresh=None)
                h, eri = problem.data.h1, problem.data.eri
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

    @unittest.skipUnless(HAVE_WALKERS, "LAFQMC trial/walker modules are not importable")
    def test_same_seed_propagation_matches_reference(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        n, nelec, nw = 3, (2, 1), 6
        psi, phi0 = random_walkers(np.random.default_rng(1), n, nelec, nw)
        out = []
        for ham in (reference_qcsor(decomposition),
                    QCSOR.from_file(self.saved(decomposition), nelec=nelec, thresh=None)):
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
            np.testing.assert_array_equal(x, y)


# --------------------------------------------------------------------------- loading, truncated

@unittest.skipUnless(HAVE_H5PY and HAVE_SOR, "h5py and ipie.hamiltonians.sor_base are required")
class TruncationTests(TemporaryFiles, unittest.TestCase):
    """from_file drops one-body channels with |eps_k - mu_s| < thresh and two-body
    channels with |W_ab| < thresh."""

    def test_none_and_zero_thresholds_keep_every_term(self):
        for label, _, decomposition, nelec in all_cases():
            with self.subTest(case=label):
                path = self.saved(decomposition)
                exact = QCSOR.from_file(path, nelec=nelec, thresh=None)
                self.assertEqual(exact.nterms, decomposition.nterms)
                assert_same_qcsor(self, exact, QCSOR.from_file(path, nelec=nelec, thresh=0.))

    def test_default_threshold_drops_only_tiny_terms(self):
        for label, _, decomposition, nelec in all_cases():
            with self.subTest(case=label):
                path = self.saved(decomposition)
                default = QCSOR.from_file(path, nelec=nelec)
                kept, _ = split_by_threshold(decomposition, DEFAULT_THRESH)
                assert_same_terms_per_key(self, reference_qcsor(kept, nelec=nelec), default)
                one, two = all_magnitudes(decomposition)
                if min(one + two) >= DEFAULT_THRESH:  # nothing tiny: identical to the exact load
                    assert_same_qcsor(self, QCSOR.from_file(path, nelec=nelec, thresh=None), default)
                else:
                    self.assertLess(default.nterms, decomposition.nterms)

    def test_matches_filtered_reference_and_truncated_hamiltonian(self):
        for label, problem, decomposition, nelec in all_cases():
            path = self.saved(decomposition)
            one, two = all_magnitudes(decomposition)
            for fraction in (.3, .6):
                thresh = threshold_between(one + two, fraction)
                with self.subTest(case=label, thresh=thresh):
                    kept, dropped = split_by_threshold(decomposition, thresh)
                    self.assertGreater(dropped.nterms, 0)
                    ham = QCSOR.from_file(path, nelec=nelec, thresh=thresh)
                    assert_same_terms_per_key(self, reference_qcsor(kept, nelec=nelec), ham)
                    np.testing.assert_allclose(float(ham.denom), decomposition.denom - dropped.denom,
                                               rtol=1e-13, atol=0)
                    # Sources are dropped whole: all branches share W.
                    _, W = channel_magnitudes(decomposition)
                    two_body = decomposition.two_body
                    for start, stop in two_body.source_slices:
                        self.assertLessEqual(len({W(c) >= thresh for c in two_body.channels[start:stop]}), 1)
                    check_truncated_hamiltonian(ham, problem, kept, dropped, nelec)

    def test_thresh_compares_the_shifted_orbital_energy(self):
        _, decomposition = unequal_shift_fixture()
        shift = decomposition.one_body.shift
        eps_k = np.asarray(decomposition.one_body.eigenvalues)
        shifted = np.stack([np.abs(eps_k - shift[0]), np.abs(eps_k - shift[1])])
        candidates = [threshold_between(shifted.ravel(), f) for f in np.linspace(0, 1, 7)]
        # A threshold that keeps some orbital for one spin and drops it for the other.
        thresh = next(t for t in candidates if np.any((shifted[0] >= t) != (shifted[1] >= t)))
        ham = QCSOR.from_file(self.saved(decomposition), thresh=thresh, nelec=(2, 1))
        sg = ham.size_groups[eps_k.size]
        r = decomposition.two_body.gauge.chol.shape[0]
        for spin, bix in ((0, r), (1, r + 1)):  # unequal mu: one h1 basis per spin
            key = (bix, (spin,))
            modes = sg.term_info[key]["p"].ravel() if key in sg.term_info else np.zeros(0)
            np.testing.assert_array_equal(np.sort(modes), np.flatnonzero(shifted[spin] >= thresh))
        kept, _ = split_by_threshold(decomposition, thresh)
        assert_same_terms_per_key(self, reference_qcsor(kept, nelec=(2, 1)), ham)

    def test_tiny_thresh_drops_exactly_the_zero_couplings(self):
        for nelec in ((2, 1), (1, 1)):
            with self.subTest(nelec=nelec):
                _, decomposition = zero_coupling_fixture(nelec)
                truncated = QCSOR.from_file(self.saved(decomposition, name="full.h5"), nelec=nelec, thresh=1e-14)
                dropped = QCSOR.from_file(self.saved(decomposition, name="drop.h5", drop_identity=True),
                                          nelec=nelec, thresh=None)
                assert_same_terms_per_key(self, dropped, truncated)
                zero = [c for c in decomposition.two_body.channels if c.d == (0., 0.)]
                self.assertEqual(truncated.nterms, decomposition.nterms - len(zero))

    def test_dropping_a_whole_class_or_every_term(self):
        problem, decomposition = shifted_fixture((2, 1), (.05, -.03))
        one, two = all_magnitudes(decomposition)
        # Every one-body term below thresh, some two-body terms above it.
        thresh = 0.5*(max(one) + min(w for w in two if w > max(one)))
        ham = QCSOR.from_file(self.saved(decomposition), nelec=(2, 1), thresh=thresh)
        self.assertEqual(ham.E1_ixs.size, 0)
        self.assertGreater(ham.E2_ixs.size, 0)
        check_truncated_hamiltonian(ham, problem, *split_by_threshold(decomposition, thresh), (2, 1))

        problem, decomposition = unequal_shift_fixture()
        one, two = all_magnitudes(decomposition)
        # Every two-body term below thresh, some one-body terms above it.
        thresh = 0.5*(max(two) + min(e for e in one if e > max(two)))
        ham = QCSOR.from_file(self.saved(decomposition, name="u.h5"), nelec=(2, 1), thresh=thresh)
        self.assertEqual(ham.E2_ixs.size, 0)
        self.assertGreater(ham.E1_ixs.size, 0)
        check_truncated_hamiltonian(ham, problem, *split_by_threshold(decomposition, thresh), (2, 1))

        with self.assertRaises(ValueError):
            QCSOR.from_file(self.saved(decomposition, name="v.h5"), nelec=(2, 1), thresh=1e9)

    def test_parse_false_then_parse(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        thresh = threshold_between(sum(all_magnitudes(decomposition), []), .5)
        path = self.saved(decomposition)
        parsed = QCSOR.from_file(path, nelec=(2, 1), thresh=thresh)
        unparsed = QCSOR.from_file(path, parse=False, thresh=thresh)
        self.assertFalse(hasattr(unparsed, "a"))
        unparsed.parse_decomposition((2, 1))
        assert_same_qcsor(self, parsed, unparsed)

    @unittest.skipUnless(HAVE_WALKERS, "LAFQMC trial/walker modules are not importable")
    def test_local_energy_is_unchanged_and_propagation_runs(self):
        _, decomposition = shifted_fixture((2, 1), (.05, -.03))
        thresh = threshold_between(sum(all_magnitudes(decomposition), []), .5)
        path = self.saved(decomposition)
        full = QCSOR.from_file(path, nelec=(2, 1), thresh=None)
        truncated = QCSOR.from_file(path, nelec=(2, 1), thresh=thresh)
        self.assertLess(truncated.nterms, full.nterms)
        n, nelec, nw = 3, (2, 1), 4
        psi, phi = random_walkers(np.random.default_rng(5), n, nelec, nw)
        energies = []
        for ham in (full, truncated):
            trial = SingleDet(psi, nelec, n)
            trial.build(ham)
            walkers = UHFWalkers(phi[0], *nelec, n, nw, None)
            walkers.phi = phi.copy()
            walkers.compute_S(trial, set_buff=True)
            energies.append(walkers.local_energy(ham, trial)[0])
        # h1, v0 and the W integrals are not truncated: the estimator still measures the full H.
        np.testing.assert_allclose(energies[1], energies[0], rtol=1e-12, atol=1e-14)

        trial = SingleDet(psi, nelec, n)
        trial.build(truncated)
        walkers = UHFWalkers(phi[0], *nelec, n, nw, None)
        walkers.phi = phi.copy()
        walkers.compute_S(trial, set_buff=True)
        sampler, b = np.random.default_rng(3), np.ones(nw)
        for _ in range(30):
            sample = sampler.choice(truncated.nterms, size=nw, p=np.asarray(truncated.a))
            truncated.parse_samples(sample)
            b = walkers.update_walkers(truncated, trial, b=b)
        self.assertTrue(np.all(np.isfinite(b)))
        self.assertTrue(np.all(np.isfinite(walkers.local_energy(truncated, trial)[0])))


if __name__ == "__main__":
    unittest.main()
