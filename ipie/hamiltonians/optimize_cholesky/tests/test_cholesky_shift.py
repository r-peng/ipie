"""Step 9: fixed Cholesky shifts mu[gamma, sigma] in the decomposition (forward model).

The rotated factors are shifted per spin, L'_{g,s} = L_g - mu[g,s] I, so the
channels use lambda[s, g, a] = ell[g, a] - mu[g, s]. On the (N_up, N_down)
sector, with m_g = sum_s N_s mu[g,s],

    H = sum_s h_s n_s + 1/2 sum_{s,s'} sum_g L'_{g,s} (x) L'_{g,s'} + E_c,
    h_s = h + sum_g (m_g - mu[g,s]) L_g,
    E_c = 1/2 sum_g (sum_s N_s mu[g,s]^2 - m_g^2),

so sum_i a_i U_i = (denom + mu.N + E_c) I - H = lambda_electronic I - H.

A tied shift (mu_up = mu_down) keeps today's layouts and code path with ell
replaced by lambda. A split shift stores per-spin amplitudes, per-spin h bases
and a per-spin mode-space strip. The tests check:
  * exactness in Fock space for every policy, both maps, tied and split shifts;
  * no shift (None or zeros) is exactly today's decomposition;
  * a tied shift agrees with "shift as preprocessing" (shifted inputs);
  * the per-spin path, forced at equal spins, agrees with the shared path;
  * the mode-space loss and the exact coordinate profiles on split shifts;
  * the inner solve and validation errors.

The gradients with a shift (roadmap step 10) are tested in test_cholesky_gradients,
the outer optimization (step 11) in test_cholesky_shift_optimization and the
file and QCSOR (step 12) in test_io_cholesky_shift.
"""

from dataclasses import replace
import itertools
import unittest

import numpy as np

from ipie.hamiltonians.bitstring_utils import get_all_configs_u11
from ipie.hamiltonians.optimize_cholesky.channels import (
    build_fixed_decomposition,
    build_one_body_channels,
    canonical_table,
    fixed_partition_amplitudes,
    get_rotation_matrix,
    one_body_matrices,
)
from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.loss import ModeSpace, objective_loss, raw_loss
from ipie.hamiltonians.optimize_cholesky.models import FixedDecomposition, InnerOptions
from ipie.hamiltonians.optimize_cholesky.partitions import (
    optimize_partitions,
    partition_bounds,
    partition_coefficients,
    with_partitions,
)
from ipie.hamiltonians.optimize_cholesky.problem import build_gauge_state, exponential_gauge, prepare_problem
from ipie.hamiltonians.optimize_cholesky.shift import shift_coefficients, with_shift
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_norm,
    example_tensors,
    physical2MB,
    rotation2MB,
)
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import assert_close

SECTORS = ((2, 1), (1, 1), (2, 0), (2, 2))          # hybrid, ab_only, aa_only, hybrid closed shell
MAPS = ("balanced_log", "anchored_log")
TAU1, TAU2 = .03, .04
# ab_only on this 3-orbital fixture needs ~1300-1700 sweeps with or without a shift.
INNER = InnerOptions(optimize_shift=True, max_sweeps=5000)


def fixture(nelec, seed=3):
    """(problem, rotated gauge) on example_tensors(3, 2)."""
    h, eri, chol = example_tensors(n=3, r=2)
    problem = prepare_problem(h, eri, chol, nelec=nelec)
    r = problem.data.nchol
    rng = np.random.default_rng(seed)
    gauge = build_gauge_state(problem.data, exponential_gauge(.3*rng.normal(size=r*(r-1)//2), r))
    return problem, gauge


def shift_of(kind, problem, seed=5):
    """A tied (r,) or split (r, 2) shift; aa_only keeps mu_down = 0."""
    rng = np.random.default_rng(seed)
    r = problem.data.nchol
    if kind == "tied":
        return .2*rng.normal(size=r)
    mu = .2*rng.normal(size=(r, 2))
    if problem.policy == "aa_only":
        mu[:, 1] = 0.
    return mu


def one_body_shift(problem):
    return (.05, 0.) if problem.policy == "aa_only" else (.05, -.02)


def build(problem, gauge, mu, partition="balanced_log", t=.1):
    return build_fixed_decomposition(problem, gauge, TAU1, TAU2, t, partition=partition,
                                     shift=one_body_shift(problem), cholesky_shift=mu)


def sector(problem):
    n = problem.data.nbasis
    basis = get_all_configs_u11((n, n), problem.nelec)
    return basis, {cf: i for i, cf in enumerate(basis)}


def green_function(decomposition, basis, basis_map, n):
    """Unnormalized sum_i a_i U_i, lifted to the sector."""
    return sum(a*rotation2MB(get_rotation_matrix(decomposition, i), basis, basis_map, n)
               for i, a in enumerate(decomposition.coefficients))


def dense_loss(decomposition, weighted=True):
    """sum_{i<j} s_i s_j ||[u_i, u_j]||_F^2 from the dense rotation matrices."""
    U = [get_rotation_matrix(decomposition, i) for i in range(decomposition.nterms)]
    a = decomposition.coefficients
    total = 0.
    for i, j in itertools.combinations(range(decomposition.nterms), 2):
        pair = sum(np.linalg.norm(x @ y - y @ x)**2 for x, y in zip(U[i], U[j])
                   if x is not None and y is not None)
        total += (a[i]*a[j] if weighted else 1.)*pair
    return total


def per_spin(decomposition):
    """The same decomposition in the per-spin (split) representation, even at equal spins."""
    one, two = decomposition.one_body, decomposition.two_body
    if not two.split:
        _, amplitudes = fixed_partition_amplitudes(two.spin_eigenvalues, two.sources, two.tau2, two.t,
                                                   two.partition, two.amplitude_bound)
        two = replace(two, amplitudes=amplitudes, table=canonical_table(two.sources, amplitudes, two.tau2))
    if not one.split:
        X, eps = np.stack([one.X, one.X]), np.stack([one.eigenvalues, one.eigenvalues])
        for array in (X, eps):
            array.setflags(write=False)
        one = replace(one, X=X, eigenvalues=eps)
    return FixedDecomposition(one, two)


class ExactnessTests(unittest.TestCase):
    def test_sector_hamiltonian_is_exact(self):
        """sum_i a_i U_i = lambda_electronic I - H on the sector, tied and split, every policy and map."""
        for nelec, partition, kind in itertools.product(SECTORS, MAPS, ("tied", "split")):
            with self.subTest(nelec=nelec, partition=partition, shift=kind):
                problem, gauge = fixture(nelec)
                decomposition = build(problem, gauge, shift_of(kind, problem), partition)
                aa_only = problem.policy == "aa_only"
                self.assertEqual(decomposition.two_body.split, kind == "split" or aa_only)
                self.assertEqual(decomposition.one_body.split, kind == "split" and not aa_only)
                basis, basis_map = sector(problem)
                n = problem.data.nbasis
                H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
                G = green_function(decomposition, basis, basis_map, n)
                check_norm(G, decomposition.lambda_electronic*np.eye(len(basis)) - H)

    def test_one_body_matrices_and_energy(self):
        problem, gauge = fixture((2, 1))
        N = 3
        mu = shift_of("tied", problem)
        h_up, h_down = one_body_matrices(problem.data.h1, gauge.chol, np.stack([mu, mu], axis=1), (2, 1))
        np.testing.assert_array_equal(h_up, h_down)
        assert_close(h_up, problem.data.h1 + (N - 1)*np.einsum("g,gpq->pq", mu, gauge.chol), label="tied h")
        tied = build(problem, gauge, mu)
        assert_close(tied.cholesky_energy, -.5*N*(N - 1)*np.sum(mu**2), label="tied E_c")
        split_mu = shift_of("split", problem)
        split = build(problem, gauge, split_mu)
        m = split_mu @ np.array([2., 1.])
        assert_close(split.cholesky_energy, .5*np.sum((split_mu**2) @ np.array([2., 1.]) - m**2), label="E_c")
        for decomposition in (tied, split):
            assert_close(decomposition.lambda_electronic,
                         decomposition.denom + decomposition.shift_energy + decomposition.cholesky_energy,
                         label="lambda")
        unshifted = build(problem, gauge, None)
        self.assertEqual(unshifted.cholesky_energy, 0.)
        self.assertEqual(unshifted.lambda_electronic, unshifted.denom + unshifted.shift_energy)


class ZeroShiftTests(unittest.TestCase):
    def test_no_shift_is_todays_decomposition(self):
        for nelec, partition in itertools.product(SECTORS, MAPS):
            with self.subTest(nelec=nelec, partition=partition):
                problem, gauge = fixture(nelec)
                reference = build_fixed_decomposition(problem, gauge, TAU1, TAU2, .1, partition=partition,
                                                      shift=one_body_shift(problem))
                r = problem.data.nchol
                for zero in (None, np.zeros(r), np.zeros((r, 2))):
                    decomposition = build(problem, gauge, zero, partition)
                    self.assertIsNone(decomposition.cholesky_shift)
                    self.assertFalse(decomposition.two_body.split or decomposition.one_body.split)
                    self.assertEqual(decomposition.two_body.table, reference.two_body.table)
                    self.assertEqual(decomposition.one_body.table, reference.one_body.table)
                    np.testing.assert_array_equal(decomposition.two_body.amplitudes, reference.two_body.amplitudes)
                    np.testing.assert_array_equal(decomposition.one_body.X, reference.one_body.X)
                    self.assertEqual(objective_loss(decomposition), objective_loss(reference))

    def test_evaluator_matches_the_builder(self):
        problem, gauge = fixture((2, 1))
        for kind in ("tied", "split"):
            with self.subTest(shift=kind):
                mu = shift_of(kind, problem)
                expected = build(problem, gauge, mu, t=.2)
                for engine, override in ((GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=mu), None),
                                         (GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2), mu)):
                    actual = engine.decomposition(gauge.O, t=.2, shift=one_body_shift(problem), cholesky_shift=override)
                    np.testing.assert_array_equal(actual.cholesky_shift, expected.cholesky_shift)
                    assert_close(actual.two_body.table.d, expected.two_body.table.d, label="two-body d")
                    assert_close(actual.one_body.table.d, expected.one_body.table.d, label="one-body d")
                    assert_close(actual.one_body.eigenvalues, expected.one_body.eigenvalues, label="eps")
                    assert_close(objective_loss(actual), objective_loss(expected), label="loss")
        engine = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, cholesky_shift=shift_of("split", problem))
        plain = engine.decomposition(gauge.O, t=.2, cholesky_shift=np.zeros(problem.data.nchol))
        self.assertIsNone(plain.cholesky_shift)


class TiedShiftTests(unittest.TestCase):
    def test_tied_shift_matches_shift_as_preprocessing(self):
        """A tied shift equals running today's code on h + (N-1) sum mu L(O) and factors L(O) - mu I."""
        for nelec, partition in itertools.product(((2, 1), (1, 1), (2, 2)), MAPS):
            with self.subTest(nelec=nelec, partition=partition):
                problem, gauge = fixture(nelec)
                mu = shift_of("tied", problem)
                N = sum(nelec)
                n = problem.data.nbasis
                shifted = gauge.chol - mu[:, None, None]*np.eye(n)
                h = problem.data.h1 + (N - 1)*np.einsum("g,gpq->pq", mu, gauge.chol)
                eri = np.einsum("gpr,gqs->prqs", shifted, shifted)
                plain = prepare_problem(h, eri, shifted, nelec=nelec)
                expected = build_fixed_decomposition(plain, build_gauge_state(plain.data), TAU1, TAU2, .1,
                                                     partition=partition, shift=one_body_shift(problem))
                actual = build(problem, gauge, mu, partition)
                for name in ("bix", "spins", "modes"):
                    np.testing.assert_array_equal(getattr(actual.two_body.table, name),
                                                  getattr(expected.two_body.table, name))
                assert_close(actual.two_body.table.d, expected.two_body.table.d, label="two-body d")
                assert_close(actual.one_body.eigenvalues, expected.one_body.eigenvalues, label="eps")
                assert_close(actual.one_body.table.d, expected.one_body.table.d, label="one-body d")
                assert_close(objective_loss(actual), objective_loss(expected), label="loss")
                a = optimize_partitions(actual, INNER)
                b = optimize_partitions(expected, INNER)
                self.assertTrue(a.converged and b.converged)
                assert_close(a.loss, b.loss, label="relaxed loss")
                assert_close(a.shift, b.shift, label="relaxed shift")
                # The constant differs only by how the identity is split between h and E_c.
                plain_constant = .5*N*(N - 1)*np.sum(mu**2)
                assert_close(actual.lambda_electronic - actual.shift_energy - actual.denom,
                             -plain_constant, label="E_c")


class PerSpinPathTests(unittest.TestCase):
    def test_per_spin_path_reduces_to_the_shared_path(self):
        """The split representation at equal spins gives today's results (1e-12)."""
        for nelec, partition, kind in itertools.product(((2, 1), (1, 1), (2, 2)), MAPS, ("none", "tied")):
            with self.subTest(nelec=nelec, partition=partition, shift=kind):
                problem, gauge = fixture(nelec)
                shared = build(problem, gauge, None if kind == "none" else shift_of("tied", problem), partition)
                split = per_spin(shared)
                self.assertTrue(split.two_body.split and split.one_body.split)
                self.assertTrue(ModeSpace.of(split).split)
                assert_close(split.two_body.table.d, shared.two_body.table.d, label="table d")
                assert_close(objective_loss(split), objective_loss(shared), label="weighted loss")
                assert_close(raw_loss(split), raw_loss(shared), label="raw loss")
                assert_close(partition_bounds(split), partition_bounds(shared), label="bounds")
                profiles = partition_coefficients(split), partition_coefficients(shared)
                for name in ("C", "left_term", "right_term"):
                    assert_close(getattr(profiles[0], name), getattr(profiles[1], name), label=name)
                for a, b in zip(shift_coefficients(split), shift_coefficients(shared)):
                    assert_close((a.weight, a.moment, a.lower), (b.weight, b.moment, b.lower), label="mu profile")
                options = INNER
                x, y = optimize_partitions(split, options), optimize_partitions(shared, options)
                self.assertEqual((x.status, x.sweeps), (y.status, y.sweeps))
                assert_close(x.t, y.t, label="t")
                assert_close(x.loss, y.loss, label="loss")
                assert_close(x.shift, y.shift, label="shift")
                self.assertTrue(x.decomposition.two_body.split)

    def test_intervals_bound_every_spin(self):
        """At a finite interval end the binding leg sits exactly at rho, on whichever spin is larger."""
        for nelec, partition in itertools.product(((2, 1), (1, 1), (2, 2)), MAPS):
            with self.subTest(nelec=nelec, partition=partition):
                problem, gauge = fixture(nelec)
                decomposition = build(problem, gauge, shift_of("split", problem), partition, t=0.)
                rho = (1 - 1e-8)*decomposition.two_body.amplitude_bound
                bounds = partition_bounds(decomposition)
                legs = decomposition.two_body.leg_amplitudes
                spins_differ = np.abs(np.abs(legs[:, :, 0]) - np.abs(legs[:, :, 1])) > 1e-3*np.abs(legs).max()
                for end, leg in ((1, 0), (0, 1)):          # the upper end binds the left leg, the lower the right
                    at_end = with_partitions(decomposition, bounds[:, end]).two_body.leg_amplitudes
                    largest = np.max(np.abs(at_end[:, leg, :]), axis=1)
                    binding = np.abs(bounds[:, end]) < 30.
                    self.assertTrue(np.any(binding & spins_differ[:, leg]))
                    assert_close(largest[binding], np.full(int(binding.sum()), rho), label=f"end {end}")

    def test_mode_space_loss_matches_the_dense_loss(self):
        for nelec, partition in itertools.product(((2, 1), (1, 1), (2, 0)), MAPS):
            with self.subTest(nelec=nelec, partition=partition):
                problem, gauge = fixture(nelec)
                decomposition = build(problem, gauge, shift_of("split", problem), partition)
                assert_close(objective_loss(decomposition), dense_loss(decomposition), label="weighted")
                assert_close(raw_loss(decomposition), dense_loss(decomposition, weighted=False), label="raw")

    def test_coordinate_profiles_are_exact_on_split_shifts(self):
        problem, gauge = fixture((2, 1))
        for partition in MAPS:
            decomposition = build(problem, gauge, shift_of("split", problem), partition, t=0.)
            bounds = partition_bounds(decomposition)
            for i in (0, 3, len(decomposition.two_body.sources) - 1):
                with self.subTest(partition=partition, source=i):
                    profile = partition_coefficients(decomposition, i)
                    for fraction in (.2, .7):
                        s = bounds[i, 0] + fraction*(bounds[i, 1] - bounds[i, 0])
                        t = np.array(decomposition.two_body.t)
                        t[i] = s
                        assert_close(profile.value(s), objective_loss(with_partitions(decomposition, t)),
                                     label="t profile")
            for profile in shift_coefficients(decomposition):
                with self.subTest(partition=partition, spin=profile.spin):
                    shift = list(decomposition.shift)
                    shift[profile.spin] += .03
                    change = objective_loss(with_shift(decomposition, shift)) - objective_loss(decomposition)
                    assert_close(profile.change(shift[profile.spin]), change,
                                 scale=objective_loss(decomposition), label="mu profile")


class InnerSolveTests(unittest.TestCase):
    def test_inner_solve_converges_and_stays_exact(self):
        for nelec, kind in itertools.product(SECTORS, ("tied", "split")):
            with self.subTest(nelec=nelec, shift=kind):
                problem, gauge = fixture(nelec)
                decomposition = build(problem, gauge, shift_of(kind, problem), "anchored_log")
                result = optimize_partitions(decomposition, INNER)
                self.assertTrue(result.converged)
                self.assertLessEqual(result.loss, result.initial_loss)
                self.assertLessEqual(result.projected_gradient_norm, result.gradient_threshold)
                final = result.decomposition
                np.testing.assert_array_equal(final.cholesky_shift, decomposition.cholesky_shift)
                self.assertEqual(final.two_body.split, decomposition.two_body.split)
                basis, basis_map = sector(problem)
                H = physical2MB(problem.data.h1, problem.data.eri, basis, basis_map)
                check_norm(green_function(final, basis, basis_map, problem.data.nbasis),
                           final.lambda_electronic*np.eye(len(basis)) - H)


class ValidationTests(unittest.TestCase):
    def test_invalid_shifts_are_rejected(self):
        problem, gauge = fixture((2, 1))
        r = problem.data.nchol
        for bad in (np.zeros(r + 1), np.zeros((r, 3)), np.full(r, np.nan)):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                build(problem, gauge, bad)
        h, eri, chol = example_tensors(n=3, r=2)
        fock = prepare_problem(h, eri, chol)                       # no particle numbers
        with self.assertRaisesRegex(ValueError, "particle numbers"):
            build_fixed_decomposition(fock, build_gauge_state(fock.data), TAU1, TAU2, .1, cholesky_shift=np.ones(r))
        triplet, triplet_gauge = fixture((2, 0))
        with self.assertRaisesRegex(ValueError, "down-spin"):
            build(triplet, triplet_gauge, np.ones((r, 2)))
        with self.assertRaisesRegex(ValueError, "gauge"):
            build_one_body_channels(problem, TAU1, cholesky_shift=np.ones(r))
        shifted, plain = build(problem, gauge, shift_of("split", problem)), build(problem, gauge, None)
        with self.assertRaisesRegex(ValueError, "different Cholesky shifts"):
            FixedDecomposition(plain.one_body, shifted.two_body)


if __name__ == "__main__":
    unittest.main()
