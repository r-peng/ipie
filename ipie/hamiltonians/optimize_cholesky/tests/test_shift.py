"""Spin-resolved one-body shifts (mu_up, mu_down): channels, Lambda, profiles, relaxation."""

from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

from ipie.hamiltonians.bitstring_utils import get_all_configs_u11
from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator,
    InnerConvergenceError,
    InnerOptions,
    LBFGSOptions,
    build_fixed_decomposition,
    build_gauge_state,
    build_one_body_channels,
    check_gauge_direction,
    exponential_gauge,
    objective_loss,
    optimal_shift,
    optimize_gauge,
    optimize_partitions,
    prepare_problem,
    relaxed_gauge_gradient,
    shift_bounds,
    shift_coefficients,
    with_shift,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_norm,
    example_tensors,
    get_MB_gf,
    physical2MB,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_weighted import fixture, two_orbital_problem


LOSSES = ("raw", "weighted")


def loss_at(decomposition, shift, loss_kind):
    return objective_loss(with_shift(decomposition, shift), loss_kind=loss_kind)


class ShiftChannelTests(unittest.TestCase):
    def test_amplitudes_domain_and_particle_number_requirements(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        legacy = build_one_body_channels(problem, .03)
        shifted = build_one_body_channels(problem, .03, (.2, -.1))
        self.assertEqual(legacy.shift, (0., 0.))
        self.assertEqual(shifted.shift, (.2, -.1))
        for old, new in zip(legacy.channels, shifted.channels):
            self.assertEqual(old.d, -legacy.eigenvalues[old.mode]*.03)
            mu = (.2, -.1)[new.spin]
            self.assertAlmostEqual(new.d, (mu-shifted.eigenvalues[new.mode])*.03, places=15)
        lower = shift_bounds(shifted)[0]
        with self.assertRaises(ValueError):  # 1+d <= 0 for the top orbital
            build_one_body_channels(problem, .03, (lower-1., 0.))
        no_counts = prepare_problem(h, eri, chol)
        build_one_body_channels(no_counts, .03)  # zero shift needs no particle numbers
        with self.assertRaises(ValueError):
            build_one_body_channels(no_counts, .03, (.1, .1))
        up_only = prepare_problem(h, eri, chol, nelec=(2, 0))
        self.assertEqual(build_one_body_channels(up_only, .03, (.3, 0.)).spins, (0,))
        with self.assertRaises(ValueError):
            build_one_body_channels(up_only, .03, (.3, .1))
        with self.assertRaises(ValueError):
            GaugeEvaluator(no_counts, tau1=.03, tau2=.04, shift=(.1, 0.))

    def test_lambda_includes_mu_n_and_reconstructs_each_sector(self):
        h, eri, chol = example_tensors()
        n = h.shape[0]
        for nelec, shift in (((2, 1), (.3, -.2)), ((1, 1), (.1, .25)), ((2, 0), (.4, 0.))):
            with self.subTest(nelec=nelec):
                problem = prepare_problem(h, eri, chol, nelec=nelec)
                gauge = build_gauge_state(problem.data, exponential_gauge([0.37], 2))
                decomposition = build_fixed_decomposition(problem, gauge, .03, .04, .1, shift=shift)
                expected = decomposition.denom+shift[0]*nelec[0]+shift[1]*nelec[1]
                self.assertAlmostEqual(decomposition.lambda_electronic, expected, places=12)
                basis = get_all_configs_u11((n, n), nelec)
                basis_map = {cf: i for i, cf in enumerate(basis)}
                H = physical2MB(h, eri, basis, basis_map)
                G = get_MB_gf(decomposition, basis, basis_map, n)
                check_norm(decomposition.lambda_electronic*np.eye(len(basis))-G, H)
        no_counts = prepare_problem(h, eri, chol)
        unshifted = build_fixed_decomposition(no_counts, build_gauge_state(no_counts.data), .03, .04)
        self.assertEqual(unshifted.lambda_electronic, unshifted.denom)

    def test_with_shift_keeps_two_body_and_weights(self):
        _, decomposition = fixture()
        shifted = with_shift(decomposition, (.05, .07))
        self.assertIs(shifted.two_body, decomposition.two_body)
        np.testing.assert_array_equal(shifted.coefficients, decomposition.coefficients)
        self.assertEqual(shifted.denom, decomposition.denom)
        self.assertEqual(shifted.shift, (.05, .07))


class ShiftProfileTests(unittest.TestCase):
    def test_loss_is_exactly_quadratic_and_the_spins_decouple(self):
        for loss_kind in LOSSES:
            with self.subTest(loss_kind=loss_kind):
                _, decomposition = fixture()
                profiles = shift_coefficients(decomposition, loss_kind=loss_kind)
                base = objective_loss(decomposition, loss_kind=loss_kind)
                for p in profiles:
                    for m in (-.4, .1, .35):
                        shift = [0., 0.]
                        shift[p.spin] = m
                        np.testing.assert_allclose(loss_at(decomposition, shift, loss_kind)-base,
                                                   p.change(m), rtol=1e-9, atol=1e-13*base)
                    h = 1e-4
                    plus, minus = [0., 0.], [0., 0.]
                    plus[p.spin], minus[p.spin] = h, -h
                    fd = (loss_at(decomposition, plus, loss_kind)-loss_at(decomposition, minus, loss_kind))/(2*h)
                    np.testing.assert_allclose(fd, p.gradient(), rtol=1e-7)
                # No mixed mu_up/mu_down term.
                a, b = .2, -.15
                mixed = (loss_at(decomposition, (a, b), loss_kind)-loss_at(decomposition, (a, 0.), loss_kind)
                         - loss_at(decomposition, (0., b), loss_kind)+base)
                self.assertLess(abs(mixed), 1e-12*base)

    def test_closed_form_minimizer_is_a_weighted_mean_of_orbital_energies(self):
        for loss_kind in LOSSES:
            for nelec in ((2, 1), (1, 1), (2, 0)):
                with self.subTest(loss_kind=loss_kind, nelec=nelec):
                    _, decomposition = fixture(nelec)
                    eps = decomposition.one_body.eigenvalues
                    best = optimal_shift(decomposition, loss_kind=loss_kind)
                    for p in shift_coefficients(decomposition, loss_kind=loss_kind):
                        self.assertGreaterEqual(best[p.spin], eps.min()-1e-14)
                        self.assertLessEqual(best[p.spin], eps.max()+1e-14)

                        def f(m, spin=p.spin):
                            shift = list(decomposition.shift)
                            shift[spin] = m
                            return loss_at(decomposition, shift, loss_kind)
                        brute = minimize_scalar(f, bounds=(eps.min(), eps.max()), method="bounded",
                                                options={"xatol": 1e-10}).x
                        self.assertAlmostEqual(best[p.spin], brute, delta=1e-7)
                    if nelec[1] == 0:
                        self.assertEqual(best[1], 0.)

    def test_two_orbitals_give_the_mean_orbital_energy_at_any_gauge_and_partition(self):
        problem = two_orbital_problem()
        rng = np.random.default_rng(9)
        for loss_kind in LOSSES:
            engine = GaugeEvaluator(problem, tau1=.05, tau2=.05, loss_kind=loss_kind)
            mean = float(np.mean(engine.one_body.eigenvalues))
            for _ in range(3):
                K = rng.normal(size=(3, 3))
                decomposition = engine.decomposition(expm(.5*(K-K.T)),
                                                     t=rng.normal(size=len(engine.sources))*.3)
                np.testing.assert_allclose(optimal_shift(decomposition, loss_kind=loss_kind), (mean, mean),
                                           rtol=0, atol=1e-13)

    def test_spin_shifts_differ_only_when_the_channels_break_spin_symmetry(self):
        problem, decomposition = fixture((2, 1))
        diagonal = np.array([s.modes[0] == s.modes[1] for s in decomposition.two_body.sources])
        symmetric = np.where(diagonal, 0., decomposition.two_body.t)
        mu = optimal_shift(with_partitions_t(decomposition, symmetric), loss_kind="weighted")
        self.assertAlmostEqual(mu[0], mu[1], delta=1e-13)
        mu = optimal_shift(decomposition, loss_kind="weighted")  # nonzero diagonal-source ratios
        self.assertGreater(abs(mu[0]-mu[1]), 1e-6)
        _, ab = fixture((1, 1))
        mu = optimal_shift(ab, loss_kind="weighted")
        self.assertGreater(abs(mu[0]-mu[1]), 1e-6)


def with_partitions_t(decomposition, t):
    from ipie.hamiltonians.optimize_cholesky import with_partitions
    return with_partitions(decomposition, t)


class ShiftRelaxationTests(unittest.TestCase):
    def test_inner_solve_starts_at_mu_star_and_returns_a_mutually_optimal_pair(self):
        for loss_kind in LOSSES:
            with self.subTest(loss_kind=loss_kind):
                _, decomposition = fixture()
                result = optimize_partitions(decomposition, loss_kind=loss_kind, optimize_shift=True,
                                             gradient_atol=1e-12, gradient_rtol=0, max_sweeps=500)
                self.assertTrue(result.converged)
                self.assertTrue(result.optimize_shift)
                self.assertLessEqual(result.initial_loss, result.input_loss)
                self.assertEqual(result.history[0].shift_change, 0.)
                self.assertLessEqual(result.history[-1].shift_change, result.shift_tolerance)
                losses = [row.loss for row in result.history]
                self.assertTrue(all(b <= a*(1+1e-14) for a, b in zip(losses, losses[1:])))
                np.testing.assert_allclose(result.shift, optimal_shift(result.decomposition, loss_kind=loss_kind),
                                           rtol=0, atol=1e-14)
                np.testing.assert_allclose(result.shift_gradient, 0., atol=1e-12*result.loss)
                np.testing.assert_allclose(result.loss, objective_loss(result.decomposition, loss_kind=loss_kind),
                                           rtol=1e-12)
                eps = decomposition.one_body.eigenvalues
                self.assertAlmostEqual(result.shift_tolerance, 1e-8*(eps.max()-eps.min()))

    def test_the_input_shift_is_not_a_seed(self):
        _, decomposition = fixture()
        first = optimize_partitions(decomposition, loss_kind="weighted", optimize_shift=True, max_sweeps=500)
        second = optimize_partitions(with_shift(decomposition, (.2, -.1)), loss_kind="weighted",
                                     optimize_shift=True, max_sweeps=500)
        self.assertEqual(first.sweeps, second.sweeps)
        np.testing.assert_allclose(first.t, second.t, rtol=0, atol=1e-14)
        np.testing.assert_allclose(first.shift, second.shift, rtol=0, atol=1e-14)
        self.assertNotEqual(first.input_loss, second.input_loss)
        self.assertAlmostEqual(first.initial_loss, second.initial_loss, delta=1e-14*first.loss)

    def test_lower_bound_clips_and_the_wall_filters_the_gradient(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        gauge = build_gauge_state(problem.data, exponential_gauge([0.37], 2))
        decomposition = build_fixed_decomposition(problem, gauge, 1.0, .04, .1)
        lower = shift_bounds(decomposition.one_body, .05)
        result = optimize_partitions(decomposition, optimize_shift=True, one_body_limit=.05,
                                     gradient_atol=1e-12, gradient_rtol=0, max_sweeps=500)
        self.assertTrue(result.converged)
        self.assertEqual(result.shift, lower)
        self.assertEqual(result.shift_gradient, (0., 0.))
        for p in shift_coefficients(result.decomposition, one_body_limit=.05):
            self.assertGreater(p.gradient(), 0.)       # descent would leave the domain
            self.assertLess(p.optimum, p.lower)
        self.assertTrue(all(c.d >= -.05-1e-15 for c in result.decomposition.one_body.channels))

    def test_requires_particle_numbers(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol)
        decomposition = build_fixed_decomposition(problem, build_gauge_state(problem.data), .03, .04)
        with self.assertRaises(ValueError):
            optimize_partitions(decomposition, optimize_shift=True)

    def test_up_only_sector_relaxes_only_mu_up(self):
        _, decomposition = fixture((2, 0))
        result = optimize_partitions(decomposition, loss_kind="weighted", optimize_shift=True, max_sweeps=500)
        self.assertTrue(result.converged)
        self.assertIsNone(result.shift_bounds[1])
        self.assertEqual(result.shift[1], 0.)
        self.assertNotEqual(result.shift[0], 0.)

    def test_relaxed_gradient_with_relaxed_shift_matches_central_differences(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        O = exponential_gauge([0.37], 2)
        K = np.array([[0., 1.], [-1., 0.]])
        check = check_gauge_direction(
            lambda X: engine.relaxed(X, gradient_atol=1e-13, gradient_rtol=0, optimize_shift=True),
            O, K, steps=(1e-4, 1e-5))
        self.assertLess(check.differences[-1].relative_error, 1e-6)

    def test_a_stale_shift_is_rejected_by_the_relaxed_gradient(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        good = engine.relaxed(exponential_gauge([0.37], 2), optimize_shift=True).inner_result
        stale = with_shift(good.decomposition, tuple(mu+.01 for mu in good.shift))
        bad = replace(good, decomposition=stale, loss=objective_loss(stale, loss_kind="weighted"))
        with self.assertRaises(InnerConvergenceError):
            relaxed_gauge_gradient(problem, bad)

    def test_outer_run_reports_the_relaxed_shift(self):
        h, eri, chol = example_tensors()
        problem = prepare_problem(h, eri, chol, nelec=(2, 1))
        engine = GaugeEvaluator(problem, tau1=.03, tau2=.04, loss_kind="weighted")
        result = optimize_gauge(engine, options=LBFGSOptions(max_iterations=2),
                                inner_options=InnerOptions(optimize_shift=True))
        self.assertLess(result.loss, result.initial_loss)
        self.assertEqual(result.history[-1].shift, result.decomposition.shift)
        self.assertTrue(result.partition_result.optimize_shift)
        self.assertTrue(all(m.passed for m in result.validation_report.checks.values()))
        legacy = optimize_gauge(GaugeEvaluator(problem, tau1=.03, tau2=.04),
                                options=LBFGSOptions(max_iterations=1))
        self.assertIsNone(legacy.history[-1].shift)

    def test_inner_options_validate_shift_settings(self):
        for kwargs in ({"optimize_shift": 1}, {"one_body_limit": 1.}, {"one_body_limit": 0.},
                       {"shift_atol": -1.}, {"shift_rtol": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                InnerOptions(**kwargs)


if __name__ == "__main__":
    unittest.main()
