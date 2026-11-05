"""Partition profiles, sequential updates, KKT stopping and exact H checks.

The coefficient oracle below forms isolated dense leg commutators. It does
not use the overlap formula or infer coefficients by fitting loss values.
"""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar

from ipie.hamiltonians.optimize_cholesky import (
    PartitionCoefficients, analytic_partition_update, build_fixed_decomposition,
    build_gauge_state, get_rotation_matrix, optimize_partitions, partition_bounds,
    partition_coefficients, prepare_problem, projected_partition_gradient,
    raw_loss, with_partitions,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_hamiltonian, make_decomposition,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_objective import dense_pair, sor_diagnostics


def dense_coefficients(decomposition, source_index):
    two = decomposition.two_body
    n = two.gauge.X.shape[1]
    offset = len(decomposition.one_body.channels)
    start, stop = two.source_slices[source_index]
    affected = set(range(offset+start, offset+stop))
    A = B = cross = 0.0
    for i in affected:
        channel = two.channels[i-offset]
        legs = []
        for leg, (spin, mode, d) in enumerate(zip(channel.spins, channel.modes, channel.d)):
            vector = two.gauge.X[channel.bix,:,mode]
            amplitude = d*math.exp((-1 if leg == 0 else 1)*two.t[source_index])
            blocks = [np.zeros((n,n)), np.zeros((n,n))]
            blocks[spin] = amplitude*np.outer(vector, vector)
            legs.append(blocks)
        for j in range(decomposition.nterms):
            if j in affected:
                continue
            for spin, external in enumerate(get_rotation_matrix(decomposition, j)):
                if external is None:
                    continue
                left, right = legs[0][spin], legs[1][spin]
                M = left @ external-external @ left
                N = right @ external-external @ right
                A += float(np.sum(M*M))
                B += float(np.sum(N*N))
                cross += 2.0*float(np.sum(M*N))
    unaffected = math.fsum(dense_pair(decomposition,i,j)
                          for i in range(decomposition.nterms)
                          for j in range(i+1,decomposition.nterms)
                          if i not in affected and j not in affected)
    return A, B, unaffected+cross


def one_source(*, partition="balanced_log", eigenvalues=(-.5,.8), t=.2, nelec=(2,0)):
    h = np.array([[.2,.13],[.13,-.1]])
    chol = np.array([np.diag(eigenvalues)])
    eri = np.einsum("gpr,gqs->prqs",chol,chol)
    problem = prepare_problem(h,eri,chol,nelec=nelec)
    decomposition = build_fixed_decomposition(problem,build_gauge_state(problem.data),
                                              .1,.04,t,partition=partition)
    return problem,decomposition


class CoefficientTests(unittest.TestCase):
    def test_coefficients_profiles_and_gradients_against_dense_oracle(self):
        for nelec in ((2,1),(1,1),(2,0)):
            for partition in ("balanced_log","anchored_log"):
                _, decomposition = make_decomposition(nelec,partition=partition)
                coefficients = partition_coefficients(decomposition)
                self.assertEqual(len(coefficients),len(decomposition.two_body.sources))
                for i,c in enumerate(coefficients):
                    with self.subTest(nelec=nelec,partition=partition,source=i):
                        A,B,C = dense_coefficients(decomposition,i)
                        np.testing.assert_allclose([c.A,c.B,c.C],[A,B,C],atol=5e-17,rtol=2e-10)
                        self.assertGreaterEqual(c.A,0)
                        self.assertGreaterEqual(c.B,0)
                        for delta in (-.17,.12):
                            t = decomposition.two_body.t.copy()
                            t[i] += delta
                            dense = raw_loss(with_partitions(decomposition,t),method="dense")
                            np.testing.assert_allclose(c.value(t[i]),dense,atol=5e-17,rtol=3e-11)
                        h = 2e-5
                        t = decomposition.two_body.t.copy()
                        t[i] += h
                        plus = raw_loss(with_partitions(decomposition,t),method="dense")
                        t[i] -= 2*h
                        minus = raw_loss(with_partitions(decomposition,t),method="dense")
                        np.testing.assert_allclose(c.gradient,(plus-minus)/(2*h),atol=2e-12,rtol=3e-7)

    def test_C_is_allowed_to_be_negative_and_minimum_matches_scalar_solver(self):
        _,decomposition = one_source()
        c = partition_coefficients(decomposition,0)
        self.assertLess(c.C,0)
        bounds = partition_bounds(decomposition)[0]
        exact = analytic_partition_update(c,bounds)
        numerical = minimize_scalar(c.value,bounds=bounds,method="bounded",
                                    options={"xatol":1e-11})
        self.assertAlmostEqual(exact,0.0,places=12)
        self.assertAlmostEqual(exact,numerical.x,places=7)
        self.assertAlmostEqual(c.derivative(exact),0.0,places=14)
        self.assertLess(c.value(exact),c.value(c.t))

    def test_flat_and_one_sided_coordinate_minima(self):
        cases = ((0.,0.,.3),(0.,2.,.8),(3.,0.,-.7),(2.,8.,math.log(4)/4+.3))
        for left,right,expected in cases:
            c = PartitionCoefficients(0,.3,4.,left,right)
            self.assertAlmostEqual(analytic_partition_update(c,(-.7,.8)),expected,places=14)
        c = PartitionCoefficients(0,0.,1.,1e-200,1e200)
        self.assertEqual(analytic_partition_update(c,(-.2,.4)),.4)

    def test_centered_coefficients_do_not_require_a_representable_A(self):
        c = PartitionCoefficients(0,-400.,-1.,2.,3.)
        self.assertEqual(c.value(-400.),4.)
        self.assertEqual(c.gradient,-2.)
        self.assertTrue(np.isfinite(analytic_partition_update(c,(-401.,-399.))))
        with self.assertRaises(FloatingPointError):
            _ = c.A


class BoundsTests(unittest.TestCase):
    def test_closed_bounds_respect_both_maps_and_nonunit_open_bound(self):
        for partition in ("balanced_log","anchored_log"):
            _,base = make_decomposition((1,1),partition=partition)
            # Build at a smaller declared open bound, without changing t/d.
            decomposition = replace(base,two_body=replace(base.two_body,amplitude_bound=.8))
            bounds = partition_bounds(decomposition,amplitude_limit=.6,max_abs_t=5)
            self.assertFalse(bounds.flags.writeable)
            for side in (0,1):
                result = with_partitions(decomposition,bounds[:,side])
                self.assertLessEqual(np.max(np.abs(result.two_body.amplitudes)),.6+2e-16)
                self.assertLess(np.max(np.abs(result.two_body.amplitudes)),.8)

    def test_balanced_and_anchored_are_coordinate_translations_for_nonzero_W(self):
        _,balanced = make_decomposition((2,1),partition="balanced_log")
        _,anchored = make_decomposition((2,1),partition="anchored_log")
        shift = []
        for source in balanced.two_body.sources:
            l,r = (balanced.two_body.gauge.eigenvalues[source.bix,p] for p in source.modes)
            shift.append(.5*(math.log(abs(l))-math.log(abs(r))))
        shifted = with_partitions(anchored,balanced.two_body.t-np.array(shift))
        np.testing.assert_allclose(shifted.two_body.amplitudes,balanced.two_body.amplitudes,
                                   atol=2e-16,rtol=3e-15)
        np.testing.assert_allclose(partition_bounds(anchored)+np.array(shift)[:,None],
                                   partition_bounds(balanced),atol=1e-15)

    def test_projected_gradient_at_both_walls_and_fixed_coordinate(self):
        bounds = np.array([[-1.,1.]]*5+[[.2,.2]])
        result = projected_partition_gradient([-1.,1.,0.,-1.,1.,.2],
                                               [2.,-3.,4.,-5.,6.,7.],bounds)
        np.testing.assert_array_equal(result,[0.,0.,4.,-5.,6.,0.])

    def test_infeasible_bound_choices_are_explicit_errors(self):
        _,decomposition = one_source(t=.2)
        for options in ({"amplitude_limit":1.}, {"amplitude_limit":.01},
                        {"max_abs_t":0}, {"max_abs_t":np.inf}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                partition_bounds(decomposition,**options)
        with self.assertRaisesRegex(ValueError,"Initial t"):
            optimize_partitions(decomposition,max_abs_t=.1)


class InnerSolverTests(unittest.TestCase):
    def test_one_sweep_is_sequential_not_a_simultaneous_update(self):
        _,decomposition = make_decomposition((2,1),partition="anchored_log")
        bounds = partition_bounds(decomposition)
        initial_coefficients = partition_coefficients(decomposition)
        simultaneous = np.array([analytic_partition_update(c,bounds[i])
                                 for i,c in enumerate(initial_coefficients)])
        manual = decomposition
        losses = [raw_loss(manual)]
        for i in range(len(bounds)):
            c = partition_coefficients(manual,i)
            t = manual.two_body.t.copy()
            t[i] = analytic_partition_update(c,bounds[i])
            manual = with_partitions(manual,t)
            losses.append(raw_loss(manual))
        result = optimize_partitions(decomposition,max_sweeps=1,gradient_atol=0,gradient_rtol=0)
        np.testing.assert_allclose(result.t,manual.two_body.t,atol=2e-15,rtol=2e-14)
        self.assertGreater(np.max(np.abs(result.t-simultaneous)),1e-4)
        self.assertTrue(np.all(np.diff(losses) <= 2e-17))
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"max_sweeps")
        self.assertGreater(result.projected_gradient_norm,0)

    def test_three_sectors_both_maps_reconstruction_and_sor_parity(self):
        for nelec in ((2,1),(1,1),(2,0)):
            for partition in ("balanced_log","anchored_log"):
                with self.subTest(nelec=nelec,partition=partition):
                    problem,decomposition = make_decomposition(nelec,partition=partition)
                    original_t = decomposition.two_body.t.copy()
                    original_d = decomposition.two_body.amplitudes.copy()
                    result = optimize_partitions(decomposition,max_sweeps=5)
                    final = result.decomposition
                    self.assertLess(result.loss,result.initial_loss)
                    self.assertTrue(np.all(np.diff([h.loss for h in result.history]) <= 1e-16))
                    np.testing.assert_allclose(result.loss,raw_loss(final,method="dense"),
                                               atol=2e-17,rtol=2e-11)
                    np.testing.assert_allclose(result.loss,sor_diagnostics(final)["comm2"],
                                               atol=2e-17,rtol=2e-11)
                    check_hamiltonian(final,problem.data.h1,problem.data.eri,nelec,
                                      full_fock=(nelec == (2,1)))
                    np.testing.assert_array_equal(final.coefficients,decomposition.coefficients)
                    self.assertIs(final.two_body.gauge,decomposition.two_body.gauge)
                    self.assertIs(final.one_body,decomposition.one_body)
                    np.testing.assert_array_equal(decomposition.two_body.t,original_t)
                    np.testing.assert_array_equal(decomposition.two_body.amplitudes,original_d)
                    self.assertFalse(result.t.flags.writeable)
                    self.assertFalse(final.two_body.amplitudes.flags.writeable)
                    self.assertFalse(result.gradient.flags.writeable)
                    for source,(left,right) in zip(final.two_body.sources,final.two_body.amplitudes):
                        ell = final.two_body.gauge.eigenvalues[source.bix]
                        expected = -ell[source.modes[0]]*ell[source.modes[1]]*final.two_body.tau2
                        np.testing.assert_allclose(left*right,expected,atol=1e-18,rtol=3e-14)

    def test_convergence_and_warm_start_use_fresh_full_gradient(self):
        _,decomposition = make_decomposition((2,1))
        result = optimize_partitions(decomposition,max_sweeps=30)
        self.assertTrue(result.converged)
        self.assertEqual(result.status,"converged")
        final_coefficients = partition_coefficients(result.decomposition)
        expected_gradient = [c.gradient for c in final_coefficients]
        np.testing.assert_allclose(result.gradient,expected_gradient,atol=1e-18,rtol=3e-12)
        self.assertLessEqual(result.projected_gradient_norm,result.gradient_threshold)
        warm = optimize_partitions(result.decomposition,max_sweeps=0)
        self.assertTrue(warm.converged)
        self.assertEqual(warm.sweeps,0)

    def test_zero_coupling_anchored_active_wall_and_nonzero_raw_gradient(self):
        for eigenvalues,expected_wall in (((0.,.7),1),((-.7,0.),0)):
            problem,decomposition = one_source(partition="anchored_log",eigenvalues=eigenvalues,t=0)
            self.assertEqual(np.prod(decomposition.two_body.amplitudes[0]),0.)
            self.assertGreater(raw_loss(decomposition),0)
            result = optimize_partitions(decomposition,max_abs_t=.4,gradient_atol=1e-14,gradient_rtol=0)
            self.assertTrue(result.converged)
            self.assertEqual(result.sweeps,1)
            self.assertEqual(result.t[0],result.bounds[0,expected_wall])
            self.assertGreater(abs(result.gradient[0]),1e-9)
            self.assertEqual(result.projected_gradient_norm,0.)
            self.assertLess(result.loss,result.initial_loss)
            check_hamiltonian(result.decomposition,problem.data.h1,problem.data.eri,(2,0))

    def test_zero_balanced_flat_source_retains_its_partition(self):
        _,decomposition = one_source(eigenvalues=(0.,.7),t=2.1)
        result = optimize_partitions(decomposition)
        self.assertTrue(result.converged)
        self.assertEqual(result.sweeps,0)
        self.assertEqual(result.t[0],2.1)
        self.assertEqual(result.loss,0.)
        self.assertEqual(result.coefficients[0].A,0.)
        self.assertEqual(result.coefficients[0].B,0.)

    def test_no_two_body_sources_and_zero_hamiltonian(self):
        for n,nelec in ((1,(1,0)),(2,(1,1))):
            problem = prepare_problem(np.zeros((n,n)),np.zeros((n,n,n,n)),
                                      np.zeros((1,n,n)),nelec=nelec)
            decomposition = build_fixed_decomposition(problem,build_gauge_state(problem.data),.1,.1)
            result = optimize_partitions(decomposition)
            self.assertTrue(result.converged)
            self.assertEqual(result.loss,0.)
            self.assertEqual(result.sweeps,0)
            self.assertEqual(result.bounds.shape,(len(result.t),2))

    def test_no_dense_rotations_eigh_or_ERI_validation_inside_inner_solver(self):
        _,decomposition = make_decomposition((2,0))
        with patch("numpy.linalg.eigh",side_effect=AssertionError("eigh in inner loop")), \
             patch("ipie.hamiltonians.optimize_cholesky.channels.validate_gauge_state",
                   side_effect=AssertionError("ERI validation in inner loop")), \
             patch("ipie.hamiltonians.optimize_cholesky.objective.get_rotation_matrix",
                   side_effect=AssertionError("dense rotations in inner loop")):
            result = optimize_partitions(decomposition,max_sweeps=2)
        self.assertLess(result.loss,result.initial_loss)

    def test_max_sweeps_zero_and_invalid_inputs(self):
        _,decomposition = make_decomposition((2,1))
        result = optimize_partitions(decomposition,max_sweeps=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.status,"max_sweeps")
        np.testing.assert_array_equal(result.t,decomposition.two_body.t)
        self.assertEqual(result.initial_loss,result.loss)
        for kwargs in ({"max_sweeps":True},{"max_sweeps":-1},{"gradient_atol":-1},
                       {"gradient_rtol":np.nan},{"amplitude_limit":0}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                optimize_partitions(decomposition,**kwargs)
        with self.assertRaises(IndexError):
            partition_coefficients(decomposition,999)
        with self.assertRaises(TypeError):
            partition_coefficients(decomposition,True)
        two = decomposition.two_body
        bad = replace(decomposition,two_body=replace(two,channels=two.channels[::-1]))
        with self.assertRaisesRegex(ValueError,"Channels disagree"):
            optimize_partitions(bad)
        with self.assertRaises(ValueError):
            with_partitions(decomposition,999.)


if __name__ == "__main__":
    unittest.main()


