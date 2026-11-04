"""Gauge derivatives: independent dense differences and re-relaxed trials."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from ipie.hamiltonians.optimize_cholesky import (
    GaugeEvaluator, GaugeRegularityError, GradientOptions, InnerConvergenceError,
    PartitionCoefficients, build_fixed_decomposition, build_gauge_state,
    check_gauge_direction, commutator_norm2, commutator_norm2_derivatives,
    eigh_vjp, exponential_gauge, exponential_gauge_vjp, fixed_partition_gradient,
    givens_gauge, givens_gauge_vjp, optimize_partitions, partition_coefficients,
    prepare_problem, raw_loss, relaxed_gauge_gradient,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import (
    check_hamiltonian, example_tensors, make_decomposition,
)


K2 = np.array([[0.,1.],[-1.,0.]])


def central(function, x, direction, h=1e-6):
    direction=np.asarray(direction)
    return (function(x+h*direction)-function(x-h*direction))/(2*h)


def fixture(nelec=(2,1),partition="balanced_log"):
    problem,decomposition = make_decomposition(nelec,partition=partition)
    evaluator = GaugeEvaluator(problem,tau1=.2,tau2=.04,partition=partition)
    return problem,decomposition.two_body.gauge.O,evaluator


def minimum_amplitude_limit(evaluator,O):
    state = evaluator._gauge(O)
    return max(np.sqrt(abs(state.eigenvalues[s.bix,s.modes[0]]*
                           state.eigenvalues[s.bix,s.modes[1]])*evaluator.tau2)
               for s in evaluator.sources)


class ElementaryDerivativeTests(unittest.TestCase):
    def test_overlap_polynomial_derivatives(self):
        rng = np.random.default_rng(291)
        for a,b in ((1,1),(1,2),(2,1),(2,2)):
            V = np.linalg.qr(rng.normal(size=(8,a)))[0]
            Z = np.linalg.qr(rng.normal(size=(8,b)))[0]
            d,e,C = rng.normal(size=a),rng.normal(size=b),V.T@Z
            value,gd,ge,gC = commutator_norm2_derivatives(d,e,C)
            self.assertAlmostEqual(value,commutator_norm2(d,e,C),places=14)
            for array,gradient,func in ((d,gd,lambda x:commutator_norm2(x,e,C)),
                                        (e,ge,lambda x:commutator_norm2(d,x,C)),
                                        (C,gC,lambda x:commutator_norm2(d,e,x))):
                direction = rng.normal(size=array.shape)
                np.testing.assert_allclose(np.sum(gradient*direction),central(func,array,direction),
                                           atol=2e-9,rtol=2e-8)

    def test_zero_leg_in_nonzero_block_has_a_derivative(self):
        d,e,C = np.array([0.,.3]),np.array([-.4]),np.array([[.2],[.5]])
        _,gd,_,_ = commutator_norm2_derivatives(d,e,C)
        self.assertGreater(abs(gd[0]),1e-5)
        np.testing.assert_allclose(gd[0],central(lambda x:commutator_norm2(x,e,C),d,[1.,0.]),
                                   atol=1e-11,rtol=2e-8)
        value,gd,ge,gC = commutator_norm2_derivatives([],[],np.empty((0,0)))
        self.assertEqual(value,0.)
        self.assertEqual(gd.size+ge.size+gC.size,0)

    def test_eigh_adjoint_against_continuously_aligned_eigenpairs(self):
        rng = np.random.default_rng(73)
        X = np.linalg.qr(rng.normal(size=(4,4)))[0]
        ell = np.array([-1.3,-.2,.4,1.1])
        L = (X*ell)@X.T
        ell,X = np.linalg.eigh(L)
        ge,gX = rng.normal(size=4),rng.normal(size=(4,4))
        derivative = eigh_vjp(ell,X,ge,gX)
        D = rng.normal(size=(4,4));D=(D+D.T)/2
        def value(A):
            values,vectors=np.linalg.eigh(A)
            vectors *= np.sign(np.sum(X*vectors,axis=0))[None,:]
            return ge@values+np.sum(gX*vectors)
        np.testing.assert_allclose(np.sum(derivative*D),central(value,L,D),atol=2e-9,rtol=2e-8)
        signs = np.array([1.,-1.,-1.,1.])
        np.testing.assert_allclose(eigh_vjp(ell,X*signs,ge,gX*signs),derivative,atol=1e-14)

    def test_eigh_degeneracies_and_invalid_options_are_explicit(self):
        for ell in ([.2,.2],[.2,.2+1e-13]):
            with self.assertRaises(GaugeRegularityError):
                eigh_vjp(ell,np.eye(2),np.ones(2),np.ones((2,2)))
        with self.assertRaises(ValueError):
            GradientOptions(gap_rtol=-1)
        with self.assertRaises(TypeError):
            eigh_vjp([.2,.4],np.eye(2),[1.,1.],np.eye(2),options={})

    def test_exponential_frechet_pullback_at_nonzero_coordinates(self):
        rng = np.random.default_rng(3)
        theta = np.array([.4,-.7,.6])
        E = rng.normal(size=(3,3))
        reference = np.diag([-1.,1.,1.])@exponential_gauge([.2,.3,-.1],3)
        gradient = exponential_gauge_vjp(theta,3,E,reference)
        for i in range(3):
            direction = np.eye(3)[i]
            fd = central(lambda x:np.sum(E*exponential_gauge(x,3,reference)),theta,direction)
            np.testing.assert_allclose(gradient[i],fd,atol=5e-10,rtol=2e-8)
        self.assertEqual(exponential_gauge_vjp([],1,[[.3]],reference=[[-1.]]).size,0)

    def test_ordered_givens_pullback_with_repeated_planes(self):
        rng = np.random.default_rng(33)
        planes = ((0,1),(1,2),(0,2),(0,1))
        angles = np.array([.3,-.4,.6,.2])
        reference = np.diag([-1.,1.,1.])
        E = rng.normal(size=(3,3))
        def rotations(values):
            return tuple((p,q,x) for (p,q),x in zip(planes,values))
        gradient = givens_gauge_vjp(3,rotations(angles),E,reference)
        for i in range(4):
            fd = central(lambda x:np.sum(E*givens_gauge(3,rotations(x),reference)),angles,np.eye(4)[i])
            np.testing.assert_allclose(gradient[i],fd,atol=5e-10,rtol=2e-8)

    def test_tiny_analytic_loss_changes_retain_the_decrease(self):
        c = PartitionCoefficients(0,0.,0.,1.,1.+2e-10)
        t = .25*(math.log(c.right_term)-math.log(c.left_term))
        self.assertLess(c.change(t),0.)
        expected = -(math.sqrt(c.left_term)-math.sqrt(c.right_term))**2
        np.testing.assert_allclose(c.change(t),expected,rtol=2e-6,atol=1e-28)


class FixedGaugeGradientTests(unittest.TestCase):
    def test_fixed_t_all_sectors_both_maps_against_dense_differences(self):
        for nelec in ((2,1),(1,1),(2,0)):
            for partition in ("balanced_log","anchored_log"):
                with self.subTest(nelec=nelec,partition=partition):
                    problem,state = make_decomposition(nelec,partition=partition)
                    result = fixed_partition_gradient(problem,state)
                    O = state.two_body.gauge.O
                    t = state.two_body.t
                    def dense(angle):
                        gauge = build_gauge_state(problem.data,expm(angle*K2)@O)
                        d = build_fixed_decomposition(problem,gauge,.03,.04,t,partition=partition)
                        return raw_loss(d,method="dense")
                    fd = (dense(1e-6)-dense(-1e-6))/(2e-6)
                    np.testing.assert_allclose(np.sum(result.gradient_O*(K2@O)),fd,atol=2e-11,rtol=2e-7)
                    np.testing.assert_allclose(result.loss,raw_loss(state,method="dense"),atol=3e-17,rtol=1e-11)
                    np.testing.assert_allclose(result.partition_gradient,
                                               [c.gradient for c in partition_coefficients(state)],atol=5e-18)
                    np.testing.assert_array_equal(result.bound_correction_O,0.)
                    self.assertEqual(result.mode,"fixed_t")

    def test_full_gradient_through_noncommuting_three_factor_coordinates(self):
        h,eri,chol = example_tensors(r=3)
        problem = prepare_problem(h,eri,chol,nelec=(2,1))
        evaluator = GaugeEvaluator(problem,tau1=.03,tau2=.04)
        theta = np.array([.17,-.24,.11])
        O = exponential_gauge(theta,3)
        t = np.linspace(-.1,.2,len(evaluator.sources))
        result = evaluator.fixed(O,t=t)
        grad = exponential_gauge_vjp(theta,3,result.gradient_O)
        for i in range(3):
            fd = central(lambda x:raw_loss(evaluator.decomposition(exponential_gauge(x,3),t=t),
                                            method="dense"),theta,np.eye(3)[i])
            np.testing.assert_allclose(grad[i],fd,atol=2e-10,rtol=3e-7)
        np.testing.assert_allclose(O.T@result.tangent_gradient+result.tangent_gradient.T@O,0.,atol=1e-16)
        local = result.gradient_O@O.T
        self.assertAlmostEqual(local[0,1]-local[1,0],2*result.generator_gradient[0,1],places=14)
        self.assertFalse(result.gradient_O.flags.writeable)
        self.assertFalse(result.eigenvalue_gradient.flags.writeable)

    def test_anchor_simple_zero_eigenvalue_and_balanced_cusp_guard(self):
        h=np.array([[.2,.1],[.1,-.1]])
        chol=np.array([np.diag([0.,.7]),[[.3,.2],[.2,-.4]]])
        eri=np.einsum("gpr,gqs->prqs",chol,chol)
        p=prepare_problem(h,eri,chol,nelec=(2,0))
        anchored=GaugeEvaluator(p,tau1=.1,tau2=.04,partition="anchored_log")
        check=check_gauge_direction(lambda O:anchored.fixed(O,t=.2),np.eye(2),K2,steps=(1e-5,1e-6))
        self.assertLess(check.differences[-1].absolute_error,1e-10)
        check=check_gauge_direction(lambda O:anchored.fixed_z(O,z=.3,max_abs_t=.4),
                                   np.eye(2),K2,steps=(1e-5,1e-6))
        self.assertLess(check.differences[-1].absolute_error,1e-10)
        balanced=GaugeEvaluator(p,tau1=.1,tau2=.04)
        with self.assertRaisesRegex(GaugeRegularityError,"zero/near-zero"):
            balanced.fixed()

    def test_no_ERI_reconstruction_or_dense_rotations_per_trial(self):
        _,O,evaluator=fixture((2,0))
        original=np.linalg.eigh
        with patch("numpy.linalg.eigh",wraps=original) as diagonalize, \
             patch("ipie.hamiltonians.optimize_cholesky.channels.validate_gauge_state",
                   side_effect=AssertionError("ERI reconstruction in a trial")), \
             patch("ipie.hamiltonians.optimize_cholesky.objective.get_rotation_matrix",
                   side_effect=AssertionError("dense channel construction in a trial")):
            evaluator.fixed(O,t=.1)
            self.assertEqual(diagonalize.call_count,1)

    def test_one_orbital_empty_partition_list(self):
        chol=np.array([[[.3]],[[.4]]])
        p=prepare_problem([[.2]],np.einsum("gpr,gqs->prqs",chol,chol),chol,nelec=(1,0))
        evaluator=GaugeEvaluator(p,tau1=.1,tau2=.1)
        for result in (evaluator.fixed(),evaluator.fixed_z(),evaluator.relaxed()):
            self.assertEqual(result.loss,0.)
            np.testing.assert_array_equal(result.gradient_O,0.)
            self.assertEqual(result.partition_gradient.size,0)


class MovingBoundTests(unittest.TestCase):
    def test_fixed_z_both_maps_with_amplitude_and_capped_bounds(self):
        for partition in ("balanced_log","anchored_log"):
            _,O,evaluator=fixture((2,1),partition)
            z=np.linspace(-.8,1.,len(evaluator.sources))
            mixed=1.025*minimum_amplitude_limit(evaluator,O)
            for rho,cap,moving in ((.25,30.,True),(.18,.3,False),(mixed,.3,True)):
                with self.subTest(partition=partition,rho=rho,cap=cap):
                    call=lambda matrix:evaluator.fixed_z(matrix,z=z,amplitude_limit=rho,max_abs_t=cap)
                    result=call(O)
                    check=check_gauge_direction(call,O,K2,steps=(1e-4,1e-5,1e-6))
                    self.assertLess(check.differences[-1].absolute_error,3e-10)
                    if moving:
                        self.assertGreater(np.linalg.norm(result.bound_correction_O),1e-7)
                    else:
                        np.testing.assert_array_equal(result.bound_correction_O,0.)
                    self.assertEqual(result.mode,"fixed_z")
                    self.assertLessEqual(np.max(np.abs(result.decomposition.two_body.amplitudes)),rho+2e-15)

    def test_active_bound_relaxed_gradients_all_sectors_both_maps(self):
        for nelec in ((2,0),(2,1),(1,1)):
            derivatives=[]
            for partition in ("balanced_log","anchored_log"):
                with self.subTest(nelec=nelec,partition=partition):
                    problem,O,evaluator=fixture(nelec,partition)
                    rho=1.025*minimum_amplitude_limit(evaluator,O)
                    center=evaluator.relaxed(O,amplitude_limit=rho,gradient_atol=1e-12,
                                             gradient_rtol=0,max_sweeps=200)
                    t=center.decomposition.two_body.t
                    bounds=center.inner_result.bounds
                    active=(t==bounds[:,0])|(t==bounds[:,1])
                    self.assertTrue(np.any(active))
                    self.assertGreater(np.max(np.abs(center.partition_gradient[active])),1e-7)
                    self.assertEqual(center.inner_result.amplitude_limit,rho)
                    h=1e-6
                    kwargs=dict(t0=t,amplitude_limit=rho,gradient_atol=1e-12,
                                gradient_rtol=0,max_sweeps=200)
                    plus=evaluator.relaxed(expm(h*K2)@O,**kwargs)
                    minus=evaluator.relaxed(expm(-h*K2)@O,**kwargs)
                    fd=(plus.loss-minus.loss)/(2*h)
                    total=np.sum(center.gradient_O*(K2@O))
                    fixed=np.sum(center.fixed_gradient_O*(K2@O))
                    np.testing.assert_allclose(total,fd,atol=1e-10,rtol=3e-7)
                    self.assertGreater(abs(fd-fixed),1e-6)
                    self.assertEqual(center.mode,"relaxed")
                    derivatives.append(total)
                    check_hamiltonian(center.decomposition,problem.data.h1,problem.data.eri,nelec)
            # Coordinate origins change; the relaxed physical derivative does not.
            np.testing.assert_allclose(*derivatives,atol=2e-12,rtol=2e-9)

    def test_interior_relaxed_gradient_and_envelope_term_vanishes(self):
        for partition in ("balanced_log","anchored_log"):
            _,O,evaluator=fixture((2,0),partition)
            center=evaluator.relaxed(O,gradient_atol=1e-13,gradient_rtol=0)
            self.assertLess(np.linalg.norm(center.bound_correction_O),2e-12)
            call=lambda matrix:evaluator.relaxed(matrix,t0=center.decomposition.two_body.t,
                                                 gradient_atol=1e-13,gradient_rtol=0)
            check=check_gauge_direction(call,O,K2,steps=(1e-4,1e-5,1e-6))
            self.assertLess(check.differences[-1].absolute_error,1e-10)

    def test_relaxed_constant_cap_has_no_bound_motion(self):
        _,O,evaluator=fixture((2,0),"anchored_log")
        center=evaluator.relaxed(O,max_abs_t=.05,gradient_atol=1e-13,gradient_rtol=0)
        t=center.decomposition.two_body.t
        self.assertTrue(np.any(np.abs(t)==.05))
        self.assertGreater(np.max(np.abs(center.partition_gradient)),1e-7)
        np.testing.assert_array_equal(center.bound_correction_O,0.)
        check=check_gauge_direction(
            lambda matrix:evaluator.relaxed(matrix,t0=t,max_abs_t=.05,
                                            gradient_atol=1e-13,gradient_rtol=0),
            O,K2,steps=(1e-5,1e-6))
        self.assertLess(check.differences[-1].absolute_error,1e-10)

    def test_two_orbital_balanced_shortcut_for_multiple_gauges(self):
        h,eri,chol=example_tensors(n=2,r=3,seed=37)
        p=prepare_problem(h,eri,chol,nelec=(1,1))
        evaluator=GaugeEvaluator(p,tau1=.1,tau2=.04)
        K=np.array([[0.,.2,-.3],[-.2,0.,.4],[.3,-.4,0.]])
        for theta in ((.1,.2,-.1),(-.2,.1,.3)):
            O=exponential_gauge(theta,3)
            fixed=evaluator.fixed(O,t=0.)
            relaxed=evaluator.relaxed(O)
            self.assertEqual(relaxed.inner_result.sweeps,0)
            np.testing.assert_allclose(fixed.gradient_O,relaxed.gradient_O,atol=2e-16,rtol=2e-12)
            check=check_gauge_direction(lambda matrix:evaluator.fixed(matrix,t=0.),O,K,steps=(1e-4,1e-5))
            self.assertLess(check.differences[-1].absolute_error,1e-10)

    def test_nonconverged_inner_solve_is_not_returned_as_a_relaxed_gradient(self):
        p,d=make_decomposition((2,0))
        incomplete=optimize_partitions(d,max_sweeps=0)
        with self.assertRaises(InnerConvergenceError) as context:
            relaxed_gauge_gradient(p,incomplete)
        self.assertIs(context.exception.result,incomplete)
        with self.assertRaises(InnerConvergenceError):
            relaxed_gauge_gradient(p,replace(incomplete,converged=True))
        evaluator=GaugeEvaluator(p,tau1=.2,tau2=.04)
        with self.assertRaises(InnerConvergenceError):
            evaluator.relaxed(d.two_body.gauge.O,max_sweeps=0)

    def test_bound_switch_and_collapsed_interval_are_explicit(self):
        _,O,evaluator=fixture((2,0))
        rho=1.2*minimum_amplitude_limit(evaluator,O)
        state=evaluator.decomposition(O)
        source=evaluator.sources[0]
        ell=state.two_body.gauge.eigenvalues[source.bix]
        base=np.sqrt(abs(ell[source.modes[0]]*ell[source.modes[1]])*.04)
        cap=math.log(rho/base)
        with self.assertRaisesRegex(GaugeRegularityError,"ties"):
            evaluator.fixed_z(O,z=.4,amplitude_limit=rho,max_abs_t=cap)
        rho=minimum_amplitude_limit(evaluator,O)
        with self.assertRaises(GaugeRegularityError):
            evaluator.fixed_z(O,z=0.,amplitude_limit=rho)
        with self.assertRaises(ValueError):
            evaluator.fixed_z(O,z=1.1)


if __name__ == "__main__":
    unittest.main()
