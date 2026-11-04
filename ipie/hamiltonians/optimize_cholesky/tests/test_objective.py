"""Dense/overlap parity and direct agreement with the real sor_base function.

Only that reference function is AST-loaded, unchanged, to avoid unrelated
SCF/MPI imports. No numerical operation inside the reference is mocked.
"""

import ast
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from ipie.hamiltonians.optimize_cholesky import (
    Channel, FixedDecomposition, OverlapEvaluator, build_fixed_decomposition,
    build_gauge_state, commutator_loss, commutator_norm2, dense_commutator_loss,
    get_rotation_matrix, overlap_commutator_loss, prepare_problem, raw_loss,
)
from ipie.hamiltonians.optimize_cholesky.tests.test_channels import make_decomposition


def load_sor_reference():
    """Read the current checked-out compute_total_commutativity, not a copy."""
    source_path = Path(__file__).resolve().parents[2] / "sor_base.py"
    tree = ast.parse(source_path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "compute_total_commutativity")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["compute_total_commutativity"]


class SORReferenceView:
    """Read-only view exposing exactly the three fields the reference needs."""

    def __init__(self, decomposition, normalize_weights=True):
        self.decomposition = decomposition
        self.nterms = decomposition.nterms
        self.a = decomposition.coefficients.copy()
        if normalize_weights:
            self.a /= decomposition.denom

    def get_rotation_matrix(self, index):
        return get_rotation_matrix(self.decomposition, index)


def sor_diagnostics(decomposition, normalize_weights=True):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        load_sor_reference()(SORReferenceView(decomposition, normalize_weights))
    result = {}
    for line in buffer.getvalue().splitlines():
        label, value = line.split("=", 1)
        result[label.strip()] = float(value)
    return result


def dense_pair(decomposition, i, j):
    result = 0.0
    for left, right in zip(get_rotation_matrix(decomposition, i),
                           get_rotation_matrix(decomposition, j)):
        if left is not None and right is not None:
            result += np.linalg.norm(left @ right - right @ left, "fro")**2
    return float(result)


def with_two_channels(decomposition, channels):
    return FixedDecomposition(replace(decomposition.one_body, channels=()),
                              replace(decomposition.two_body, channels=tuple(channels)))


class KernelTests(unittest.TestCase):
    def test_random_orthonormal_subspaces_and_signed_coefficients(self):
        rng = np.random.default_rng(83)
        for left_rank in (1,2):
            for right_rank in (1,2):
                for _ in range(5):
                    V = np.linalg.qr(rng.normal(size=(7,left_rank)))[0]
                    Z = np.linalg.qr(rng.normal(size=(7,right_rank)))[0]
                    d, e = rng.normal(size=left_rank), rng.normal(size=right_rank)
                    A, B = (V*d) @ V.T, (Z*e) @ Z.T
                    expected = np.linalg.norm(A @ B - B @ A, "fro")**2
                    np.testing.assert_allclose(commutator_norm2(d,e,V.T @ Z), expected,
                                                atol=2e-13, rtol=2e-12)

    def test_rank_one_analytic_identity_and_empty_blocks(self):
        for cosine in (0.,0.2,-0.7,1.-1e-10,1.):
            d, e = .2, -.3
            expected = 2*d*d*e*e*cosine*cosine*(1-cosine*cosine)
            np.testing.assert_allclose(commutator_norm2([d],[e],[[cosine]]), expected,
                                        atol=1e-17, rtol=1e-6)
        self.assertEqual(commutator_norm2([], [.2], np.empty((0,1))),0)
        self.assertEqual(commutator_norm2([], [], np.empty((0,0))),0)

    def test_negative_roundoff_is_distinct_from_invalid_overlaps(self):
        self.assertEqual(commutator_norm2([1.],[1.],[[1.+np.finfo(float).eps]]),0.)
        for d,e in ((1.,1.), (1e-8,2e-8)):
            with self.assertRaises(FloatingPointError):
                commutator_norm2([d],[e],[[1.1]])

    def test_invalid_kernel_inputs(self):
        for args in (([1.],[1.],[[1.,2.]]), ([[1.]],[1.],[[1.]]),
                     ([1.],[1.],[[np.nan]]), ([1j],[1.],[[.1]])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                commutator_norm2(*args)


class TotalLossTests(unittest.TestCase):
    def assert_results_equal(self, left, right):
        for name in ("comm2","weighted_comm2","weight"):
            np.testing.assert_allclose(getattr(left,name),getattr(right,name),
                                        atol=2e-14,rtol=2e-11,err_msg=name)
        if left.weighted_mean is None:
            self.assertIsNone(right.weighted_mean)
        else:
            np.testing.assert_allclose(left.weighted_mean,right.weighted_mean,atol=2e-14,rtol=2e-11)
        self.assertEqual(left.pair_count,right.pair_count)

    def test_three_policies_partitions_gauges_and_actual_sor_function(self):
        for nelec in ((2,1),(1,1),(2,0)):
            for partition in ("balanced_log","anchored_log"):
                for rotated in (False,True):
                    with self.subTest(nelec=nelec,partition=partition,rotated=rotated):
                        _, decomposition = make_decomposition(nelec,partition=partition,rotated=rotated)
                        dense = dense_commutator_loss(decomposition)
                        fast = overlap_commutator_loss(decomposition)
                        self.assert_results_equal(dense,fast)
                        reference = sor_diagnostics(decomposition)
                        for name in ("comm2","weighted_comm2","weight"):
                            np.testing.assert_allclose(getattr(fast,name),reference[name],
                                                        atol=2e-14,rtol=2e-11)
                        np.testing.assert_allclose(fast.weighted_mean,
                            reference["weighted_comm2/weight"],atol=2e-14,rtol=2e-11)
                        self.assertEqual(raw_loss(decomposition),fast.comm2)
                        json.dumps(fast.as_dict(),allow_nan=False)

    def test_every_pair_and_cache_without_dense_matrix_calls(self):
        _, decomposition = make_decomposition((2,1),partition="anchored_log")
        evaluator = OverlapEvaluator(decomposition)
        n = decomposition.nterms
        for i in range(n):
            for j in range(i+1,n):
                np.testing.assert_allclose(evaluator.pair_norm2(i,j),dense_pair(decomposition,i,j),
                                            atol=2e-15,rtol=2e-10)
                np.testing.assert_allclose(evaluator.pair_norm2(i,j),evaluator.pair_norm2(j,i),
                                            atol=2e-15,rtol=2e-10)
        r = decomposition.two_body.gauge.X.shape[0]
        self.assertLessEqual(evaluator.overlap_cache_size,r*(r+1)//2)
        with patch("ipie.hamiltonians.optimize_cholesky.objective.get_rotation_matrix",
                   side_effect=AssertionError("overlap evaluator must not build dense rotations")):
            self.assertGreater(evaluator.evaluate().comm2,0)

    def test_opposite_spin_order_and_quadrature(self):
        _, base = make_decomposition((2,1))
        c1=Channel(0,(0,1),1.,(0,1),(.2,-.3))
        c2=Channel(1,(1,0),2.,(1,2),(.5,.4))
        decomposition=with_two_channels(base,[c1,c2])
        X=base.two_body.gauge.X
        up=float(X[0,:,0] @ X[1,:,2]);down=float(X[0,:,1] @ X[1,:,1])
        expected=2*(.2*.4)**2*up**2*(1-up**2)+2*(-.3*.5)**2*down**2*(1-down**2)
        evaluator=OverlapEvaluator(decomposition)
        np.testing.assert_allclose(evaluator.pair_norm2(0,1),expected,atol=1e-16)
        self.assert_results_equal(evaluator.evaluate(),dense_commutator_loss(decomposition))
        disjoint=with_two_channels(base,[Channel(0,(0,0),1.,(0,1),(.2,.3)),
                                         Channel(1,(1,1),2.,(0,1),(.4,.1))])
        self.assertEqual(overlap_commutator_loss(disjoint).comm2,0.)

    def test_same_basis_channels_commute_and_repeated_modes_merge(self):
        _, base=make_decomposition((2,1))
        repeated=Channel(0,(0,0),1.,(0,0),(.2,.3))
        external=Channel(1,(0,0),1.,(1,2),(.1,-.2))
        decomposition=with_two_channels(base,[repeated,external])
        evaluator=OverlapEvaluator(decomposition)
        np.testing.assert_allclose(evaluator.blocks[0][0].kappa,[.2+.3+.2*.3])
        self.assert_results_equal(evaluator.evaluate(),dense_commutator_loss(decomposition))
        same_basis=with_two_channels(base,[repeated,replace(external,bix=0)])
        self.assertEqual(overlap_commutator_loss(same_basis).comm2,0.)

    def test_zero_channels_keep_all_pair_weights(self):
        h=np.zeros((2,2));chol=np.zeros((2,2,2));eri=np.zeros((2,2,2,2))
        p=prepare_problem(h,eri,chol,nelec=(1,1))
        decomposition=build_fixed_decomposition(p,build_gauge_state(p.data),.03,.04)
        result=overlap_commutator_loss(decomposition)
        self.assertEqual(result.comm2,0.)
        self.assertEqual(result.weighted_mean,0.)
        a=decomposition.coefficients/decomposition.denom
        expected_weight=.5*(a.sum()**2-np.dot(a,a))
        self.assertAlmostEqual(result.weight,expected_weight)
        self.assertEqual(result.pair_count,decomposition.nterms*(decomposition.nterms-1)//2)
        self.assert_results_equal(result,dense_commutator_loss(decomposition))

    def test_single_channel_has_an_explicit_undefined_weighted_mean(self):
        h=np.array([[.2]]);chol=np.zeros((1,1,1));eri=np.zeros((1,1,1,1))
        p=prepare_problem(h,eri,chol,nelec=(1,0))
        decomposition=build_fixed_decomposition(p,build_gauge_state(p.data),.1,.1)
        result=overlap_commutator_loss(decomposition)
        self.assertEqual(result.comm2,0.)
        self.assertEqual(result.weight,0.)
        self.assertIsNone(result.weighted_mean)
        self.assertEqual(result.pair_count,0)
        self.assert_results_equal(result,dense_commutator_loss(decomposition))

    def test_zero_coupling_anchored_legs_and_both_partition_maps(self):
        h=np.array([[.2,.1],[.1,-.1]])
        chol=np.array([np.diag([0.,.7]),[[.3,.2],[.2,-.4]]])
        eri=np.einsum("gpr,gqs->prqs",chol,chol)
        for nelec in ((2,1),(1,1),(2,0)):
            p=prepare_problem(h,eri,chol,nelec=nelec)
            for partition in ("balanced_log","anchored_log"):
                d=build_fixed_decomposition(p,build_gauge_state(p.data),.03,.04,.17,
                                            partition=partition)
                self.assert_results_equal(dense_commutator_loss(d),overlap_commutator_loss(d))

    def test_coefficient_scaling_and_raw_weight_diagnostics(self):
        _, decomposition=make_decomposition((2,1))
        scale=7.
        changed=FixedDecomposition(
            replace(decomposition.one_body,channels=tuple(replace(c,a_i=scale*c.a_i)
                                                         for c in decomposition.one_body.channels)),
            replace(decomposition.two_body,channels=tuple(replace(c,a_i=scale*c.a_i)
                                                         for c in decomposition.two_body.channels)))
        original=commutator_loss(decomposition)
        self.assert_results_equal(original,commutator_loss(changed))
        raw=commutator_loss(decomposition,normalize_weights=False)
        scaled=commutator_loss(changed,normalize_weights=False)
        np.testing.assert_allclose(scaled.comm2,raw.comm2,atol=1e-15)
        np.testing.assert_allclose(scaled.weight,scale**2*raw.weight)
        np.testing.assert_allclose(scaled.weighted_comm2,scale**2*raw.weighted_comm2)
        np.testing.assert_allclose(scaled.weighted_mean,raw.weighted_mean)
        reference=sor_diagnostics(decomposition,normalize_weights=False)
        for name in ("comm2","weighted_comm2","weight"):
            np.testing.assert_allclose(getattr(raw,name),reference[name],rtol=2e-11,atol=2e-14)

    def test_bad_basis_method_and_indices(self):
        _, decomposition=make_decomposition((1,1))
        bad=replace(decomposition,one_body=replace(decomposition.one_body,X=2*decomposition.one_body.X))
        with self.assertRaises(ValueError):
            OverlapEvaluator(bad)
        with self.assertRaises(ValueError):
            commutator_loss(decomposition,method="unknown")
        with self.assertRaises(TypeError):
            commutator_loss(decomposition,normalize_weights="yes")
        evaluator=OverlapEvaluator(decomposition)
        with self.assertRaises(IndexError):
            evaluator.pair_norm2(-1,0)
        with self.assertRaises(TypeError):
            evaluator.pair_norm2(0.5,0)


def run_reference_checks():
    for nelec in ((2,1),(1,1),(2,0)):
        problem,decomposition=make_decomposition(nelec)
        dense=dense_commutator_loss(decomposition)
        fast=overlap_commutator_loss(decomposition)
        reference=sor_diagnostics(decomposition)
        print(f"{problem.policy:8s} raw_dense={dense.comm2:.12e} raw_overlap={fast.comm2:.12e} "
              f"raw_sor={reference['comm2']:.12e} diff={abs(dense.comm2-fast.comm2):.3e} "
              f"weighted_mean={fast.weighted_mean:.12e}")


if __name__ == "__main__":
    run_reference_checks()
    unittest.main()

