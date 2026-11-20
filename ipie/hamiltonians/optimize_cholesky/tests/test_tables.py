"""The array data model: tables, lazy record views, canonical vs caller-built decompositions.

Decompositions store sources, channels and partition profiles as read-only
arrays. Records (DensitySource, Channel, OneBodyChannel, PartitionCoefficients)
are views built on access and must equal what the 3680f29 object model held;
the frozen oracle in tests/reference supplies those reference records. The
optimizer's own paths never materialize Channel records.
"""

from dataclasses import replace
import itertools
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from ipie.hamiltonians.optimize_cholesky import models
from ipie.hamiltonians.optimize_cholesky.channels import (
    build_fixed_decomposition,
    canonical_table,
    enumerate_sources,
    with_channels,
)
from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.loss import objective_loss
from ipie.hamiltonians.optimize_cholesky.models import (
    Channel,
    ChannelTable,
    InnerOptions,
    OneBodyChannel,
    OneBodyTable,
    PartitionCoefficients,
    PartitionProfiles,
    SourceTable,
)
from ipie.hamiltonians.optimize_cholesky.partitions import (
    optimize_partitions,
    partition_coefficients,
    with_partitions,
)
from ipie.hamiltonians.optimize_cholesky.problem import (
    build_gauge_state,
    exponential_gauge,
    prepare_problem,
)
from ipie.hamiltonians.optimize_cholesky.tests.reference import reference_kernels as rk
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import synthetic
from ipie.hamiltonians.optimize_cholesky.tests.tolerance import RTOL, assert_close

SECTORS = ((2, 1), (1, 1), (2, 0))


def fixture(nelec, *, n=4, r=3, seed=5, partition="balanced_log", shift=None):
    h, eri, chol = synthetic(n, r, seed)
    problem = prepare_problem(h, eri, chol, nelec=nelec)
    O = exponential_gauge(np.random.default_rng(seed).normal(size=r*(r-1)//2), r)
    gauge = build_gauge_state(problem.data, O)
    count = len(enumerate_sources(problem))
    t = np.linspace(-.2, .3, count)
    if shift is None:
        shift = (.05, -.03) if nelec[1] else (.05, 0.)
    decomposition = build_fixed_decomposition(problem, gauge, .03, .04, t, partition=partition, shift=shift)
    oracle = rk.ReferenceOracle(h, chol, nelec, tau1=.03, tau2=.04, partition=partition, shift=shift)
    return problem, decomposition, oracle, O, t


class TableTests(unittest.TestCase):
    def test_tables_are_read_only_arrays(self):
        _, decomposition, _, _, _ = fixture((2, 1))
        one, two = decomposition.one_body, decomposition.two_body
        arrays = [getattr(two.table, f) for f in ("bix", "spins", "modes", "d", "a")]
        arrays += [getattr(one.table, f) for f in ("spin", "mode", "d", "a")]
        arrays += [getattr(two.sources, f) for f in ("bix", "modes", "kind", "spins", "slices", "channel_source",
                                                     "channel_sign", "channel_spins", "channel_weight")]
        for array in arrays:
            self.assertFalse(array.flags.writeable)
        self.assertIsInstance(two.table, ChannelTable)
        self.assertIsInstance(one.table, OneBodyTable)
        self.assertIsInstance(two.sources, SourceTable)
        self.assertIs(two.channels, two.table)
        self.assertIs(one.channels, one.table)

    def test_records_match_the_frozen_object_model(self):
        for nelec, partition in itertools.product(SECTORS, ("balanced_log", "anchored_log")):
            with self.subTest(nelec=nelec, partition=partition):
                _, decomposition, oracle, O, t = fixture(nelec, partition=partition)
                ref = oracle.decomposition(O, t=t)
                two, one = decomposition.two_body, decomposition.one_body
                self.assertEqual(len(two.sources), len(ref.two_body.sources))
                for mine, theirs in zip(two.sources, ref.two_body.sources):
                    self.assertEqual((mine.bix, mine.modes, mine.kind, mine.spins), tuple(theirs))
                self.assertEqual(two.source_slices.tolist(), [list(s) for s in ref.two_body.source_slices])
                # Topology exactly; weights and amplitudes to the 1e-12 standard.
                self.assertEqual(len(two.channels), len(ref.two_body.channels))
                for mine, theirs in zip(two.channels, ref.two_body.channels):
                    self.assertIsInstance(mine, Channel)
                    self.assertEqual((mine.bix, mine.spins, mine.modes), (theirs.bix, theirs.spins, theirs.modes))
                assert_close([c.a_i for c in two.channels], [c.a_i for c in ref.two_body.channels], label="a2")
                assert_close([c.d for c in two.channels], [c.d for c in ref.two_body.channels], label="d2")
                self.assertEqual(len(one.channels), len(ref.one_body.channels))
                for mine, theirs in zip(one.channels, ref.one_body.channels):
                    self.assertIsInstance(mine, OneBodyChannel)
                    self.assertEqual((mine.spin, mine.mode), (theirs.spin, theirs.mode))
                assert_close([c.a_i for c in one.channels], [c.a_i for c in ref.one_body.channels], label="a1")
                assert_close([c.d for c in one.channels], [c.d for c in ref.one_body.channels], label="d1")
                assert_close(decomposition.coefficients, rk.coefficients(ref), label="coefficients")
                assert_close(decomposition.denom, rk.denom(ref), label="denom")

    def test_sequence_protocol(self):
        _, decomposition, _, _, _ = fixture((2, 1))
        table = decomposition.two_body.channels
        records = tuple(table)
        self.assertEqual(len(records), len(table))
        self.assertEqual(table[-1], records[-1])
        self.assertEqual(table[np.int64(3)], records[3])
        self.assertEqual(table[2:7], records[2:7])
        self.assertEqual(table, records)
        self.assertNotEqual(table, records[::-1])
        self.assertEqual(table, canonical_table(decomposition.two_body.sources,
                                                decomposition.two_body.amplitudes, decomposition.two_body.tau2))
        with self.assertRaises(IndexError):
            table[len(table)]
        with self.assertRaises(TypeError):
            table[1.0]
        with self.assertRaises(TypeError):
            hash(table)

    def test_sources_compare_by_value(self):
        problem, decomposition, _, _, _ = fixture((2, 2))
        self.assertEqual(decomposition.two_body.sources, enumerate_sources(problem))
        other, _, _, _, _ = fixture((2, 0))
        self.assertNotEqual(enumerate_sources(other), enumerate_sources(problem))


class CanonicalTests(unittest.TestCase):
    def test_builders_and_partition_updates_are_canonical(self):
        _, decomposition, _, _, t = fixture((2, 1))
        self.assertTrue(decomposition.two_body.canonical)
        moved = with_partitions(decomposition, t + .01)
        self.assertTrue(moved.two_body.canonical)

    def test_with_channels_is_non_canonical_and_rejected_by_the_solver(self):
        _, decomposition, _, _, _ = fixture((2, 1))
        same = with_channels(decomposition, two_body=list(decomposition.two_body.channels))
        self.assertFalse(same.two_body.canonical)
        self.assertEqual(same.two_body.table, decomposition.two_body.table)
        np.testing.assert_allclose(objective_loss(same), objective_loss(decomposition), rtol=RTOL, atol=0)
        with self.assertRaisesRegex(ValueError, "not canonical"):
            optimize_partitions(same)
        only_one_body = with_channels(decomposition, one_body=list(decomposition.one_body.channels)[:3])
        self.assertTrue(only_one_body.two_body.canonical)
        self.assertEqual(len(only_one_body.one_body.channels), 3)

    def test_with_channels_validates_its_input(self):
        _, decomposition, _, _, _ = fixture((2, 1))
        good = decomposition.two_body.channels[0]
        n = decomposition.two_body.gauge.X.shape[1]
        r = decomposition.two_body.gauge.X.shape[0]
        for bad in (replace(good, bix=r), replace(good, modes=(0, n))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                with_channels(decomposition, two_body=[bad])
        with self.assertRaises(ValueError):
            Channel(0, (0, 2), 1., (0, 1), (.1, .2))
        with self.assertRaises(ValueError):
            Channel(0, (0, 1), 0., (0, 1), (.1, .2))
        with self.assertRaises(TypeError):
            with_channels(decomposition, two_body=[(0, (0, 1), 1., (0, 1), (.1, .2))])
        with self.assertRaises(ValueError):
            with_channels(decomposition, one_body=[OneBodyChannel(2, 1., 0, .1)])
        with self.assertRaises(ValueError):
            with_channels(decomposition, one_body=[OneBodyChannel(0, -1., 0, .1)])

    def test_only_canonical_decompositions_can_be_saved(self):
        try:
            import h5py  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("h5py is not installed")
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        _, decomposition, _, _, _ = fixture((2, 1))
        with tempfile.TemporaryDirectory() as folder:
            save_decomposition(os.path.join(folder, "ok.h5"), decomposition)
            with self.assertRaisesRegex(ValueError, "canonical"):
                save_decomposition(os.path.join(folder, "bad.h5"),
                                   with_channels(decomposition, two_body=list(decomposition.two_body.channels)))


class ProfileTests(unittest.TestCase):
    def test_result_profiles_are_arrays_with_record_views(self):
        _, decomposition, _, _, _ = fixture((2, 1))
        result = optimize_partitions(decomposition, InnerOptions(max_sweeps=3), loss_kind="weighted")
        profiles = result.coefficients
        self.assertIsInstance(profiles, PartitionProfiles)
        self.assertEqual(len(profiles), len(decomposition.two_body.sources))
        for field in ("t", "C", "left_term", "right_term"):
            self.assertFalse(getattr(profiles, field).flags.writeable)
        np.testing.assert_array_equal(profiles.gradient, result.gradient)
        record = profiles[4]
        self.assertIsInstance(record, PartitionCoefficients)
        self.assertEqual(record.source_index, 4)
        single = partition_coefficients(result.decomposition, 4)
        self.assertEqual((record.t, record.left_term, record.right_term),
                         (single.t, single.left_term, single.right_term))
        everything = partition_coefficients(result.decomposition)
        self.assertIsInstance(everything, PartitionProfiles)
        self.assertEqual(everything[4].left_term, single.left_term)


class HotPathTests(unittest.TestCase):
    def test_optimizer_paths_build_no_channel_records(self):
        problem, _, _, O, _ = fixture((2, 1))
        counts = {"two": 0, "one": 0}
        original_two, original_one = models.Channel.__post_init__, OneBodyChannel.__init__

        def counted_two(self):
            counts["two"] += 1
            original_two(self)

        def counted_one(self, *args, **kwargs):
            counts["one"] += 1
            original_one(self, *args, **kwargs)

        with mock.patch.object(models.Channel, "__post_init__", counted_two), \
                mock.patch.object(OneBodyChannel, "__init__", counted_one):
            ev = GaugeEvaluator(problem, tau1=.03, tau2=.04)
            ev.fixed(O, t=.1)
            ev.fixed_z(O, z=.2)
            ev.relaxed(O, options=InnerOptions(max_sweeps=50, optimize_shift=True))
        self.assertEqual(counts, {"two": 0, "one": 0})


if __name__ == "__main__":
    unittest.main()
