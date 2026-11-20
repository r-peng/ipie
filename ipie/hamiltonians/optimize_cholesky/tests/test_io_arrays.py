"""Step 6: save_decomposition reads the tables directly; files at the amplitude limits load.

The second test pins the parse_decomposition fix made in 2d8c568: until
then it asserted d >= -1 + thresh, which crashed for thresh=None and rejected
amplitudes sitting at their default limits (1 - 1e-8).
"""

from unittest import mock
import os
import tempfile
import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky import models
from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions
from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, prepare_problem
from ipie.hamiltonians.optimize_cholesky.shift import shift_bounds
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import synthetic

try:
    import h5py  # noqa: F401
    HAVE_H5PY = True
except ImportError:  # pragma: no cover
    HAVE_H5PY = False

POLICIES = {"hybrid": (2, 1), "ab_only": (1, 1), "aa_only": (2, 0)}
LIMIT = 1.0 - 1e-8                                   # default amplitude_limit and one_body_limit


def evaluator(nelec, partition="balanced_log"):
    h1, eri, chol = synthetic(4, 3, 5)
    problem = prepare_problem(h1, eri, chol, nelec=nelec)
    O = exponential_gauge(np.random.default_rng(5).normal(size=3), 3)
    return GaugeEvaluator(problem, tau1=.03, tau2=.04, partition=partition), O


def file_amplitudes(ham):
    """Every stored amplitude of a loaded QCSOR, one- and two-body."""
    values = [np.asarray(info["d"]).reshape(-1)
              for group in ham.size_groups.values() for info in group.term_info.values()]
    return np.concatenate(values)


@unittest.skipUnless(HAVE_H5PY, "h5py is required to write and read decomposition files")
class SaveTests(unittest.TestCase):
    def test_save_reads_the_tables_and_builds_no_records(self):
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        from ipie.hamiltonians.sor_base import QCSOR
        refuse = mock.Mock(side_effect=AssertionError("a per-record object was built while saving"))
        for policy, nelec in POLICIES.items():
            ev, O = evaluator(nelec)
            result = ev.relaxed(O, options=InnerOptions(optimize_shift=True))
            decomposition = result.decomposition
            for drop in (False, True):
                with self.subTest(policy=policy, drop_identity=drop), tempfile.TemporaryDirectory() as folder:
                    path = os.path.join(folder, "decomposition.h5")
                    with mock.patch.object(models.Channel, "__post_init__", refuse), \
                         mock.patch.object(models.OneBodyChannel, "__init__", refuse), \
                         mock.patch.object(models.DensitySource, "__init__", refuse):
                        save_decomposition(path, result, drop_identity=drop)
                    ham = QCSOR.from_file(path, thresh=None)
                    one, two = decomposition.one_body.table, decomposition.two_body.table
                    keep1 = np.ones(len(one), bool) if not drop else one.d != 0
                    keep2 = np.ones(len(two), bool) if not drop else ~np.all(two.d == 0, axis=1)
                    self.assertEqual(ham.nterms, int(keep1.sum() + keep2.sum()))
                    np.testing.assert_allclose(ham.denom, one.a[keep1].sum() + two.a[keep2].sum(), rtol=1e-13)


@unittest.skipUnless(HAVE_H5PY, "h5py is required to write and read decomposition files")
class AtTheLimitTests(unittest.TestCase):
    def test_decompositions_at_their_limits_load_with_the_default_thresh(self):
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        from ipie.hamiltonians.sor_base import QCSOR
        for policy, nelec in POLICIES.items():
            for partition in ("balanced_log", "anchored_log"):
                with self.subTest(policy=policy, partition=partition), tempfile.TemporaryDirectory() as folder:
                    ev, O = evaluator(nelec, partition)
                    lower = shift_bounds(ev.one_body)                  # mu at the one-body wall
                    shift = tuple(0. if low is None else low for low in lower)
                    # z = 1: every source at the upper end of its interval, i.e. its left leg at
                    # |d| = (1 - 1e-8) * amplitude_bound (none of these intervals is capped).
                    result = ev.fixed_z(O, z=1., shift=shift)
                    one, two = result.decomposition.one_body.table, result.decomposition.two_body.table
                    np.testing.assert_allclose(one.d.min(), -LIMIT, rtol=0, atol=1e-14)
                    np.testing.assert_allclose(np.abs(two.d).max(), LIMIT, rtol=0, atol=1e-14)
                    # Exactly what the old check `d < -1 + thresh` (thresh=1e-6) rejected:
                    self.assertTrue(one.d.min() < -1 + 1e-6 and two.d.min() < -1 + 1e-6)
                    path = os.path.join(folder, "decomposition.h5")
                    save_decomposition(path, result)
                    for thresh in (1e-6, None):                       # the default, and no truncation
                        ham = QCSOR.from_file(path, thresh=thresh)
                        d = file_amplitudes(ham)
                        self.assertLess(d.min(), -1 + 1e-6)           # the at-limit terms are kept
                        self.assertGreater(d.min(), -1.)
                    self.assertEqual(QCSOR.from_file(path, thresh=None).nterms, result.decomposition.nterms)


if __name__ == "__main__":
    unittest.main()
