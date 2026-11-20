"""Golden-data regression tests against the frozen commit 3680f29.

reference/golden_3680f29.npz holds numbers produced by the package at the
frozen commit (and, for H2/cc-pVDZ, by the frozen oracle). Every recorded
evaluation is replayed here through the public entry points and must agree to
a relative 1e-12:

  * PackageGoldenTests: the current package (every kernel GaugeEvaluator
    accepts, if it has a kernel= option) reproduces the golden numbers;
  * OracleGoldenTests: reference/reference_kernels.py still reproduces them,
    i.e. the frozen oracle has not drifted.

Comparison: arrays and scalars by max|a-b| <= 1e-12*max|reference|; t and mu
by max|a-b| <= 1e-12*max(1,|reference|); converged flags exactly. Outer L-BFGS
trajectories (5 iterations) use the same rule; the iteration counts and status
must match exactly.

Large cases take minutes (LiH/STO-3G) to hours (H2/cc-pVDZ) with pair-loop
kernels: the package's "reference" kernel, a package without a kernel option,
and the oracle. They run only with OPTIMIZE_CHOLESKY_GOLDEN_LARGE=1;
fast kernels always run them. Whether the package default is a pair-loop
kernel is detected (PAIR_LOOP_KERNELS), so from step 3 on the large cases run
by default. Like the other tests of this package
they carry no pytest marker, so run them with

    python -m pytest ipie/hamiltonians/optimize_cholesky/tests/test_reference_golden.py -m ""
"""

import inspect
import json
import os
import unittest

import numpy as np

from ipie.hamiltonians.optimize_cholesky.tests.reference import reference_kernels as rk
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import GOLDEN_PATH, molecule, synthetic

RTOL = 1e-12
LARGE = os.environ.get("OPTIMIZE_CHOLESKY_GOLDEN_LARGE", "") not in ("", "0")
LARGE_FOR_PACKAGE = ("h2_ccpvdz", "lih_sto3g")
def _package_has_pair_loops():
    """True while the package's default loss still loops over channel pairs (before step 3)."""
    from ipie.hamiltonians.optimize_cholesky import loss
    return hasattr(loss, "OverlapEvaluator")


# Kernels for which LARGE_FOR_PACKAGE is gated: an explicit "reference" kernel, and the
# package default (None) while it still uses pair loops.
PAIR_LOOP_KERNELS = (None, "reference") if _package_has_pair_loops() else ("reference",)
LARGE_FOR_ORACLE = ("h2_ccpvdz", "lih_sto3g")


def load_golden():
    with np.load(GOLDEN_PATH) as data:
        values = {key: data[key] for key in data.files}
    metadata = json.loads(str(values.pop("metadata")))
    return values, metadata


GOLDEN, METADATA = load_golden()
CASES = METADATA["cases"]


def tensors(spec):
    if spec["source"] == "synthetic":
        return synthetic(spec["n"], spec["r"], spec["seed"])
    m = molecule(spec["system"])
    return m["h1"], m["eri"], m["chol"]


def case_values(name):
    prefix = f"{name}/"
    return {key[len(prefix):]: value for key, value in GOLDEN.items() if key.startswith(prefix)}


def shift_of(spec):
    return None if spec.get("shift") is None else tuple(spec["shift"])


class GoldenComparison:
    def assert_close(self, name, field, actual, expected):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        label = f"{name}/{field}"
        if expected.dtype.kind in "bU" or field.endswith(("converged", "iterations", "status")):
            self.assertEqual(actual.tolist(), expected.tolist(), label)
            return
        self.assertEqual(actual.shape, expected.shape, label)
        if expected.size == 0:
            return
        diff = float(np.max(np.abs(actual.astype(float)-expected.astype(float))))
        if field.endswith(("/t", "/shift")):
            scale = max(1.0, float(np.max(np.abs(expected))))
        else:
            scale = float(np.max(np.abs(expected)))
        self.assertLessEqual(diff, RTOL*scale, f"{label}: max|diff|={diff:.3e} > {RTOL:.0e}*{scale:.3e}")

    def compare(self, name, actual):
        expected = case_values(name)
        for field, value in actual.items():
            self.assertIn(field, expected, f"{name}/{field} is not in the golden file")
            self.assert_close(name, field, value, expected[field])
        return len(actual)


def package_kernels():
    from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
    parameters = inspect.signature(GaugeEvaluator.__init__).parameters
    if "kernel" not in parameters:
        return (None,)
    try:
        from ipie.hamiltonians.optimize_cholesky.gradients import KERNELS
        return tuple(KERNELS)
    except ImportError:
        return (parameters["kernel"].default,)


def package_replay(name, spec, kernel):
    """Re-run every recorded evaluation of one case with the package's public entry points."""
    from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
    from ipie.hamiltonians.optimize_cholesky.loss import objective_loss
    from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
    from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
    from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem

    h1, eri, chol = tensors(spec)
    problem = prepare_problem(h1, eri, chol, nelec=spec["nelec"])
    stored = case_values(name)
    extra = {} if kernel is None else {"kernel": kernel}
    is_h2 = "input/shift" in stored
    ev = GaugeEvaluator(problem, tau1=METADATA["tau1"], tau2=METADATA["tau2"], partition=spec["partition"],
                        loss_kind=spec["loss_kind"], shift=None if is_h2 else shift_of(spec), **extra)
    O, t = stored["input/O"], stored["input/t"]
    out = {}
    call_shift = tuple(stored["input/shift"]) if is_h2 else None
    g = ev.fixed(O, t=t, shift=call_shift)
    out.update({"fixed/loss": g.loss, "fixed/gradient_O": g.gradient_O,
                "fixed/partition_gradient": g.partition_gradient})
    if "fixed_z/loss" in stored:
        g = ev.fixed_z(O, z=stored["input/z"])
        out.update({"fixed_z/loss": g.loss, "fixed_z/gradient_O": g.gradient_O})
    if "objective/raw" in stored:
        decomposition = ev.decomposition(O, t=t)
        for kind in ("raw", "weighted"):
            out[f"objective/{kind}"] = objective_loss(decomposition, loss_kind=kind)
    for label, relax_shift in (("relaxed_fixed_shift", False), ("relaxed_shift", True)):
        if f"{label}/loss" not in stored:
            continue
        g = ev.relaxed(O, options=InnerOptions(**METADATA["inner"], optimize_shift=relax_shift))
        inner = g.inner_result
        out.update({f"{label}/loss": g.loss, f"{label}/gradient_O": g.gradient_O,
                    f"{label}/tangent_gradient": g.tangent_gradient, f"{label}/t": inner.t,
                    f"{label}/shift": np.asarray(inner.shift), f"{label}/converged": bool(inner.converged)})
    if "outer/loss" in stored:
        result = optimize_gauge(ev, theta0=stored["outer/theta0"], options=LBFGSOptions(**METADATA["outer"]),
                                inner_options=InnerOptions(**METADATA["inner"], optimize_shift=True))
        out.update({"outer/history_loss": np.array([row.loss for row in result.history]),
                    "outer/theta": result.theta, "outer/gradient_theta": result.gradient_theta,
                    "outer/loss": result.loss, "outer/iterations": int(result.iterations),
                    "outer/status": np.array(result.status)})
    return out


def oracle_replay(name, spec):
    """Re-run the recorded single evaluations with the frozen oracle (no outer runs)."""
    h1, _, chol = tensors(spec)
    stored = case_values(name)
    is_h2 = "input/shift" in stored
    oracle = rk.ReferenceOracle(h1, chol, spec["nelec"], tau1=METADATA["tau1"], tau2=METADATA["tau2"],
                                partition=spec["partition"], loss_kind=spec["loss_kind"],
                                shift=None if is_h2 else shift_of(spec))
    O, t = stored["input/O"], stored["input/t"]
    out = {}
    g = oracle.fixed(O, t=t, shift=tuple(stored["input/shift"]) if is_h2 else None)
    out.update({"fixed/loss": g["loss"], "fixed/gradient_O": g["gradient_O"],
                "fixed/partition_gradient": g["partition_gradient"]})
    if "fixed_z/loss" in stored:
        g = oracle.fixed_z(O, z=stored["input/z"])
        out.update({"fixed_z/loss": g["loss"], "fixed_z/gradient_O": g["gradient_O"]})
    if "objective/raw" in stored:
        for kind in ("raw", "weighted"):
            out[f"objective/{kind}"] = oracle.objective_loss(O, t=t, loss_kind=kind)
    for label, relax_shift in (("relaxed_fixed_shift", False), ("relaxed_shift", True)):
        if f"{label}/loss" not in stored:
            continue
        g = oracle.relaxed(O, inner=rk.InnerSettings(**METADATA["inner"], optimize_shift=relax_shift))
        out.update({f"{label}/loss": g["loss"], f"{label}/gradient_O": g["gradient_O"],
                    f"{label}/tangent_gradient": g["tangent_gradient"], f"{label}/t": g["t"],
                    f"{label}/shift": np.asarray(g["shift"]), f"{label}/converged": bool(g["converged"])})
    return out


def selected(kind, large_names):
    for name, spec in sorted(CASES.items()):
        if (spec["source"] == "synthetic") != (kind == "synthetic"):
            continue
        if name in large_names and not LARGE:
            continue
        yield name, spec


class GoldenFileTests(unittest.TestCase):
    def test_golden_file_is_from_the_frozen_commit(self):
        self.assertEqual(METADATA["frozen_commit"], rk.FROZEN_COMMIT)
        self.assertEqual(METADATA["recorded_at_head"], rk.FROZEN_COMMIT)
        self.assertEqual(sum(spec["source"] == "synthetic" for spec in CASES.values()), 24)
        self.assertTrue("h2_ccpvdz" in CASES, "golden file lacks the H2/cc-pVDZ case")


class PackageGoldenTests(GoldenComparison, unittest.TestCase):
    def _run(self, kind, large):
        for kernel in package_kernels():
            gated = large if kernel in PAIR_LOOP_KERNELS else ()
            for name, spec in selected(kind, gated):
                with self.subTest(case=name, kernel=kernel):
                    self.compare(name, package_replay(name, spec, kernel))

    def test_synthetic_cases(self):
        self._run("synthetic", LARGE_FOR_PACKAGE)

    def test_molecules(self):
        self._run("molecule", LARGE_FOR_PACKAGE)


class OracleGoldenTests(GoldenComparison, unittest.TestCase):
    def test_synthetic_cases(self):
        for name, spec in selected("synthetic", LARGE_FOR_ORACLE):
            with self.subTest(case=name):
                self.compare(name, oracle_replay(name, spec))

    def test_molecules(self):
        for name, spec in selected("molecule", LARGE_FOR_ORACLE):
            with self.subTest(case=name):
                self.compare(name, oracle_replay(name, spec))


if __name__ == "__main__":
    unittest.main()
