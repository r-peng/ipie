"""Timing benchmark for optimize_cholesky, to track every refactoring phase.

Run from the repository root (systems come from systems.npz; no PySCF needed):

    python ipie/hamiltonians/optimize_cholesky/tests/reference/benchmark.py
    python .../benchmark.py --systems h2_ccpvdz --outer 20 --json bench.json

For each system and each kernel the installed GaugeEvaluator accepts
(only its default if it has no kernel= option) it reports:

    prepare      prepare_problem (input validation)
    relaxed      one relaxed evaluation: inner solve + envelope gauge gradient
                 (median of --repeat runs at a fixed gauge), with loss and sweeps
    outer        --outer L-BFGS iterations of optimize_gauge (weak-Wolfe),
                 seconds per objective evaluation

Pair-loop kernels ("reference", or the default of a package that still has
pair loops, i.e. before step 3) are not run on systems whose estimated relaxed time exceeds --budget seconds; the
estimate (number of channel pairs times the measured cost of the frozen
oracle's pair kernel, times the passes of one relaxed evaluation) is printed
instead. Default systems: h2_ccpvdz (ab_only, the target size), h4_631g and
lih_sto3g (hybrid).
"""

import argparse
import inspect
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."]*5)))

from ipie.hamiltonians.optimize_cholesky.tests.reference import reference_kernels as rk  # noqa: E402
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import molecule, theta  # noqa: E402

TAU1, TAU2 = 0.03, 0.04
INNER = dict(max_sweeps=3000, gradient_atol=1e-10, gradient_rtol=1e-10, optimize_shift=True)


def kernels():
    from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
    parameters = inspect.signature(GaugeEvaluator.__init__).parameters
    if "kernel" not in parameters:
        return [None]
    try:
        from ipie.hamiltonians.optimize_cholesky.gradients import KERNELS
        return list(KERNELS)
    except ImportError:
        return [parameters["kernel"].default]


def package_has_pair_loops():
    """True while the package's default loss still loops over channel pairs (before step 3)."""
    from ipie.hamiltonians.optimize_cholesky import loss
    return hasattr(loss, "OverlapEvaluator")


def channel_count(policy, nchol, nbasis):
    sources = rk.enumerate_sources(policy, nchol, nbasis)
    return sum(len(rk._BRANCHES[s.kind]) for s in sources) + (1 if policy == "aa_only" else 2)*nbasis


def pair_kernel_seconds(samples=20000):
    """Cost of one frozen pair-kernel call (commutator_norm2 plus its derivatives)."""
    rng = np.random.default_rng(0)
    C = rng.normal(size=(1, 1))
    d, e = np.array([.3]), np.array([.2])
    start = time.perf_counter()
    for _ in range(samples):
        rk.commutator_norm2_derivatives(d, e, C)
    return (time.perf_counter()-start)/samples


def estimated_reference_seconds(policy, nchol, nbasis, kernel_seconds, sweeps=8):
    """Rough pair-loop cost of one relaxed evaluation: about (2*sweeps + 3) full pair passes."""
    T = channel_count(policy, nchol, nbasis)
    return (2*sweeps + 3) * 2*T*(T-1)/2 * kernel_seconds, T


def run_system(name, args, kernel_seconds):
    from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
    from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
    from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
    from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, prepare_problem

    m = molecule(name)
    chol = m["chol"]
    r, n = chol.shape[0], chol.shape[1]
    start = time.perf_counter()
    problem = prepare_problem(m["h1"], m["eri"], chol, nelec=m["nelec"])
    prepare = time.perf_counter()-start
    estimate, T = estimated_reference_seconds(problem.policy, r, n, kernel_seconds)
    O = exponential_gauge(theta(r, 0.1, 1), r)
    rows = []
    for kernel in kernels():
        row = dict(system=name, policy=problem.policy, n=n, nchol=r, channels=T, kernel=kernel,
                   prepare_s=prepare)
        pair_loops = kernel == "reference" or (kernel is None and package_has_pair_loops())
        if pair_loops and estimate > args.budget:
            row.update(skipped=True, estimated_relaxed_s=estimate)
            rows.append(row)
            continue
        extra = {} if kernel is None else {"kernel": kernel}
        ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, **extra)
        options = InnerOptions(**INNER)
        times, g = [], None
        for _ in range(args.repeat):
            start = time.perf_counter()
            g = ev.relaxed(O, options=options)
            times.append(time.perf_counter()-start)
        row.update(relaxed_s=statistics.median(times), loss=g.loss, sweeps=int(g.inner_result.sweeps))
        if args.outer:
            start = time.perf_counter()
            result = optimize_gauge(ev, theta0=theta(r, 0.1, 0),
                                    options=LBFGSOptions(max_iterations=args.outer, line_search="weak_wolfe",
                                                         validate_final=False),
                                    inner_options=options)
            seconds = time.perf_counter()-start
            row.update(outer_iterations=int(result.iterations), outer_evaluations=int(result.evaluations),
                       outer_s=seconds, outer_s_per_eval=seconds/max(1, result.evaluations),
                       outer_loss=(float(result.initial_loss), float(result.loss)))
        rows.append(row)
    return rows


def show(row):
    head = (f"{row['system']:10s} {row['policy']:7s} n={row['n']:2d} r={row['nchol']:3d} "
            f"channels={row['channels']:6d} kernel={str(row['kernel']):10s} prepare={row['prepare_s']:6.2f}s")
    if row.get("skipped"):
        return head + f"  skipped (estimated relaxed ~{row['estimated_relaxed_s']/60:.0f} min)"
    text = head + f"  relaxed={row['relaxed_s']:7.3f}s loss={row['loss']:.6e} sweeps={row['sweeps']}"
    if "outer_s" in row:
        text += (f"  outer: {row['outer_iterations']} it/{row['outer_evaluations']} evals in "
                 f"{row['outer_s']:.1f}s ({row['outer_s_per_eval']:.3f} s/eval), "
                 f"loss {row['outer_loss'][0]:.4e} -> {row['outer_loss'][1]:.4e}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", nargs="+", default=["h2_ccpvdz", "h4_631g", "lih_sto3g"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--outer", type=int, default=0, help="L-BFGS iterations to time (0: skip)")
    parser.add_argument("--budget", type=float, default=600.0,
                        help="skip pair-loop kernels whose estimated relaxed time exceeds this (s)")
    parser.add_argument("--json", help="append the rows to this JSON file")
    args = parser.parse_args()
    kernel_seconds = pair_kernel_seconds()
    print(f"frozen pair kernel: {kernel_seconds*1e6:.1f} us/call; kernels: {kernels()}")
    rows = []
    for name in args.systems:
        for row in run_system(name, args, kernel_seconds):
            print(show(row), flush=True)
            rows.append(row)
    if args.json:
        log = []
        if os.path.exists(args.json):
            with open(args.json) as handle:
                log = json.load(handle)
        log.append(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                        numpy=np.__version__, machine=platform.platform(), rows=rows))
        with open(args.json, "w") as handle:
            json.dump(log, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
