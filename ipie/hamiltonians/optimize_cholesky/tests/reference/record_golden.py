"""Record golden_3680f29.npz: reference numbers from the frozen package.

Run from the root of a checkout of the FROZEN commit (3680f29) with this
directory copied in; the script refuses to run elsewhere unless --any-commit:

    python ipie/hamiltonians/optimize_cholesky/tests/reference/record_golden.py
    # H2/cc-pVDZ (oracle only, ~1.5 h of pair loops; split over processes):
    python .../record_golden.py --h2-chunk 0 2 &  python .../record_golden.py --h2-chunk 1 2
    python .../record_golden.py --h2-combine 2

What is recorded (keys "<case>/<evaluation>/<field>", inputs stored next to
outputs so a test can replay every evaluation exactly):

  synthetic cases   2 sizes x 3 sectors x 2 partition maps x 2 loss kinds, each with
                    fixed (fixed t), fixed_z (fixed z), objective (raw and weighted)
                    and relaxed (without and with the shift relaxed)
  small molecules   h2_631g (ab_only), h2_631g_triplet (aa_only), h4_sto3g (hybrid):
                    fixed, objective, relaxed (shift relaxed) and a 5-iteration
                    weak-Wolfe L-BFGS run;  lih_sto3g (hybrid): fixed, objective
  h2_ccpvdz         fixed-t loss and gauge gradient at one gauge, computed with the
                    frozen oracle (the package's pair loops take the same time)

Everything except h2_ccpvdz comes from the package's public entry points
(GaugeEvaluator.fixed/fixed_z/relaxed, loss.objective_loss, optimize_gauge).
"""

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."]*5))
sys.path.insert(0, ROOT)

from ipie.hamiltonians.optimize_cholesky.tests.reference import reference_kernels as rk  # noqa: E402
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import (  # noqa: E402
    GOLDEN_PATH, molecule, synthetic, theta)

TAU1, TAU2 = 0.03, 0.04
INNER = dict(max_sweeps=3000, gradient_atol=1e-12, gradient_rtol=1e-10)
OUTER = dict(max_iterations=5, line_search="weak_wolfe", validate_final=False)
SYNTHETIC_SIZES = ((3, 2, 23), (4, 3, 5))
SECTORS = ((2, 1), (1, 1), (2, 0))
MAPS = ("balanced_log", "anchored_log")
KINDS = ("weighted", "raw")
SMALL_MOLECULES = ("h2_631g", "h2_631g_triplet", "h4_sto3g")
FIXED_ONLY_MOLECULES = ("lih_sto3g",)
H2 = dict(name="h2_ccpvdz", theta_scale=0.1, theta_seed=11, t=0.0, shift=(0.25, 0.25),
          partition="balanced_log", loss_kind="weighted")


def synthetic_cases():
    for (n, r, seed), nelec, partition, kind in [(s, e, p, k) for s in SYNTHETIC_SIZES for e in SECTORS
                                                 for p in MAPS for k in KINDS]:
        name = f"syn_n{n}_r{r}_s{seed}_{nelec[0]}{nelec[1]}_{partition[:4]}_{kind}"
        shift = (0.05, -0.03) if nelec[1] else (0.05, 0.0)
        yield name, dict(source="synthetic", n=n, r=r, seed=seed, nelec=nelec, partition=partition,
                         loss_kind=kind, shift=shift, theta_scale=0.5, theta_seed=seed)


def molecule_cases():
    for name in SMALL_MOLECULES + FIXED_ONLY_MOLECULES:
        yield name, dict(source="molecule", system=name, nelec=molecule(name)["nelec"],
                         partition="balanced_log", loss_kind="weighted", shift=None,
                         theta_scale=0.3, theta_seed=3, relaxed=name in SMALL_MOLECULES,
                         outer=name in SMALL_MOLECULES)


def tensors(spec):
    if spec["source"] == "synthetic":
        return synthetic(spec["n"], spec["r"], spec["seed"])
    m = molecule(spec["system"])
    return m["h1"], m["eri"], m["chol"]


def evaluation_inputs(spec, chol):
    """O, fixed t, fixed z: drawn from the frozen oracle's bounds, then stored verbatim."""
    r = chol.shape[0]
    O = rk.exponential_gauge(theta(r, spec["theta_scale"], spec["theta_seed"]), r)
    h1 = np.zeros(chol.shape[1:])
    oracle = rk.ReferenceOracle(h1, chol, spec["nelec"], tau1=TAU1, tau2=TAU2, partition=spec["partition"])
    bounds = oracle.bounds(oracle.gauge(O))
    rng = np.random.default_rng(spec["theta_seed"]+1)
    S = bounds.shape[0]
    t = bounds[:, 0] + rng.uniform(0.2, 0.8, S)*(bounds[:, 1]-bounds[:, 0])
    z = rng.uniform(-0.9, 0.9, S)
    return O, t, z


def record_case(name, spec, out):
    from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
    from ipie.hamiltonians.optimize_cholesky.loss import objective_loss
    from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions
    from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
    from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem

    h1, eri, chol = tensors(spec)
    problem = prepare_problem(h1, eri, chol, nelec=spec["nelec"])
    O, t, z = evaluation_inputs(spec, chol)
    ev = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=spec["partition"],
                        loss_kind=spec["loss_kind"], shift=spec["shift"])
    out[f"{name}/input/O"] = O
    out[f"{name}/input/t"] = t
    out[f"{name}/input/z"] = z

    g = ev.fixed(O, t=t)
    out[f"{name}/fixed/loss"] = g.loss
    out[f"{name}/fixed/gradient_O"] = g.gradient_O
    out[f"{name}/fixed/partition_gradient"] = g.partition_gradient

    if spec["source"] == "synthetic":
        g = ev.fixed_z(O, z=z)
        out[f"{name}/fixed_z/loss"] = g.loss
        out[f"{name}/fixed_z/gradient_O"] = g.gradient_O

    decomposition = ev.decomposition(O, t=t)
    for kind in KINDS:
        out[f"{name}/objective/{kind}"] = objective_loss(decomposition, loss_kind=kind)

    shifts = (False, True) if spec["source"] == "synthetic" else ((True,) if spec.get("relaxed") else ())
    for relax_shift in shifts:
        key = f"{name}/relaxed_{'shift' if relax_shift else 'fixed_shift'}"
        g = ev.relaxed(O, options=InnerOptions(**INNER, optimize_shift=relax_shift))
        inner = g.inner_result
        out[f"{key}/loss"] = g.loss
        out[f"{key}/gradient_O"] = g.gradient_O
        out[f"{key}/tangent_gradient"] = g.tangent_gradient
        out[f"{key}/t"] = np.asarray(inner.t)
        out[f"{key}/shift"] = np.asarray(inner.shift)
        out[f"{key}/converged"] = bool(inner.converged)
        out[f"{key}/sweeps"] = int(inner.sweeps)

    if spec.get("outer"):
        r = chol.shape[0]
        theta0 = theta(r, 0.1, spec["theta_seed"]+7)
        result = optimize_gauge(ev, theta0=theta0, options=LBFGSOptions(**OUTER),
                                inner_options=InnerOptions(**INNER, optimize_shift=True))
        key = f"{name}/outer"
        out[f"{key}/theta0"] = theta0
        out[f"{key}/history_loss"] = np.array([row.loss for row in result.history])
        out[f"{key}/theta"] = np.asarray(result.theta)
        out[f"{key}/gradient_theta"] = np.asarray(result.gradient_theta)
        out[f"{key}/loss"] = result.loss
        out[f"{key}/iterations"] = int(result.iterations)
        out[f"{key}/evaluations"] = int(result.evaluations)
        out[f"{key}/status"] = np.array(result.status)


def h2_chunk_path(k, nchunks):
    return os.path.join(HERE, f"_h2_ccpvdz_chunk{k}of{nchunks}.npz")


def h2_setup():
    m = molecule(H2["name"])
    chol = m["chol"]
    r = chol.shape[0]
    O = rk.exponential_gauge(theta(r, H2["theta_scale"], H2["theta_seed"]), r)
    oracle = rk.ReferenceOracle(m["h1"], chol, m["nelec"], tau1=TAU1, tau2=TAU2, partition=H2["partition"],
                                loss_kind=H2["loss_kind"])
    decomposition = oracle.decomposition(O, t=H2["t"], shift=H2["shift"])
    return m, oracle, O, decomposition


def h2_row_ranges(nblocks, nchunks):
    """Split the outer pair index so every chunk owns about the same number of pairs."""
    pairs = np.arange(nblocks)[::-1].astype(float)              # row i owns nblocks-1-i pairs
    cumulative = np.cumsum(pairs)
    edges = [0] + [int(np.searchsorted(cumulative, cumulative[-1]*k/nchunks)) for k in range(1, nchunks)] + [nblocks]
    return [(edges[k], edges[k+1]) for k in range(nchunks)]


def run_h2_chunk(k, nchunks):
    _, oracle, _, decomposition = h2_setup()
    nblocks = len(decomposition.one_body.channels) + len(decomposition.two_body.channels)
    rows = h2_row_ranges(nblocks, nchunks)[k]
    start = time.perf_counter()
    loss, gamp, gX = rk.loss_pullback(decomposition, H2["loss_kind"], rows=rows)
    np.savez(h2_chunk_path(k, nchunks), loss=loss, g_amplitude=gamp, gX=gX, rows=np.array(rows),
             seconds=time.perf_counter()-start)
    print(f"chunk {k}/{nchunks} rows {rows}: {time.perf_counter()-start:.0f} s")


def combine_h2(nchunks, out):
    m, oracle, O, decomposition = h2_setup()
    parts = [np.load(h2_chunk_path(k, nchunks)) for k in range(nchunks)]
    loss = math.fsum(float(p["loss"]) for p in parts)
    gamp = sum(p["g_amplitude"] for p in parts)
    gX = sum(p["gX"] for p in parts)
    result = rk.assemble(oracle.chol, decomposition, oracle.settings, loss_kind=H2["loss_kind"],
                         pullback=(loss, gamp, gX))
    key = H2["name"]
    out[f"{key}/input/O"] = O
    out[f"{key}/input/t"] = np.full(len(oracle.sources), H2["t"])
    out[f"{key}/input/shift"] = np.array(H2["shift"])
    out[f"{key}/fixed/loss"] = result["loss"]
    out[f"{key}/fixed/gradient_O"] = result["gradient_O"]
    out[f"{key}/fixed/partition_gradient"] = result["partition_gradient"]
    return sum(float(p["seconds"]) for p in parts)


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h2-chunk", nargs=2, type=int, metavar=("K", "N"))
    parser.add_argument("--h2-combine", type=int, metavar="N", help="combine N H2/cc-pVDZ chunks into the golden file")
    parser.add_argument("--any-commit", action="store_true", help="allow recording outside the frozen commit")
    args = parser.parse_args()
    head = git_head()
    if head != rk.FROZEN_COMMIT and not args.any_commit:
        sys.exit(f"HEAD is {head}, not the frozen commit {rk.FROZEN_COMMIT}; refusing to record golden data")
    if args.h2_chunk:
        run_h2_chunk(*args.h2_chunk)
        return

    out, specs = {}, {}
    if os.path.exists(GOLDEN_PATH):
        with np.load(GOLDEN_PATH) as old:
            out.update({key: old[key] for key in old.files if key != "metadata"})
            specs.update(json.loads(str(old["metadata"]))["cases"])
    if args.h2_combine:
        seconds = combine_h2(args.h2_combine, out)
        specs[H2["name"]] = dict(source="molecule", system=H2["name"], nelec=molecule(H2["name"])["nelec"],
                                 partition=H2["partition"], loss_kind=H2["loss_kind"], shift=H2["shift"],
                                 theta_scale=H2["theta_scale"], theta_seed=H2["theta_seed"], t=H2["t"],
                                 computed_by="reference_kernels (oracle)", oracle_seconds=seconds)
    else:
        for name, spec in list(synthetic_cases()) + list(molecule_cases()):
            start = time.perf_counter()
            record_case(name, spec, out)
            spec["computed_by"] = "package"
            spec["seconds"] = time.perf_counter()-start
            specs[name] = spec
            print(f"{name:42s} {spec['seconds']:7.1f} s", flush=True)
    metadata = dict(frozen_commit=rk.FROZEN_COMMIT, recorded_at_head=head, numpy=np.__version__,
                    scipy=scipy.__version__, python=platform.python_version(), platform=platform.platform(),
                    recorded=time.strftime("%Y-%m-%d %H:%M:%S %Z"), tau1=TAU1, tau2=TAU2, inner=INNER,
                    outer=OUTER, cases=specs)
    np.savez_compressed(GOLDEN_PATH, metadata=np.array(json.dumps(metadata, default=list)), **out)
    print("wrote", GOLDEN_PATH, f"({len(out)} arrays, {len(specs)} cases)")


if __name__ == "__main__":
    main()
