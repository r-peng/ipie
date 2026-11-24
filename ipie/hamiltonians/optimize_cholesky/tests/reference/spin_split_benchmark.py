"""Roadmap step 13: does a spin-split Cholesky shift pay over a tied one?

Run from the repository root (systems come from systems.npz; no PySCF needed):

    python ipie/hamiltonians/optimize_cholesky/tests/reference/spin_split_benchmark.py
    python .../spin_split_benchmark.py --cases h2_631g:1,1 lih_sto3g:3,1 --seeds 0 1 --json split.json

A case is a stored molecule's integrals on a chosen (N_up, N_down) sector
(SYSTEM:NUP,NDOWN). aa_only sectors are skipped: their mu_down is 0, so split
and tied are the same. For each case and seed, from theta0 = theta(r, .1, seed)
and the median start, with weak-Wolfe L-BFGS:

  theta|median   theta only at the fixed median shift (--iterations); its final
                 gauge is the fixed gauge below
  joint tied     theta and a tied mu (--iterations)
  joint split    theta and a split mu (--iterations), same start moved off the
                 tied subspace by a spin-odd kick (see below)
  mu tied        at the fixed gauge, mu only, tied, from the median there (--iterations)
  + tied         mu tied continued from that result (--continuation)
  + split        mu split from the same result plus a spin-odd kick (--continuation):
                 the extra gain of splitting at a fixed gauge, against the same
                 budget tied
  asym           |s (g_up - g_down)/2| at the 'mu tied' result: the first-order
                 pull of splitting (the scaled split gradient's spin-odd part)
  probe          min over factors g and signs of L(mu + eps s_g (e_up - e_down)) - L(mu)
                 at the 'mu tied' result (eps = --probe): negative means a split
                 direction lowers the loss even where asym vanishes (a saddle)

Spin symmetry. When N_up = N_down the loss is unchanged by swapping the spins
together with mu_up <-> mu_down, so a tied point is stationary in every split
direction (asym = 0) and a gradient method started there never splits. The
spin-odd kick (--kick, relative to s, random signs) lets the split runs leave
the tied subspace if that lowers the loss.

Joint runs differ by trajectory as well as by the shift form, so compare them
with the seed-to-seed spread; the fixed-gauge rows isolate the shift form.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, *[".."]*5)))

from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator  # noqa: E402
from ipie.hamiltonians.optimize_cholesky.models import InnerOptions, LBFGSOptions  # noqa: E402
from ipie.hamiltonians.optimize_cholesky.optimizers import (  # noqa: E402
    cholesky_shift_scale,
    optimize_gauge,
    resolve_cholesky_shift,
)
from ipie.hamiltonians.optimize_cholesky.problem import exponential_gauge, near_zero_eigenvalues, prepare_problem  # noqa: E402
from ipie.hamiltonians.optimize_cholesky.tests.reference.systems import molecule, theta  # noqa: E402

TAU1, TAU2 = 0.03, 0.04
DEFAULT_CASES = ("h2_631g:1,1", "h2_ccpvdz:1,1", "h2_631g:2,1", "h4_sto3g:3,1", "lih_sto3g:3,1",
                 "h4_sto3g:2,2", "lih_sto3g:2,2")


def parse_case(text):
    name, counts = text.split(":")
    up, down = (int(x) for x in counts.split(","))
    return name, (up, down)


def options(iterations):
    return LBFGSOptions(max_iterations=iterations, max_evaluations=4*iterations, line_search="weak_wolfe",
                        verbosity=0)


def summary(result, seconds):
    mu = result.cholesky_shift
    kinks = near_zero_eigenvalues(result, rtol=1e-6)
    return dict(loss=float(result.loss), initial_loss=float(result.initial_loss), status=result.status,
                iterations=int(result.iterations), evaluations=int(result.evaluations), seconds=seconds,
                kinks=len(kinks), split=None if mu is None else float(np.max(np.abs(mu[:, 0] - mu[:, 1]))))


def timed(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start


def kicked(problem, mu, scale, size, seed):
    """mu (r, 2) moved by +-size*s on spin up and the opposite on spin down (random signs)."""
    mu = np.array(mu, dtype=np.float64)
    if size == 0 or problem.policy == "aa_only":
        return mu
    signs = np.random.default_rng(1000 + seed).choice([-1., 1.], size=mu.shape[0])
    mu[:, 0] += size*scale*signs
    mu[:, 1] -= size*scale*signs
    return mu


def probe(evaluator, O, mu, scale, size, inner):
    """min over factors and signs of the relaxed loss change along spin-odd directions."""
    base = evaluator.relaxed(O, options=inner, cholesky_shift=mu).loss
    changes = []
    for g in range(mu.shape[0]):
        for sign in (1., -1.):
            trial = np.array(mu)
            trial[g, 0] += sign*size*scale[g]
            trial[g, 1] -= sign*size*scale[g]
            try:
                changes.append(evaluator.relaxed(O, options=inner, cholesky_shift=trial).loss - base)
            except ValueError:
                pass
    return float(min(changes)), int(sum(c < 0 for c in changes)), len(changes), float(base)


def run_case(name, nelec, seed, args):
    m = molecule(name)
    problem = prepare_problem(m["h1"], m["eri"], m["chol"], nelec=nelec)
    if problem.policy == "aa_only":
        return None
    r = problem.data.nchol
    theta0 = theta(r, .1, seed)
    inner = InnerOptions(optimize_shift=True)
    evaluator = GaugeEvaluator(problem, tau1=TAU1, tau2=TAU2, partition=args.partition)
    row = dict(system=name, nelec=list(nelec), policy=problem.policy, seed=seed, partition=args.partition)
    fixed, t = timed(optimize_gauge, evaluator, theta0=theta0, options=options(args.iterations),
                     inner_options=inner, cholesky_shift0="median")
    row["theta|median"] = summary(fixed, t)
    start = exponential_gauge(theta0, r)
    for mode in ("tied", "split"):
        mu0 = resolve_cholesky_shift(problem, "median", start)
        if mode == "split":
            mu0 = kicked(problem, mu0, cholesky_shift_scale(problem, start), args.kick, seed)
        result, t = timed(optimize_gauge, evaluator, theta0=theta0, options=options(args.iterations),
                          inner_options=inner, optimize_cholesky_shift=mode, cholesky_shift0=mu0)
        row[f"joint {mode}"] = summary(result, t)
    # Fixed gauge: the theta|median end point.
    angles = np.asarray(fixed.theta)
    O = exponential_gauge(angles, r)
    scale = cholesky_shift_scale(problem, O)
    tied, t = timed(optimize_gauge, evaluator, theta0=angles, options=options(args.iterations),
                    inner_options=inner, optimize_cholesky_shift="tied", optimize_theta=False)
    row["mu tied"] = summary(tied, t)
    mu = np.asarray(tied.cholesky_shift)
    for mode in ("tied", "split"):
        mu0 = kicked(problem, mu, scale, args.kick, seed) if mode == "split" else mu
        result, t = timed(optimize_gauge, evaluator, theta0=angles, options=options(args.continuation),
                          inner_options=inner, optimize_cholesky_shift=mode, optimize_theta=False,
                          cholesky_shift0=mu0)
        row[f"+ {mode}"] = summary(result, t)
    split = evaluator.with_cholesky_gradient("split").relaxed(O, options=inner, cholesky_shift=mu)
    g = np.asarray(split.cholesky_gradient)
    row["asym"] = float(np.linalg.norm(scale*(g[:, 0] - g[:, 1])/2))
    row["sym"] = float(np.linalg.norm(scale*(g[:, 0] + g[:, 1])))
    row["probe"], row["probe_down"], row["probe_count"], row["probe_base"] = probe(
        evaluator, O, mu, scale, args.probe, inner)
    return row


def relative(a, b):
    return (a - b)/b


def show(row):
    text = f"{row['system']} {tuple(row['nelec'])} {row['policy']:7s} seed {row['seed']}:"
    for key in ("theta|median", "joint tied", "joint split", "mu tied", "+ tied", "+ split"):
        item = row[key]
        text += f"\n  {key:13s} {item['initial_loss']:12.6f} -> {item['loss']:12.6f}  {item['status']:18s} " \
                f"it {item['iterations']:4d} ev {item['evaluations']:5d} {item['seconds']:6.1f}s kinks {item['kinks']:2d}" \
                + ("" if item["split"] is None else f" max|mu_up-mu_down| {item['split']:.3g}")
    text += (f"\n  joint split vs tied {100*relative(row['joint split']['loss'], row['joint tied']['loss']):+.2f}%"
             f", fixed gauge + split vs + tied {100*relative(row['+ split']['loss'], row['+ tied']['loss']):+.3f}%"
             f", |asym| {row['asym']:.3e} (|sym| {row['sym']:.3e})"
             f", probe min dL {row['probe']:+.3e} ({row['probe_down']}/{row['probe_count']} down)")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--continuation", type=int, default=200)
    parser.add_argument("--partition", default="balanced_log")
    parser.add_argument("--kick", type=float, default=1e-3, help="spin-odd start kick of split runs, relative to s")
    parser.add_argument("--probe", type=float, default=.02, help="spin-odd probe step, relative to s")
    parser.add_argument("--json", help="write the rows to this JSON file")
    args = parser.parse_args()
    rows = []
    for case in args.cases:
        name, nelec = parse_case(case)
        for seed in args.seeds:
            row = run_case(name, nelec, seed, args)
            if row is None:
                print(f"{name} {nelec}: aa_only, split and tied are the same; skipped", flush=True)
                break
            print(show(row), flush=True)
            rows.append(row)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(),
                           numpy=np.__version__, machine=platform.platform(), arguments=vars(args), rows=rows),
                      handle, indent=1)


if __name__ == "__main__":
    main()
