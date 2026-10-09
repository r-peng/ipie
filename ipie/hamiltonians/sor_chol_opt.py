#!/usr/bin/env python3
"""Pairwise Jacobi optimization of the orthogonal gauge of Cholesky factors.

Given real symmetric Cholesky factors L[t] satisfying

    (pr|qs) = sum_t L[t,p,r] L[t,q,s],

an orthogonal rotation in Cholesky-index space,

    L'[a] =  cos(theta) L[a] + sin(theta) L[b]
    L'[b] = -sin(theta) L[a] + cos(theta) L[b],

leaves the represented ERI tensor exactly unchanged.  This script uses repeated
pairwise Givens/Jacobi rotations to reduce a commutator proxy built from the
density eigenmodes of the individual Cholesky factors.

For each leaf t,

    L[t] = U[t] diag(lam[t]) U[t]^T,

and P[t,k] = u[t,k] u[t,k]^T.  For two leaves t != u,

    ||[P[t,k], P[u,m]]||_F^2
      = 2 s^2 (1-s^2),
      s = u[t,k]^T u[u,m].

The activity weight used here is the X-DF/LAFQMC proxy

    q[t,k] = |lam[t,k]| * sum_l |lam[t,l]|,

which follows from Z[t,kl] = lam[t,k] lam[t,l] and d^2 ~ |Z[t,kl]|.

Objectives:
  geometry       : sum_{t<u,k,m} 2 s^2(1-s^2)
  weighted_sum   : sum q[t,k] q[u,m] 2 s^2(1-s^2)   [recommended]
  weighted_mean  : weighted_sum / sum q[t,k]q[u,m]

This is a proxy for the ERI-only LAFQMC commutator objective.  It does not
include h1 terms, trial-dependent mean shifts, or the precise spin/multiplicity
weights in a particular SOR implementation.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class LeafData:
    U: np.ndarray
    lam: np.ndarray
    q: np.ndarray


@dataclass
class MetricStats:
    geometry_sum: float
    weighted_sum: float
    weighted_den: float

    @property
    def weighted_mean(self) -> float:
        if self.weighted_den <= 0.0:
            return 0.0
        return self.weighted_sum / self.weighted_den


@dataclass
class CholeskyGaugeResult:
    chol: np.ndarray
    gauge: np.ndarray
    history: List[Dict[str, float]]
    objective: str
    n_sweeps: int
    n_accepted: int


def _symmetrize_chol(chol: np.ndarray) -> np.ndarray:
    chol = np.asarray(chol, dtype=np.float64)
    if chol.ndim != 3 or chol.shape[1] != chol.shape[2]:
        raise ValueError(
            "chol must have shape (nchol, nbasis, nbasis); "
            f"got {chol.shape}"
        )
    return 0.5 * (chol + chol.transpose(0, 2, 1))


def _leaf_data(L: np.ndarray, eig_tol: float = 0.0) -> LeafData:
    """Diagonalize one real symmetric Cholesky factor."""
    L = 0.5 * (L + L.T)
    lam, U = np.linalg.eigh(L)

    # Sorting is not mathematically necessary, but gives more reproducible output.
    order = np.argsort(np.abs(lam))[::-1]
    lam = lam[order]
    U = U[:, order]

    if eig_tol > 0.0:
        scale = max(float(np.max(np.abs(lam))), 1.0)
        lam = lam.copy()
        lam[np.abs(lam) < eig_tol * scale] = 0.0

    abs_lam = np.abs(lam)
    q = abs_lam * abs_lam.sum()
    return LeafData(U=U, lam=lam, q=q)


def _pair_stats(a: LeafData, b: LeafData) -> MetricStats:
    """Commutator-proxy statistics for one pair of DF leaves."""
    S = a.U.T @ b.U
    s2 = S * S
    # Roundoff can make s^2 a few ulps larger than one.
    s2 = np.clip(s2, 0.0, 1.0)
    geom = 2.0 * s2 * (1.0 - s2)

    geometry_sum = float(np.sum(geom))
    weights = a.q[:, None] * b.q[None, :]
    weighted_sum = float(np.sum(weights * geom))
    weighted_den = float(np.sum(weights))
    return MetricStats(geometry_sum, weighted_sum, weighted_den)


def _add_stats(x: MetricStats, y: MetricStats, sign: float = 1.0) -> MetricStats:
    return MetricStats(
        x.geometry_sum + sign * y.geometry_sum,
        x.weighted_sum + sign * y.weighted_sum,
        x.weighted_den + sign * y.weighted_den,
    )


def _global_stats(leaves: List[LeafData]) -> MetricStats:
    total = MetricStats(0.0, 0.0, 0.0)
    nt = len(leaves)
    for t in range(nt):
        for u in range(t + 1, nt):
            total = _add_stats(total, _pair_stats(leaves[t], leaves[u]))
    return total


def _local_stats(leaves: List[LeafData], a: int, b: int) -> MetricStats:
    """All global pair contributions involving leaf a or b, counted once."""
    total = _pair_stats(leaves[a], leaves[b])
    for k in range(len(leaves)):
        if k == a or k == b:
            continue
        total = _add_stats(total, _pair_stats(leaves[a], leaves[k]))
        total = _add_stats(total, _pair_stats(leaves[b], leaves[k]))
    return total


def _objective_value(stats: MetricStats, objective: str) -> float:
    if objective == "geometry":
        return stats.geometry_sum
    if objective == "weighted_sum":
        return stats.weighted_sum
    if objective == "weighted_mean":
        return stats.weighted_mean
    raise ValueError(f"unknown objective {objective!r}")


def _report_row(sweep: int, stats: MetricStats, nleaf: int, nbasis: int,
                accepted: int) -> Dict[str, float]:
    npairs = nleaf * (nleaf - 1) / 2
    geom_den = max(npairs * nbasis * nbasis, 1.0)
    return {
        "sweep": float(sweep),
        "geometry_sum": float(stats.geometry_sum),
        "geometry_mean": float(stats.geometry_sum / geom_den),
        "weighted_sum": float(stats.weighted_sum),
        "weighted_den": float(stats.weighted_den),
        "weighted_mean": float(stats.weighted_mean),
        "accepted": float(accepted),
    }


def optimize_cholesky_gauge(
    chol: np.ndarray,
    *,
    objective: str = "weighted_sum",
    max_sweeps: int = 8,
    angle_grid: int = 17,
    angle_tol: float = 1.0e-7,
    accept_rtol: float = 1.0e-10,
    sweep_rtol: float = 1.0e-8,
    eig_tol: float = 0.0,
    random_order: bool = False,
    seed: int = 7,
    verbose: bool = True,
) -> CholeskyGaugeResult:
    """Optimize the orthogonal gauge of a set of Cholesky factors.

    Parameters
    ----------
    chol
        Array with shape (nchol, nbasis, nbasis).
    objective
        One of {'geometry', 'weighted_sum', 'weighted_mean'}.
        'weighted_sum' is the recommended first ERI-only proxy.
    max_sweeps
        Maximum number of complete Jacobi sweeps over all Cholesky pairs.
    angle_grid
        Number of coarse angles in [-pi/4, pi/4] used before local refinement.
        Use an odd value so theta=0 is included.
    angle_tol
        Absolute tolerance for the one-dimensional bounded refinement.
    accept_rtol
        Relative decrease required to accept a pair rotation.
    sweep_rtol
        Stop if the relative objective improvement over a full sweep is below this.
    eig_tol
        Relative threshold for zeroing tiny eigenvalues of each L[t] when building
        the proxy only.  The Cholesky factors themselves are never truncated.
    random_order
        Randomize the order of leaf pairs in each sweep.
    seed
        RNG seed used only when random_order=True.
    """
    if objective not in {"geometry", "weighted_sum", "weighted_mean"}:
        raise ValueError("objective must be geometry, weighted_sum, or weighted_mean")
    if angle_grid < 3:
        raise ValueError("angle_grid must be at least 3")
    if angle_grid % 2 == 0:
        raise ValueError("angle_grid should be odd so that theta=0 is sampled")

    L = _symmetrize_chol(chol).copy()
    nleaf, nbasis, _ = L.shape
    if nleaf < 2:
        raise ValueError("need at least two Cholesky factors to optimize the gauge")

    gauge = np.eye(nleaf)
    leaves = [_leaf_data(L[t], eig_tol=eig_tol) for t in range(nleaf)]
    stats = _global_stats(leaves)

    history: List[Dict[str, float]] = []
    history.append(_report_row(0, stats, nleaf, nbasis, 0))

    if verbose:
        print("initial metrics")
        print(f"  geometry_sum  = {stats.geometry_sum:.12e}")
        print(f"  weighted_sum  = {stats.weighted_sum:.12e}")
        print(f"  weighted_mean = {stats.weighted_mean:.12e}")
        print(f"  objective({objective}) = {_objective_value(stats, objective):.12e}")

    rng = np.random.default_rng(seed)
    pair_list = [(a, b) for a in range(nleaf) for b in range(a + 1, nleaf)]
    grid = np.linspace(-0.25 * np.pi, 0.25 * np.pi, angle_grid)

    total_accepted = 0

    for sweep in range(1, max_sweeps + 1):
        obj_before_sweep = _objective_value(stats, objective)
        accepted_this_sweep = 0

        pairs = pair_list.copy()
        if random_order:
            rng.shuffle(pairs)

        for a, b in pairs:
            La0 = L[a].copy()
            Lb0 = L[b].copy()
            old_local = _local_stats(leaves, a, b)
            current_obj = _objective_value(stats, objective)

            def evaluate(theta: float):
                c = np.cos(theta)
                s = np.sin(theta)
                La = c * La0 + s * Lb0
                Lb = -s * La0 + c * Lb0
                da = _leaf_data(La, eig_tol=eig_tol)
                db = _leaf_data(Lb, eig_tol=eig_tol)

                trial_leaves_a = leaves[a]
                trial_leaves_b = leaves[b]
                leaves[a] = da
                leaves[b] = db
                new_local = _local_stats(leaves, a, b)
                leaves[a] = trial_leaves_a
                leaves[b] = trial_leaves_b

                new_stats = _add_stats(_add_stats(stats, old_local, sign=-1.0), new_local)
                return _objective_value(new_stats, objective)

            vals = np.array([evaluate(theta) for theta in grid])
            ibest = int(np.argmin(vals))
            best_theta = float(grid[ibest])
            best_obj = float(vals[ibest])

            # Refine only when the best coarse point is not at a periodic boundary.
            if 0 < ibest < angle_grid - 1:
                lo = float(grid[ibest - 1])
                hi = float(grid[ibest + 1])
                res = minimize_scalar(
                    evaluate,
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": angle_tol},
                )
                if res.fun < best_obj:
                    best_theta = float(res.x)
                    best_obj = float(res.fun)

            threshold = accept_rtol * max(abs(current_obj), 1.0e-16)
            if best_obj < current_obj - threshold:
                c = np.cos(best_theta)
                s = np.sin(best_theta)

                La = c * La0 + s * Lb0
                Lb = -s * La0 + c * Lb0
                da = _leaf_data(La, eig_tol=eig_tol)
                db = _leaf_data(Lb, eig_tol=eig_tol)

                # Compute the exact local delta using the accepted leaf data.
                old_a, old_b = leaves[a], leaves[b]
                leaves[a], leaves[b] = da, db
                new_local = _local_stats(leaves, a, b)
                stats = _add_stats(_add_stats(stats, old_local, sign=-1.0), new_local)

                L[a], L[b] = La, Lb

                # Accumulate the orthogonal gauge: L_current = gauge @ L_initial.
                ga = gauge[a].copy()
                gb = gauge[b].copy()
                gauge[a] = c * ga + s * gb
                gauge[b] = -s * ga + c * gb

                accepted_this_sweep += 1
                total_accepted += 1
            else:
                # Restore references explicitly for clarity; they were not modified.
                leaves[a], leaves[b] = leaves[a], leaves[b]

        # Recompute from scratch once per sweep to eliminate incremental roundoff.
        leaves = [_leaf_data(L[t], eig_tol=eig_tol) for t in range(nleaf)]
        stats = _global_stats(leaves)
        history.append(_report_row(sweep, stats, nleaf, nbasis, accepted_this_sweep))

        obj_after_sweep = _objective_value(stats, objective)
        rel_improve = (obj_before_sweep - obj_after_sweep) / max(
            abs(obj_before_sweep), 1.0e-16
        )

        if verbose:
            print(
                f"sweep {sweep:2d}: accepted={accepted_this_sweep:4d}, "
                f"objective={obj_after_sweep:.12e}, "
                f"geometry_sum={stats.geometry_sum:.12e}, "
                f"weighted_sum={stats.weighted_sum:.12e}, "
                f"weighted_mean={stats.weighted_mean:.12e}"
            )

        if accepted_this_sweep == 0 or rel_improve < sweep_rtol:
            break

    # Reconstruct from the accumulated gauge to enforce the exact gauge relation and
    # avoid small accumulation differences from repeated direct rotations.
    L0 = _symmetrize_chol(chol)
    Lfinal = np.einsum("tg,gpq->tpq", gauge, L0, optimize=True)
    Lfinal = _symmetrize_chol(Lfinal)

    return CholeskyGaugeResult(
        chol=Lfinal,
        gauge=gauge,
        history=history,
        objective=objective,
        n_sweeps=len(history) - 1,
        n_accepted=total_accepted,
    )


def commutator_proxy_metrics(chol: np.ndarray, eig_tol: float = 0.0) -> Dict[str, float]:
    """Return geometric and activity-weighted cross-leaf proxy metrics."""
    L = _symmetrize_chol(chol)
    leaves = [_leaf_data(x, eig_tol=eig_tol) for x in L]
    stats = _global_stats(leaves)
    nleaf, nbasis, _ = L.shape
    npairs = nleaf * (nleaf - 1) / 2
    geom_den = max(npairs * nbasis * nbasis, 1.0)
    return {
        "geometry_sum": stats.geometry_sum,
        "geometry_mean": stats.geometry_sum / geom_den,
        "weighted_sum": stats.weighted_sum,
        "weighted_den": stats.weighted_den,
        "weighted_mean": stats.weighted_mean,
    }


def gauge_invariance_error(chol0: np.ndarray, chol1: np.ndarray) -> float:
    """Relative pair-space ERI difference; intended for small verification tests."""
    A = _symmetrize_chol(chol0).reshape(chol0.shape[0], -1)
    B = _symmetrize_chol(chol1).reshape(chol1.shape[0], -1)
    V0 = A.T @ A
    V1 = B.T @ B
    denom = max(np.linalg.norm(V0), 1.0e-300)
    return float(np.linalg.norm(V1 - V0) / denom)


def _load_chol(path: str, key: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        d = np.load(p)
        if key not in d:
            raise KeyError(f"{p} contains keys {list(d.keys())}; requested {key!r}")
        return d[key]
    if p.suffix in {".pkl", ".pickle"}:
        with open(p, "rb") as f:
            d = pickle.load(f)
        if isinstance(d, dict):
            if key not in d:
                raise KeyError(f"{p} contains keys {list(d.keys())}; requested {key!r}")
            return np.asarray(d[key])
        return np.asarray(d)
    raise ValueError("input must be .npy, .npz, .pkl, or .pickle")


def _history_arrays(history: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    keys = history[0].keys()
    return {f"history_{k}": np.asarray([row[k] for row in history]) for k in keys}


def _self_test() -> int:
    rng = np.random.default_rng(1234)
    nleaf, n = 5, 3
    A = rng.standard_normal((nleaf, n, n))
    chol = 0.5 * (A + A.transpose(0, 2, 1))

    before = commutator_proxy_metrics(chol)
    result = optimize_cholesky_gauge(
        chol,
        objective="weighted_sum",
        max_sweeps=3,
        angle_grid=13,
        verbose=False,
    )
    after = commutator_proxy_metrics(result.chol)
    eri_err = gauge_invariance_error(chol, result.chol)
    orth_err = np.linalg.norm(result.gauge @ result.gauge.T - np.eye(nleaf))

    print("self-test")
    print(f"  weighted_sum before = {before['weighted_sum']:.12e}")
    print(f"  weighted_sum after  = {after['weighted_sum']:.12e}")
    print(f"  ERI gauge error      = {eri_err:.12e}")
    print(f"  gauge orthogonality  = {orth_err:.12e}")

    ok = (
        after["weighted_sum"] <= before["weighted_sum"] * (1.0 + 1.0e-10)
        and eri_err < 1.0e-11
        and orth_err < 1.0e-11
    )
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize the orthogonal Cholesky gauge to reduce a DF/LAFQMC commutator proxy."
    )
    parser.add_argument("input", nargs="?", help="Input .npy/.npz/.pkl containing chol factors")
    parser.add_argument("--key", default="chol", help="Array key for .npz/.pkl input")
    parser.add_argument("--output", default="chol_comm_opt.npz")
    parser.add_argument(
        "--objective",
        choices=["geometry", "weighted_sum", "weighted_mean"],
        default="weighted_sum",
        help="Gauge-optimization objective; weighted_sum is recommended",
    )
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--angle-grid", type=int, default=17)
    parser.add_argument("--angle-tol", type=float, default=1.0e-7)
    parser.add_argument("--accept-rtol", type=float, default=1.0e-10)
    parser.add_argument("--sweep-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--eig-tol",
        type=float,
        default=0.0,
        help="Relative eigenvalue threshold used only in the proxy metric",
    )
    parser.add_argument("--random-order", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Explicitly build the pair-space ERI matrix before/after (small systems only)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if args.input is None:
        parser.error("input is required unless --self-test is used")

    chol = _load_chol(args.input, args.key)
    before = commutator_proxy_metrics(chol, eig_tol=args.eig_tol)

    print(f"chol shape = {chol.shape}")
    print("before optimization:")
    for k, v in before.items():
        print(f"  {k:16s} = {v:.12e}")

    result = optimize_cholesky_gauge(
        chol,
        objective=args.objective,
        max_sweeps=args.sweeps,
        angle_grid=args.angle_grid,
        angle_tol=args.angle_tol,
        accept_rtol=args.accept_rtol,
        sweep_rtol=args.sweep_rtol,
        eig_tol=args.eig_tol,
        random_order=args.random_order,
        seed=args.seed,
        verbose=True,
    )

    after = commutator_proxy_metrics(result.chol, eig_tol=args.eig_tol)
    orth_err = np.linalg.norm(result.gauge @ result.gauge.T - np.eye(result.gauge.shape[0]))

    print("after optimization:")
    for k, v in after.items():
        print(f"  {k:16s} = {v:.12e}")
    print(f"accepted rotations = {result.n_accepted}")
    print(f"completed sweeps    = {result.n_sweeps}")
    print(f"||O O^T - I||_F     = {orth_err:.12e}")

    eri_err = np.nan
    if args.verify:
        eri_err = gauge_invariance_error(chol, result.chol)
        print(f"relative ERI change = {eri_err:.12e}")

    payload = {
        "chol": result.chol,
        "gauge": result.gauge,
        "objective": np.array(result.objective),
        "n_sweeps": np.array(result.n_sweeps),
        "n_accepted": np.array(result.n_accepted),
        "gauge_orthogonality_error": np.array(orth_err),
        "verified_relative_eri_change": np.array(eri_err),
    }
    for prefix, d in (("before", before), ("after", after)):
        for k, v in d.items():
            payload[f"{prefix}_{k}"] = np.array(v)
    payload.update(_history_arrays(result.history))

    np.savez(args.output, **payload)
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
