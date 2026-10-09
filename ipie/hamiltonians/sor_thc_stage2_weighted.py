#!/usr/bin/env python3
"""Stage-2 THC optimization with W-weighted commutator geometry.

Fits a real molecular ERI tensor in the ordering

    eri[p, r, q, s] = (p r | q s)

with the THC form

    (p r | q s) ~= sum_{a,b} X[p,a] X[r,a] W[a,b] X[q,b] X[s,b].

Columns of X are normalized to unit 2-norm.  For fixed X, W is obtained by
variable projection (least squares):

    Z[(p,r),a] = X[p,a] X[r,a]
    G = Z.T @ Z
    B = Z.T @ V @ Z
    W = G^+ B G^+.

Stage 2 uses the LAFQMC-motivated activity

    q_a = sum_b |W[a,b]|,

because for the current sum-of-rotations decomposition at zero mean shift,
|d_{a|ab}|^2 is proportional to |W_ab|.  With s_ab = x_a^T x_b,
we define W-weighted geometric diagnostics

    L_comm = < 2 s_ab^2 (1 - s_ab^2) >_{q_a q_b, a<b}
    L_dup  = < s_ab^4 >_{q_a q_b, a<b}
    L_orth = < s_ab^2 >_{q_a q_b, a<b}.

The optimized objective is

    L_total = L_ERI + comm_lambda * L_comm + dup_lambda * L_dup.

IMPORTANT: the pure commutator factor 2 s^2 (1-s^2) still vanishes for both
orthogonal and collinear THC columns.  Therefore Stage 2 can suffer the same
s -> +/-1 pathology as Stage 1 if L_dup is omitted.  By default this script
sets

    dup_lambda = 2 * comm_lambda,

so the weighted geometric contribution becomes exactly

    2 * comm_lambda * L_orth,

for fixed activity weights q.  Thus collinear columns are penalized rather than
rewarded, while the true W-weighted commutator loss is still reported as a
separate diagnostic.

Why an outer loop?
------------------
q depends on W, and W depends on X.  The envelope theorem used for the ERI
variable-projection gradient does NOT remove dW/dX from a W-weighted
commutator penalty.  Rather than use an inconsistent gradient, this script uses
an alternating / iteratively reweighted scheme:

  1. For the current X, compute W and q_a = sum_b |W_ab|.
  2. Freeze q and optimize X with L-BFGS-B.  For frozen q, the geometric
     gradient is analytic and the ERI envelope-theorem gradient is valid.
  3. Recompute W and q from the new X and repeat.

This is a controlled Stage-2 surrogate optimization.  It is not the exact
full derivative of L_comm[X, W_opt(X)], but each inner optimization has a
consistent objective and gradient.

The script reports/saves diagnostics for collapse and ill conditioning:

    X.T X, Z.T Z, eig(Z.T Z), cond(Z.T Z), ||W||_F, max|W_ab|, q.

Setting comm_lambda=dup_lambda=0 reduces to the original ERI-only variable-
projection fit (the outer loop is then automatically reduced to one pass).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from scipy.optimize import minimize


@dataclass
class THCFitResult:
    X: np.ndarray
    W: np.ndarray
    relative_error: float
    objective: float
    eri_objective: float
    comm_objective: float
    dup_objective: float
    orth_objective: float
    activity: np.ndarray
    outer_iterations: int
    nit: int
    success: bool
    message: str
    gram_X: np.ndarray
    gram_Z: np.ndarray
    gram_Z_eigs: np.ndarray
    gram_Z_condition: float
    W_fro: float
    W_max: float


def _check_eri(eri: np.ndarray, sym_tol: float = 1.0e-8) -> np.ndarray:
    eri = np.asarray(eri)
    if eri.ndim != 4:
        raise ValueError(f"ERI must be rank 4, got shape {eri.shape}")
    m = eri.shape[0]
    if eri.shape != (m, m, m, m):
        raise ValueError(f"ERI must have shape (M,M,M,M), got {eri.shape}")

    if np.iscomplexobj(eri):
        if np.max(np.abs(eri.imag)) > sym_tol:
            raise NotImplementedError("This fitter currently supports real ERIs only.")
        eri = eri.real

    eri = np.asarray(eri, dtype=np.float64)
    V = eri.reshape(m * m, m * m)
    rel_pair_sym = np.linalg.norm(V - V.T) / max(np.linalg.norm(V), 1.0)
    if rel_pair_sym > sym_tol:
        raise ValueError(
            "ERI is not symmetric under (p,r)<->(q,s) in the supplied axis order. "
            f"Relative pair-space asymmetry = {rel_pair_sym:.3e}. "
            "Expected eri[p,r,q,s] = (pr|qs)."
        )
    return eri


def _normalize_columns(Y: np.ndarray, floor: float = 1.0e-12):
    norms = np.linalg.norm(Y, axis=0)
    norms = np.maximum(norms, floor)
    return Y / norms[None, :], norms


def _build_Z(X: np.ndarray) -> np.ndarray:
    """Z[(p,r),a] = X[p,a] X[r,a]."""
    m, rank = X.shape
    return np.einsum("pa,ra->pra", X, X, optimize=True).reshape(m * m, rank)


def optimal_W(V: np.ndarray, X: np.ndarray, rcond: float = 1.0e-10) -> np.ndarray:
    """Least-squares-optimal symmetric W for fixed column-normalized X."""
    Z = _build_Z(X)
    G = Z.T @ Z
    Ginv = np.linalg.pinv(G, rcond=rcond)
    B = Z.T @ V @ Z
    W = Ginv @ B @ Ginv
    return 0.5 * (W + W.T)


def reconstruct_eri(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    m = X.shape[0]
    Z = _build_Z(X)
    return (Z @ W @ Z.T).reshape(m, m, m, m)


def activity_from_W(W: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """q_a = sum_b |W_ab|.

    A small optional floor can keep completely inactive columns from receiving
    exactly zero geometric weight.  The default floor=0 preserves the direct
    LAFQMC-motivated definition.
    """
    q = np.sum(np.abs(W), axis=1)
    if floor > 0.0:
        scale = max(float(np.mean(q)), 1.0)
        q = np.maximum(q, floor * scale)
    return q


def weighted_geometric_losses_and_grads(
    X: np.ndarray,
    q: np.ndarray,
):
    """W-activity-weighted geometric losses for *fixed* q.

    Let s_ab = x_a^T x_b and w_ab = q_a q_b.  Returns

        L_comm = sum_{a<b} w_ab 2 s_ab^2(1-s_ab^2) / sum_{a<b} w_ab
        L_dup  = sum_{a<b} w_ab s_ab^4              / sum_{a<b} w_ab
        L_orth = sum_{a<b} w_ab s_ab^2              / sum_{a<b} w_ab

    together with gradients with respect to the normalized X.  q is treated as
    constant in these gradients.
    """
    rank = X.shape[1]
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (rank,):
        raise ValueError(f"q must have shape ({rank},), got {q.shape}")

    if rank < 2:
        z = np.zeros_like(X)
        return 0.0, 0.0, 0.0, z.copy(), z.copy(), z.copy()

    S = X.T @ X
    S2 = S * S

    Q = np.outer(q, q)
    np.fill_diagonal(Q, 0.0)
    denom = 0.5 * float(np.sum(Q))

    if not np.isfinite(denom) or denom <= 0.0:
        z = np.zeros_like(X)
        return 0.0, 0.0, 0.0, z.copy(), z.copy(), z.copy()

    comm_pair = 2.0 * S2 * (1.0 - S2)
    dup_pair = S2 * S2
    orth_pair = S2

    np.fill_diagonal(comm_pair, 0.0)
    np.fill_diagonal(dup_pair, 0.0)
    np.fill_diagonal(orth_pair, 0.0)

    comm_loss = 0.5 * float(np.sum(Q * comm_pair)) / denom
    dup_loss = 0.5 * float(np.sum(Q * dup_pair)) / denom
    orth_loss = 0.5 * float(np.sum(Q * orth_pair)) / denom

    # Pair derivatives with respect to s:
    #   d[2 s^2(1-s^2)]/ds = 4 s (1 - 2 s^2)
    #   d[s^4]/ds           = 4 s^3
    #   d[s^2]/ds           = 2 s
    A_comm = Q * (4.0 * S * (1.0 - 2.0 * S2)) / denom
    A_dup = Q * (4.0 * S * S2) / denom
    A_orth = Q * (2.0 * S) / denom

    np.fill_diagonal(A_comm, 0.0)
    np.fill_diagonal(A_dup, 0.0)
    np.fill_diagonal(A_orth, 0.0)

    grad_comm_X = X @ A_comm
    grad_dup_X = X @ A_dup
    grad_orth_X = X @ A_orth

    return (
        comm_loss,
        dup_loss,
        orth_loss,
        grad_comm_X,
        grad_dup_X,
        grad_orth_X,
    )


def _eri_value_grad_X(
    X: np.ndarray,
    V: np.ndarray,
    vnorm2: float,
    rcond: float,
):
    """ERI variable-projection objective and gradient wrt normalized X."""
    Z = _build_Z(X)
    G = Z.T @ Z
    Ginv = np.linalg.pinv(G, rcond=rcond)
    B = Z.T @ V @ Z
    W = Ginv @ B @ Ginv
    W = 0.5 * (W + W.T)

    E = Z @ W @ Z.T - V
    eri_f = 0.5 * float(np.vdot(E, E).real) / vnorm2

    # Envelope theorem for the ERI least-squares objective.
    grad_Z = 2.0 * (E @ Z @ W) / vnorm2
    grad_X = np.empty_like(X)
    for a in range(X.shape[1]):
        A = grad_Z[:, a].reshape(X.shape[0], X.shape[0])
        grad_X[:, a] = (A + A.T) @ X[:, a]

    return eri_f, grad_X, W


def _self_consistent_components(
    X: np.ndarray,
    V: np.ndarray,
    vnorm2: float,
    rcond: float,
    comm_lambda: float,
    dup_lambda: float,
    activity_floor: float,
):
    """Evaluate Stage-2 components using q computed from W_opt(X)."""
    eri_f, _, W = _eri_value_grad_X(X, V, vnorm2, rcond)
    q = activity_from_W(W, floor=activity_floor)
    comm_f, dup_f, orth_f, *_ = weighted_geometric_losses_and_grads(X, q)
    total_f = eri_f + comm_lambda * comm_f + dup_lambda * dup_f
    return total_f, eri_f, comm_f, dup_f, orth_f, W, q


def _fit_diagnostics(X: np.ndarray, W: np.ndarray):
    gram_X = X.T @ X
    Z = _build_Z(X)
    gram_Z = Z.T @ Z
    gram_Z = 0.5 * (gram_Z + gram_Z.T)
    gram_Z_eigs = np.linalg.eigvalsh(gram_Z)
    gram_Z_condition = float(np.linalg.cond(gram_Z))
    W_fro = float(np.linalg.norm(W, "fro"))
    W_max = float(np.max(np.abs(W))) if W.size else 0.0
    return gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max


def fit_thc_from_eri(
    eri: np.ndarray,
    rank: int,
    *,
    n_starts: int = 4,
    maxiter: int = 500,
    outer_iters: int = 8,
    outer_tol: float = 1.0e-4,
    activity_mix: float = .5, #1.0,
    activity_floor: float = 0.0,
    seed: int = 7,
    rcond: float = 1.0e-10,
    ftol: float = 1.0e-13,
    gtol: float = 1.0e-8,
    comm_lambda: float = 0.0,
    dup_lambda: Optional[float] = None,
    init_X: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> THCFitResult:
    """Fit THC factors with the iteratively W-weighted Stage-2 objective.

    Parameters
    ----------
    comm_lambda
        Coefficient of the true W-weighted geometric commutator loss.
    dup_lambda
        Coefficient of the W-weighted duplicate/collinearity penalty.  If None,
        defaults to 2*comm_lambda, converting the combined frozen-q geometric
        penalty into a weighted orthogonality penalty.
    outer_iters
        Number of activity reweighting iterations.
    outer_tol
        Stop when ||q_new-q_old||/max(||q_old||,1) is below this value.
    activity_mix
        Mixing for activity updates: q <- (1-mix) q_old + mix q_new.
        Must lie in (0,1].
    activity_floor
        Optional relative floor for q_a.  Usually leave at zero.
    init_X
        Optional initial X for start 1.  Additional starts remain random.
    """
    eri = _check_eri(eri)
    m = eri.shape[0]
    if rank <= 0:
        raise ValueError("rank must be positive")
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    if outer_iters <= 0:
        raise ValueError("outer_iters must be positive")
    if not (0.0 < activity_mix <= 1.0):
        raise ValueError("activity_mix must be in (0,1]")
    if comm_lambda < 0.0:
        raise ValueError("comm_lambda must be nonnegative")

    if dup_lambda is None:
        dup_lambda = 2.0 * comm_lambda
    if dup_lambda < 0.0:
        raise ValueError("dup_lambda must be nonnegative")

    if init_X is not None:
        init_X = np.asarray(init_X, dtype=np.float64)
        if init_X.shape != (m, rank):
            raise ValueError(
                f"init_X must have shape {(m, rank)}, got {init_X.shape}"
            )
        init_X, _ = _normalize_columns(init_X)

    V = eri.reshape(m * m, m * m)
    vnorm2 = float(np.vdot(V, V).real)
    if vnorm2 == 0.0:
        X = np.eye(m, rank) if rank <= m else np.pad(
            np.eye(m), ((0, 0), (0, rank - m))
        )
        X, _ = _normalize_columns(X)
        W = np.zeros((rank, rank))
        q = np.zeros(rank)
        gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max = (
            _fit_diagnostics(X, W)
        )
        return THCFitResult(
            X=X,
            W=W,
            relative_error=0.0,
            objective=0.0,
            eri_objective=0.0,
            comm_objective=0.0,
            dup_objective=0.0,
            orth_objective=0.0,
            activity=q,
            outer_iterations=0,
            nit=0,
            success=True,
            message="zero ERI",
            gram_X=gram_X,
            gram_Z=gram_Z,
            gram_Z_eigs=gram_Z_eigs,
            gram_Z_condition=gram_Z_condition,
            W_fro=W_fro,
            W_max=W_max,
        )

    # With no geometric regularization there is no reason to reweight q.
    if comm_lambda == 0.0 and dup_lambda == 0.0:
        outer_iters = 1

    rng = np.random.default_rng(seed)
    best_data = None

    for istart in range(n_starts):
        if istart == 0 and init_X is not None:
            Y = init_X.copy()
            init_label = "provided init"
        else:
            Y = rng.standard_normal((m, rank))
            init_label = "random"

        X0, _ = _normalize_columns(Y)
        W0 = optimal_W(V, X0, rcond=rcond)
        q = activity_from_W(W0, floor=activity_floor)

        total_nit = 0
        last_res = None
        used_outer = 0

        if verbose:
            print(f"\nstart {istart + 1}/{n_starts} ({init_label})")

        for iouter in range(outer_iters):
            used_outer = iouter + 1
            q_fixed = q.copy()

            def value_and_grad(yflat: np.ndarray):
                Yloc = yflat.reshape(m, rank)
                X, norms = _normalize_columns(Yloc)

                eri_f, grad_eri_X, _ = _eri_value_grad_X(
                    X, V, vnorm2, rcond
                )
                (
                    comm_f,
                    dup_f,
                    _,
                    grad_comm_X,
                    grad_dup_X,
                    _,
                ) = weighted_geometric_losses_and_grads(X, q_fixed)

                f = eri_f + comm_lambda * comm_f + dup_lambda * dup_f
                grad_X = (
                    grad_eri_X
                    + comm_lambda * grad_comm_X
                    + dup_lambda * grad_dup_X
                )

                # Backpropagate through x_a = y_a / ||y_a||.
                radial = np.sum(X * grad_X, axis=0)
                grad_Y = (grad_X - X * radial[None, :]) / norms[None, :]
                return f, grad_Y.ravel()

            res = minimize(
                value_and_grad,
                Y.ravel(),
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": maxiter,
                    "ftol": ftol,
                    "gtol": gtol,
                    "maxls": 50,
                },
            )
            last_res = res
            total_nit += int(res.nit)
            Y = res.x.reshape(m, rank)
            X, _ = _normalize_columns(Y)

            # Self-consistent W and new activity after the frozen-q inner solve.
            (
                total_sc,
                eri_sc,
                comm_sc,
                dup_sc,
                orth_sc,
                W_sc,
                q_new,
            ) = _self_consistent_components(
                X,
                V,
                vnorm2,
                rcond,
                comm_lambda,
                dup_lambda,
                activity_floor,
            )

            q_rel = np.linalg.norm(q_new - q_fixed) / max(
                np.linalg.norm(q_fixed), 1.0e-16
            )

            if verbose:
                print(
                    f"  outer {iouter + 1}/{outer_iters}: "
                    f"inner_obj={res.fun:.6e}, self_obj={total_sc:.6e}, "
                    f"eri={eri_sc:.3e}, comm={comm_sc:.3e}, "
                    f"dup={dup_sc:.3e}, orth={orth_sc:.3e}, "
                    f"dq/q={q_rel:.3e}, nit={res.nit}, success={res.success}"
                )

            if q_rel < outer_tol:
                q = q_new
                break

            q = (1.0 - activity_mix) * q_fixed + activity_mix * q_new

        assert last_res is not None
        X, _ = _normalize_columns(Y)
        (
            total_sc,
            eri_sc,
            comm_sc,
            dup_sc,
            orth_sc,
            W_sc,
            q_sc,
        ) = _self_consistent_components(
            X,
            V,
            vnorm2,
            rcond,
            comm_lambda,
            dup_lambda,
            activity_floor,
        )

        if verbose:
            print(
                f"end start {istart + 1}: self_obj={total_sc:.6e}, "
                f"eri={eri_sc:.3e}, comm={comm_sc:.3e}, dup={dup_sc:.3e}, "
                f"orth={orth_sc:.3e}, outer={used_outer}, total_nit={total_nit}"
            )

        # Rank starts by the self-consistent Stage-2 objective.
        if best_data is None or total_sc < best_data["objective"]:
            best_data = {
                "X": X.copy(),
                "W": W_sc.copy(),
                "q": q_sc.copy(),
                "objective": float(total_sc),
                "eri": float(eri_sc),
                "comm": float(comm_sc),
                "dup": float(dup_sc),
                "orth": float(orth_sc),
                "outer": int(used_outer),
                "nit": int(total_nit),
                "success": bool(last_res.success),
                "message": str(last_res.message),
            }

    assert best_data is not None
    X = best_data["X"]
    W = best_data["W"]
    q = best_data["q"]

    Vfit = _build_Z(X) @ W @ _build_Z(X).T
    relerr = float(np.linalg.norm(Vfit - V) / np.linalg.norm(V))

    gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max = (
        _fit_diagnostics(X, W)
    )

    return THCFitResult(
        X=X,
        W=W,
        relative_error=relerr,
        objective=best_data["objective"],
        eri_objective=best_data["eri"],
        comm_objective=best_data["comm"],
        dup_objective=best_data["dup"],
        orth_objective=best_data["orth"],
        activity=q,
        outer_iterations=best_data["outer"],
        nit=best_data["nit"],
        success=best_data["success"],
        message=best_data["message"],
        gram_X=gram_X,
        gram_Z=gram_Z,
        gram_Z_eigs=gram_Z_eigs,
        gram_Z_condition=gram_Z_condition,
        W_fro=W_fro,
        W_max=W_max,
    )


def load_eri(path: str, key: str = "eri") -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        if key not in data:
            raise KeyError(f"{p} has keys {list(data.keys())}; requested key '{key}'")
        return data[key]
    raise ValueError("Input must be .npy or .npz")


def load_init_X(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    data = np.load(path)
    if isinstance(data, np.ndarray):
        X = data
    else:
        if "X" not in data:
            raise KeyError(f"Initialization file {path} does not contain key 'X'")
        X = data["X"]
    return np.asarray(X, dtype=np.float64)


def _gradient_self_test() -> None:
    """Finite-difference check for the frozen-q geometric gradient."""
    rng = np.random.default_rng(1234)
    m, rank = 4, 3
    Y = rng.standard_normal((m, rank))
    X, _ = _normalize_columns(Y)
    q = np.array([0.7, 1.3, 2.0])

    comm, dup, _, gc, gd, _ = weighted_geometric_losses_and_grads(X, q)
    lamc = 0.37
    lamd = 0.74
    gX = lamc * gc + lamd * gd

    # Project through column normalization, because the optimizer variable is Y.
    norms = np.linalg.norm(Y, axis=0)
    radial = np.sum(X * gX, axis=0)
    gY = (gX - X * radial[None, :]) / norms[None, :]

    def f_y(Ytrial):
        Xt, _ = _normalize_columns(Ytrial)
        c, d, *_ = weighted_geometric_losses_and_grads(Xt, q)
        return lamc * c + lamd * d

    eps = 1.0e-6
    gfd = np.zeros_like(Y)
    for i in range(m):
        for a in range(rank):
            Yp = Y.copy()
            Ym = Y.copy()
            Yp[i, a] += eps
            Ym[i, a] -= eps
            gfd[i, a] = (f_y(Yp) - f_y(Ym)) / (2.0 * eps)

    err = np.max(np.abs(gY - gfd))
    print(f"frozen-q geometric gradient max abs error = {err:.3e}")
    if err > 2.0e-6:
        raise RuntimeError("gradient self-test failed")


def _synthetic_self_test() -> int:
    _gradient_self_test()

    rng = np.random.default_rng(123)
    m, rank = 4, 5
    X0 = rng.standard_normal((m, rank))
    X0, _ = _normalize_columns(X0)
    W0 = rng.standard_normal((rank, rank))
    W0 = 0.5 * (W0 + W0.T)
    eri = reconstruct_eri(X0, W0)

    # ERI-only regression mode.
    fit = fit_thc_from_eri_stage2(
        eri,
        rank,
        n_starts=3,
        maxiter=800,
        outer_iters=1,
        seed=19,
        comm_lambda=0.0,
        dup_lambda=0.0,
        verbose=True,
    )
    print(f"synthetic relative reconstruction error = {fit.relative_error:.6e}")
    return 0 if fit.relative_error < 2.0e-3 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-2 W-weighted THC fit for LAFQMC commutator reduction."
    )
    parser.add_argument("input", nargs="?", help="ERI .npy or .npz file")
    parser.add_argument("--key", default="eri", help="ERI key for .npz input")
    parser.add_argument("--rank", type=int, help="THC rank / number of columns")
    parser.add_argument("--output", default="thc_stage2.npz", help="output .npz")
    parser.add_argument("--starts", type=int, default=4, help="number of starts")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="maximum L-BFGS iterations per outer reweighting step",
    )
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=8,
        help="maximum number of W-activity reweighting iterations",
    )
    parser.add_argument(
        "--outer-tol",
        type=float,
        default=1.0e-4,
        help="relative activity-change stopping tolerance",
    )
    parser.add_argument(
        "--activity-mix",
        type=float,
        default=1.0,
        help="activity update mixing in (0,1]",
    )
    parser.add_argument(
        "--activity-floor",
        type=float,
        default=0.0,
        help="optional relative floor for q_a; normally leave at zero",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--ftol", type=float, default=1.0e-13)
    parser.add_argument("--gtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--comm-lambda",
        type=float,
        default=0.0,
        help="coefficient of W-weighted true commutator geometry",
    )
    parser.add_argument(
        "--dup-lambda",
        type=float,
        default=None,
        help=(
            "coefficient of W-weighted s^4 duplicate penalty; "
            "default is 2*comm_lambda"
        ),
    )
    parser.add_argument(
        "--init",
        default=None,
        help="optional .npz/.npy containing initial X; used for start 1",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _synthetic_self_test()
    if args.input is None or args.rank is None:
        parser.error("input and --rank are required unless --self-test is used")

    dup_lambda = (
        2.0 * args.comm_lambda if args.dup_lambda is None else args.dup_lambda
    )

    eri = load_eri(args.input, key=args.key)
    init_X = load_init_X(args.init)

    fit = fit_thc_from_eri_stage2(
        eri,
        args.rank,
        n_starts=args.starts,
        maxiter=args.maxiter,
        outer_iters=args.outer_iters,
        outer_tol=args.outer_tol,
        activity_mix=args.activity_mix,
        activity_floor=args.activity_floor,
        seed=args.seed,
        rcond=args.rcond,
        ftol=args.ftol,
        gtol=args.gtol,
        comm_lambda=args.comm_lambda,
        dup_lambda=dup_lambda,
        init_X=init_X,
        verbose=True,
    )

    print("\nBEST FIT")
    print("X:")
    print(fit.X)
    print("W:")
    print(fit.W)
    print("activity q = sum_b |W_ab|:")
    print(fit.activity)
    print(f"relative ERI reconstruction error = {fit.relative_error:.8e}")
    print(f"total objective = {fit.objective:.8e}")
    print(f"eri objective = {fit.eri_objective:.8e}")
    print(f"weighted comm objective = {fit.comm_objective:.8e}")
    print(f"weighted duplicate objective = {fit.dup_objective:.8e}")
    print(f"weighted orth objective = {fit.orth_objective:.8e}")
    print(f"outer iterations = {fit.outer_iterations}")
    print(f"total inner nit = {fit.nit}")
    print(f"optimizer success = {fit.success}: {fit.message}")
    print("X^T X:")
    print(fit.gram_X)
    print("eig(Z^T Z):")
    print(fit.gram_Z_eigs)
    print(f"cond(Z^T Z) = {fit.gram_Z_condition:.8e}")
    print(f"||W||_F = {fit.W_fro:.8e}")
    print(f"max |W_ab| = {fit.W_max:.8e}")

    np.savez(
        args.output,
        X=fit.X,
        W=fit.W,
        q_activity=fit.activity,
        relative_error=np.array(fit.relative_error),
        objective=np.array(fit.objective),
        eri_objective=np.array(fit.eri_objective),
        comm_objective=np.array(fit.comm_objective),
        dup_objective=np.array(fit.dup_objective),
        orth_objective=np.array(fit.orth_objective),
        comm_lambda=np.array(args.comm_lambda),
        dup_lambda=np.array(dup_lambda),
        outer_iterations=np.array(fit.outer_iterations),
        gram_X=fit.gram_X,
        gram_Z=fit.gram_Z,
        gram_Z_eigs=fit.gram_Z_eigs,
        gram_Z_condition=np.array(fit.gram_Z_condition),
        W_fro=np.array(fit.W_fro),
        W_max=np.array(fit.W_max),
        eri_order=np.array("prqs"),
    )
    print(f"saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
