#!/usr/bin/env python3
"""Algebraic THC fit with a Stage-1 commutator/duplicate regularizer.

Fits a real molecular ERI tensor in the ordering

    eri[p, r, q, s] = (p r | q s)

with

    (p r | q s) ~= sum_{a,b} X[p,a] X[r,a] W[a,b] X[q,b] X[s,b].

The fit uses variable projection: for fixed column-normalized X,

    Z[(p,r),a] = X[p,a] X[r,a]
    V ~= Z W Z^T

and the least-squares-optimal symmetric W is solved analytically.

In addition to the ERI reconstruction loss, this version can add two
X-only geometric penalties:

    L_comm = average_{a<b} 2 s_ab^2 (1 - s_ab^2)
    L_dup  = average_{a<b} s_ab^4

where s_ab = x_a^T x_b and each x_a is unit-normalized.

The total objective is

    L_total = L_ERI + comm_lambda * L_comm + dup_lambda * L_dup.

If dup_lambda is omitted, it defaults to

    dup_lambda = 2 * comm_lambda,

so the geometric contribution becomes exactly

    2 * comm_lambda * average_{a<b} s_ab^2,

which is a pure orthogonality penalty.  This prevents the pathological
escape in which the true projector-commutator loss is made small by driving
columns toward collinearity, x_b ~= +/- x_a, which makes Z nearly singular
and can cause very large, cancelling entries in W.

The code also reports and saves diagnostics for this pathology:

    eig(Z^T Z), cond(Z^T Z), ||W||_F, max|W_ab|.

Setting comm_lambda=0 and dup_lambda=0 recovers the original ERI-only fit.
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
            raise NotImplementedError(
                "This reference fitter currently supports real molecular ERIs only."
            )
        eri = eri.real

    eri = np.asarray(eri, dtype=np.float64)

    # V[(p,r),(q,s)] = (pr|qs) should be symmetric.
    V = eri.reshape(m * m, m * m)
    rel_pair_sym = np.linalg.norm(V - V.T) / max(np.linalg.norm(V), 1.0)
    if rel_pair_sym > sym_tol:
        raise ValueError(
            "ERI is not symmetric under (p,r)<->(q,s) in the supplied axis order. "
            f"Relative pair-space asymmetry = {rel_pair_sym:.3e}. "
            "This script expects eri[p,r,q,s] = (pr|qs)."
        )

    return eri


def _normalize_columns(Y: np.ndarray, floor: float = 1.0e-12):
    """Normalize columns of Y and return (X, original_norms)."""
    norms = np.linalg.norm(Y, axis=0)
    norms = np.maximum(norms, floor)
    return Y / norms[None, :], norms


def _build_Z(X: np.ndarray) -> np.ndarray:
    """Return Z[(p,r),a] = X[p,a] X[r,a]."""
    m, rank = X.shape
    return np.einsum("pa,ra->pra", X, X, optimize=True).reshape(m * m, rank)


def optimal_W(V: np.ndarray, X: np.ndarray, rcond: float = 1.0e-10) -> np.ndarray:
    """Least-squares-optimal symmetric W for fixed, normalized X."""
    Z = _build_Z(X)
    G = Z.T @ Z
    Ginv = np.linalg.pinv(G, rcond=rcond)
    B = Z.T @ V @ Z
    W = Ginv @ B @ Ginv
    return 0.5 * (W + W.T)


def reconstruct_eri(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return eri_fit[p,r,q,s] reconstructed from THC factors."""
    m = X.shape[0]
    Z = _build_Z(X)
    return (Z @ W @ Z.T).reshape(m, m, m, m)


def geometric_losses_and_grads(
    X: np.ndarray,
    *,
    normalize_pairs: bool = True,
):
    """Return geometric losses and gradients with respect to normalized X.

    For unit-normalized columns x_a and s_ab = x_a^T x_b,

        L_comm = sum_{a<b} 2 s_ab^2 (1 - s_ab^2)
        L_dup  = sum_{a<b} s_ab^4
        L_orth = sum_{a<b} s_ab^2

    If normalize_pairs=True, each is divided by rank*(rank-1)/2.

    Returns
    -------
    comm_loss, dup_loss, orth_loss, grad_comm_X, grad_dup_X, grad_orth_X
    """
    rank = X.shape[1]

    if rank < 2:
        z = np.zeros_like(X)
        return 0.0, 0.0, 0.0, z.copy(), z.copy(), z.copy()

    S = X.T @ X
    S2 = S * S

    # Off-diagonal masks are implemented by zeroing the diagonal.
    comm_pair = 2.0 * S2 * (1.0 - S2)
    dup_pair = S2 * S2
    orth_pair = S2.copy()

    np.fill_diagonal(comm_pair, 0.0)
    np.fill_diagonal(dup_pair, 0.0)
    np.fill_diagonal(orth_pair, 0.0)

    # 1/2 converts the symmetric full matrix sum to a<b.
    comm_loss = 0.5 * float(np.sum(comm_pair))
    dup_loss = 0.5 * float(np.sum(dup_pair))
    orth_loss = 0.5 * float(np.sum(orth_pair))

    # For a<b pair contributions:
    # d/ds [2 s^2 (1-s^2)] = 4 s (1 - 2 s^2)
    # d/ds [s^4]           = 4 s^3
    # d/ds [s^2]           = 2 s
    A_comm = 4.0 * S * (1.0 - 2.0 * S2)
    A_dup = 4.0 * S * S2
    A_orth = 2.0 * S

    np.fill_diagonal(A_comm, 0.0)
    np.fill_diagonal(A_dup, 0.0)
    np.fill_diagonal(A_orth, 0.0)

    # Column-a gradient is sum_{b!=a} f'(s_ab) x_b.
    grad_comm_X = X @ A_comm
    grad_dup_X = X @ A_dup
    grad_orth_X = X @ A_orth

    if normalize_pairs:
        npairs = rank * (rank - 1) / 2.0
        comm_loss /= npairs
        dup_loss /= npairs
        orth_loss /= npairs
        grad_comm_X /= npairs
        grad_dup_X /= npairs
        grad_orth_X /= npairs

    return (
        comm_loss,
        dup_loss,
        orth_loss,
        grad_comm_X,
        grad_dup_X,
        grad_orth_X,
    )


def _fit_diagnostics(X: np.ndarray, W: np.ndarray):
    """Return Gram/conditioning and W-size diagnostics."""
    gram_X = X.T @ X
    Z = _build_Z(X)
    gram_Z = Z.T @ Z

    # gram_Z is PSD up to roundoff.
    gram_Z_eigs = np.linalg.eigvalsh(0.5 * (gram_Z + gram_Z.T))
    gram_Z_condition = float(np.linalg.cond(gram_Z))
    W_fro = float(np.linalg.norm(W, "fro"))
    W_max = float(np.max(np.abs(W))) if W.size else 0.0

    return gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max


def _evaluate_components(
    yflat: np.ndarray,
    *,
    m: int,
    rank: int,
    V: np.ndarray,
    vnorm2: float,
    rcond: float,
    comm_lambda: float,
    dup_lambda: float,
    normalize_geom: bool,
):
    """Evaluate objective components at yflat without computing gradients."""
    Y = yflat.reshape(m, rank)
    X, _ = _normalize_columns(Y)
    Z = _build_Z(X)

    G = Z.T @ Z
    Ginv = np.linalg.pinv(G, rcond=rcond)
    B = Z.T @ V @ Z
    W = Ginv @ B @ Ginv
    W = 0.5 * (W + W.T)

    E = Z @ W @ Z.T - V
    eri_f = 0.5 * float(np.vdot(E, E).real) / vnorm2

    comm_f, dup_f, orth_f, _, _, _ = geometric_losses_and_grads(
        X, normalize_pairs=normalize_geom
    )

    total_f = eri_f + comm_lambda * comm_f + dup_lambda * dup_f
    return total_f, eri_f, comm_f, dup_f, orth_f, X, W


def fit_thc_from_eri(
    eri: np.ndarray,
    rank: int,
    *,
    n_starts: int = 4,
    maxiter: int = 1000,
    seed: int = 7,
    rcond: float = 1.0e-10,
    ftol: float = 1.0e-13,
    gtol: float = 1.0e-8,
    comm_lambda: float = 0.0,
    dup_lambda: Optional[float] = None,
    normalize_geom: bool = True,
    verbose: bool = True,
) -> THCFitResult:
    """Fit real THC factors X,W to eri[p,r,q,s]=(pr|qs).

    Parameters
    ----------
    eri
        Dense real ERI tensor in exact axis order [p,r,q,s].
    rank
        THC rank / number of alpha columns.
    n_starts
        Number of random L-BFGS starts.
    maxiter
        Maximum L-BFGS iterations per start.
    seed
        Random seed.
    rcond
        Pseudoinverse cutoff used to solve for optimal W.
    ftol, gtol
        L-BFGS-B convergence tolerances.
    comm_lambda
        Coefficient multiplying the true rank-one projector-commutator loss
        average 2 s_ab^2 (1-s_ab^2).
    dup_lambda
        Coefficient multiplying average s_ab^4. If None, use
        2*comm_lambda, so the total geometric penalty is proportional to
        average s_ab^2 and no longer rewards collinear columns.
    normalize_geom
        Divide geometric losses by the number of column pairs.
    verbose
        Print per-start and final diagnostics.

    Notes
    -----
    The geometric losses depend only on X. Therefore the envelope-theorem
    gradient for the ERI variable-projection part remains valid; there is no
    need to differentiate W(X) for this Stage-1 regularization.
    """
    eri = _check_eri(eri)
    m = eri.shape[0]

    if rank <= 0:
        raise ValueError("rank must be positive")
    if comm_lambda < 0.0:
        raise ValueError("comm_lambda must be nonnegative")

    if dup_lambda is None:
        dup_lambda = 2.0 * comm_lambda
    if dup_lambda < 0.0:
        raise ValueError("dup_lambda must be nonnegative")

    V = eri.reshape(m * m, m * m)
    vnorm2 = float(np.vdot(V, V).real)

    if vnorm2 == 0.0:
        if rank <= m:
            X = np.eye(m, rank)
        else:
            X = np.zeros((m, rank))
            X[:, :m] = np.eye(m)
            # Extra unit columns for completeness; W=0 makes them irrelevant.
            for a in range(m, rank):
                X[a % m, a] = 1.0

        W = np.zeros((rank, rank))
        comm_f, dup_f, orth_f, _, _, _ = geometric_losses_and_grads(
            X, normalize_pairs=normalize_geom
        )
        gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max = (
            _fit_diagnostics(X, W)
        )
        total = comm_lambda * comm_f + dup_lambda * dup_f

        return THCFitResult(
            X=X,
            W=W,
            relative_error=0.0,
            objective=float(total),
            eri_objective=0.0,
            comm_objective=float(comm_f),
            dup_objective=float(dup_f),
            orth_objective=float(orth_f),
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

    rng = np.random.default_rng(seed)

    def value_and_grad(yflat: np.ndarray):
        Y = yflat.reshape(m, rank)
        X, norms = _normalize_columns(Y)
        Z = _build_Z(X)

        # Variable projection: solve W analytically for current X.
        G = Z.T @ Z
        Ginv = np.linalg.pinv(G, rcond=rcond)
        B = Z.T @ V @ Z
        W = Ginv @ B @ Ginv
        W = 0.5 * (W + W.T)

        # ERI reconstruction objective.
        E = Z @ W @ Z.T - V
        eri_f = 0.5 * float(np.vdot(E, E).real) / vnorm2

        # Envelope theorem for the ERI-only variable-projection part.
        grad_Z = 2.0 * (E @ Z @ W) / vnorm2

        grad_eri_X = np.empty_like(X)
        for a in range(rank):
            A = grad_Z[:, a].reshape(m, m)
            grad_eri_X[:, a] = (A + A.T) @ X[:, a]

        # Stage-1 X-only geometric penalties.
        (
            comm_f,
            dup_f,
            _orth_f,
            grad_comm_X,
            grad_dup_X,
            _grad_orth_X,
        ) = geometric_losses_and_grads(X, normalize_pairs=normalize_geom)

        total_f = eri_f + comm_lambda * comm_f + dup_lambda * dup_f
        grad_X = (
            grad_eri_X
            + comm_lambda * grad_comm_X
            + dup_lambda * grad_dup_X
        )

        # Backpropagate through x_a = y_a / ||y_a||.
        radial = np.sum(X * grad_X, axis=0)
        grad_Y = (grad_X - X * radial[None, :]) / norms[None, :]

        return total_f, grad_Y.ravel()

    best = None

    for istart in range(n_starts):
        Y0 = rng.standard_normal((m, rank))
        res = minimize(
            value_and_grad,
            Y0.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxls": 50,
            },
        )

        if verbose:
            total_f, eri_f, comm_f, dup_f, orth_f, Xs, Ws = _evaluate_components(
                res.x,
                m=m,
                rank=rank,
                V=V,
                vnorm2=vnorm2,
                rcond=rcond,
                comm_lambda=comm_lambda,
                dup_lambda=dup_lambda,
                normalize_geom=normalize_geom,
            )
            _, Gz, eigs, cond, Wfro, Wmax = _fit_diagnostics(Xs, Ws)
            print(
                f"start {istart + 1}/{n_starts}: "
                f"total={total_f:.6e}, eri={eri_f:.6e}, "
                f"comm={comm_f:.6e}, dup={dup_f:.6e}, orth={orth_f:.6e}, "
                f"cond(ZTZ)={cond:.3e}, max|W|={Wmax:.3e}, "
                f"nit={res.nit}, success={res.success}"
            )
            if eigs.size:
                print(
                    f"    eig(ZTZ): min={eigs[0]:.3e}, max={eigs[-1]:.3e}; "
                    f"||W||_F={Wfro:.3e}"
                )

        if best is None or res.fun < best.fun:
            best = res

    assert best is not None

    # Reconstruct best factors and all diagnostics.
    (
        total_f,
        eri_f,
        comm_f,
        dup_f,
        orth_f,
        X,
        W,
    ) = _evaluate_components(
        best.x,
        m=m,
        rank=rank,
        V=V,
        vnorm2=vnorm2,
        rcond=rcond,
        comm_lambda=comm_lambda,
        dup_lambda=dup_lambda,
        normalize_geom=normalize_geom,
    )

    Z = _build_Z(X)
    Vfit = Z @ W @ Z.T
    relerr = float(np.linalg.norm(Vfit - V) / np.linalg.norm(V))

    gram_X, gram_Z, gram_Z_eigs, gram_Z_condition, W_fro, W_max = (
        _fit_diagnostics(X, W)
    )

    if verbose:
        print("\nBest-fit diagnostics")
        print("--------------------")
        print(f"total objective       = {total_f:.12e}")
        print(f"ERI objective         = {eri_f:.12e}")
        print(f"comm objective        = {comm_f:.12e}")
        print(f"duplicate objective   = {dup_f:.12e}")
        print(f"orth objective        = {orth_f:.12e}")
        print(f"relative ERI error    = {relerr:.12e}")
        print(f"cond(Z^T Z)           = {gram_Z_condition:.12e}")
        if gram_Z_eigs.size:
            print(f"min eig(Z^T Z)        = {gram_Z_eigs[0]:.12e}")
            print(f"max eig(Z^T Z)        = {gram_Z_eigs[-1]:.12e}")
        print(f"||W||_F               = {W_fro:.12e}")
        print(f"max |W_ab|            = {W_max:.12e}")

    return THCFitResult(
        X=X,
        W=W,
        relative_error=relerr,
        objective=float(total_f),
        eri_objective=float(eri_f),
        comm_objective=float(comm_f),
        dup_objective=float(dup_f),
        orth_objective=float(orth_f),
        nit=int(best.nit),
        success=bool(best.success),
        message=str(best.message),
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


def _finite_difference_gradient_test() -> int:
    """Check the analytic X-gradient of the geometric loss combination."""
    rng = np.random.default_rng(101)
    m, rank = 4, 5
    Y = rng.standard_normal((m, rank))
    X, _ = _normalize_columns(Y)

    comm_lambda = 0.7
    dup_lambda = 1.4

    comm, dup, _, gc, gd, _ = geometric_losses_and_grads(
        X, normalize_pairs=True
    )
    analytic = comm_lambda * gc + dup_lambda * gd

    eps = 1.0e-6
    numeric = np.zeros_like(X)

    def f_of_X(Q):
        c, d, _, _, _, _ = geometric_losses_and_grads(Q, normalize_pairs=True)
        return comm_lambda * c + dup_lambda * d

    for p in range(m):
        for a in range(rank):
            Xp = X.copy()
            Xm = X.copy()
            Xp[p, a] += eps
            Xm[p, a] -= eps
            numeric[p, a] = (f_of_X(Xp) - f_of_X(Xm)) / (2.0 * eps)

    err = np.max(np.abs(analytic - numeric))
    print(f"geometric gradient max abs error = {err:.6e}")
    print(f"base comm={comm:.6e}, dup={dup:.6e}")
    return 0 if err < 1.0e-6 else 1


def _synthetic_self_test() -> int:
    """Regression test: comm=dup=0 should retain the original ERI fitting behavior."""
    rng = np.random.default_rng(123)
    m, rank = 4, 6
    X0 = rng.standard_normal((m, rank))
    W0 = rng.standard_normal((rank, rank))
    W0 = W0 @ W0.T
    eri = reconstruct_eri(X0, W0)

    fit = fit_thc_from_eri(
        eri,
        rank,
        n_starts=6,
        maxiter=1500,
        seed=19,
        comm_lambda=0.0,
        dup_lambda=0.0,
        verbose=True,
    )

    print(f"synthetic relative reconstruction error = {fit.relative_error:.6e}")
    return 0 if fit.relative_error < 5.0e-4 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit molecular THC factors X,W from eri[p,r,q,s]=(pr|qs), with "
            "optional Stage-1 commutator/duplicate regularization."
        )
    )

    parser.add_argument("input", nargs="?", help="ERI .npy or .npz file")
    parser.add_argument("--key", default="eri", help="ERI key for .npz input")
    parser.add_argument("--rank", type=int, help="THC rank / number of alpha columns")
    parser.add_argument("--output", default="thc_factors_regularized.npz", help="output .npz")
    parser.add_argument("--starts", type=int, default=4, help="random starts")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--ftol", type=float, default=1.0e-13)
    parser.add_argument("--gtol", type=float, default=1.0e-8)

    parser.add_argument(
        "--comm-lambda",
        type=float,
        default=0.0,
        help=(
            "Coefficient of average 2*s_ab^2*(1-s_ab^2), where "
            "s_ab = x_a^T x_b."
        ),
    )
    parser.add_argument(
        "--dup-lambda",
        type=float,
        default=None,
        help=(
            "Coefficient of average s_ab^4. If omitted, defaults to "
            "2*comm_lambda; then the combined geometric penalty is a pure "
            "orthogonality penalty proportional to average s_ab^2."
        ),
    )
    parser.add_argument(
        "--unnormalized-geom",
        action="store_true",
        help="Do not divide geometric losses by the number of THC column pairs.",
    )

    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gradient-test", action="store_true")

    args = parser.parse_args(argv)

    if args.gradient_test:
        return _finite_difference_gradient_test()

    if args.self_test:
        return _synthetic_self_test()

    if args.input is None or args.rank is None:
        parser.error("input and --rank are required unless a test flag is used")

    dup_lambda = (
        2.0 * args.comm_lambda if args.dup_lambda is None else args.dup_lambda
    )

    eri = load_eri(args.input, key=args.key)

    fit = fit_thc_from_eri(
        eri,
        args.rank,
        n_starts=args.starts,
        maxiter=args.maxiter,
        seed=args.seed,
        rcond=args.rcond,
        ftol=args.ftol,
        gtol=args.gtol,
        comm_lambda=args.comm_lambda,
        dup_lambda=dup_lambda,
        normalize_geom=not args.unnormalized_geom,
        verbose=True,
    )

    np.savez(
        args.output,
        X=fit.X,
        W=fit.W,
        relative_error=np.array(fit.relative_error),
        total_objective=np.array(fit.objective),
        eri_objective=np.array(fit.eri_objective),
        comm_objective=np.array(fit.comm_objective),
        dup_objective=np.array(fit.dup_objective),
        orth_objective=np.array(fit.orth_objective),
        comm_lambda=np.array(args.comm_lambda),
        dup_lambda=np.array(dup_lambda),
        gram_X=fit.gram_X,
        gram_Z=fit.gram_Z,
        gram_Z_eigs=fit.gram_Z_eigs,
        gram_Z_condition=np.array(fit.gram_Z_condition),
        W_fro=np.array(fit.W_fro),
        W_max=np.array(fit.W_max),
        eri_order=np.array("prqs"),
    )

    print(f"\nsaved {args.output}")
    print(f"X shape = {fit.X.shape}")
    print(f"W shape = {fit.W.shape}")
    print(f"relative ERI reconstruction error = {fit.relative_error:.8e}")
    print(f"optimizer success = {fit.success}: {fit.message}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
