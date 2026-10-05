#!/usr/bin/env python3
"""Algebraic tensor-hypercontraction (THC) fit from a molecular ERI tensor.

This script is intended for *molecular / finite* real two-electron integrals that
are already available as a dense four-index tensor

    eri[p, r, q, s] = (p r | q s)

and fits

    (p r | q s) ~= sum_{a,b} X[p,a] X[r,a] W[a,b] X[q,b] X[s,b].

Important
---------
This is NOT a literal Python port of the supplied libpbc ISDF code.  That code
constructs X from AO values at selected real-space interpolation points and W
from interpolation vectors and a Coulomb kernel.  Those data cannot in general
be recovered uniquely from the ERI tensor alone.

Instead, this file performs an algebraic least-squares THC fit directly to the
ERI tensor.  It is useful for testing LAFQMC/THC machinery when the full ERI is
already available.  For production-size calculations, a real-space ISDF/THC
implementation is preferable because it avoids O(M^4) storage/work.

The fit uses variable projection.  For fixed X, define

    Z[(p,r),a] = X[p,a] X[r,a].

Then V ~= Z W Z^T, where V[(p,r),(q,s)] = eri[p,r,q,s].  The optimal W for
fixed X is obtained from the normal equations

    G W G = B,
    G = Z^T Z,
    B = Z^T V Z.

Only X is optimized nonlinearly.  Columns of X are normalized to remove the
trivial column-scaling gauge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


@dataclass
class THCFitResult:
    X: np.ndarray
    W: np.ndarray
    relative_error: float
    objective: float
    nit: int
    success: bool
    message: str


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

    # In the requested ordering V[(p,r),(q,s)] = (pr|qs), V should be symmetric.
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
    norms = np.linalg.norm(Y, axis=0)
    norms = np.maximum(norms, floor)
    return Y / norms[None, :], norms


def _build_Z(X: np.ndarray) -> np.ndarray:
    """Z[(p,r),a] = X[p,a] X[r,a]."""
    m, rank = X.shape
    return np.einsum("pa,ra->pra", X, X, optimize=True).reshape(m * m, rank)


def optimal_W(V: np.ndarray, X: np.ndarray, rcond: float = 1.0e-10) -> np.ndarray:
    """Least-squares-optimal symmetric W for fixed X."""
    Z = _build_Z(X)
    G = Z.T @ Z
    Ginv = np.linalg.pinv(G, rcond=rcond)
    B = Z.T @ V @ Z
    W = Ginv @ B @ Ginv
    return 0.5 * (W + W.T)


def reconstruct_eri(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return eri_fit[p,r,q,s] from THC factors."""
    m = X.shape[0]
    Z = _build_Z(X)
    return (Z @ W @ Z.T).reshape(m, m, m, m)


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
    verbose: bool = True,
) -> THCFitResult:
    """Fit real THC factors X,W to eri[p,r,q,s]=(pr|qs).

    Parameters
    ----------
    eri
        Dense real ERI tensor in the exact axis order ``[p,r,q,s]``.
    rank
        Number of THC interpolation/factor columns (the alpha dimension).
    n_starts
        Number of random L-BFGS starts.  THC fitting is non-convex; multiple
        starts are strongly recommended.
    maxiter
        Maximum L-BFGS iterations per start.
    seed
        RNG seed.
    rcond
        Pseudoinverse cutoff used when solving for the optimal W.

    Returns
    -------
    THCFitResult
    """
    eri = _check_eri(eri)
    m = eri.shape[0]
    if rank <= 0:
        raise ValueError("rank must be positive")

    V = eri.reshape(m * m, m * m)
    vnorm2 = float(np.vdot(V, V).real)
    if vnorm2 == 0.0:
        X = np.eye(m, rank) if rank <= m else np.pad(np.eye(m), ((0, 0), (0, rank - m)))
        W = np.zeros((rank, rank))
        return THCFitResult(X, W, 0.0, 0.0, 0, True, "zero ERI")

    rng = np.random.default_rng(seed)

    def value_and_grad(yflat: np.ndarray):
        Y = yflat.reshape(m, rank)
        X, norms = _normalize_columns(Y)
        Z = _build_Z(X)

        # Variable projection: optimize W analytically for the current X.
        G = Z.T @ Z
        Ginv = np.linalg.pinv(G, rcond=rcond)
        B = Z.T @ V @ Z
        W = Ginv @ B @ Ginv
        W = 0.5 * (W + W.T)

        E = Z @ W @ Z.T - V
        f = 0.5 * float(np.vdot(E, E).real) / vnorm2

        # Envelope theorem: because W is the LS optimum for this X, the
        # derivative of the minimized objective is the partial derivative at W.
        grad_Z = 2.0 * (E @ Z @ W) / vnorm2

        grad_X = np.empty_like(X)
        for a in range(rank):
            A = grad_Z[:, a].reshape(m, m)
            grad_X[:, a] = (A + A.T) @ X[:, a]

        # Backpropagate through X[:,a] = Y[:,a] / ||Y[:,a]||.
        radial = np.sum(X * grad_X, axis=0)
        grad_Y = (grad_X - X * radial[None, :]) / norms[None, :]
        return f, grad_Y.ravel()

    best = None
    for istart in range(n_starts):
        # Unit-scale random initialization; column normalization is handled
        # inside the objective.
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
            print(
                f"start {istart + 1}/{n_starts}: "
                f"objective={res.fun:.6e}, nit={res.nit}, success={res.success}"
            )
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    Y = best.x.reshape(m, rank)
    X, _ = _normalize_columns(Y)
    W = optimal_W(V, X, rcond=rcond)

    Vfit = _build_Z(X) @ W @ _build_Z(X).T
    relerr = float(np.linalg.norm(Vfit - V) / np.linalg.norm(V))

    return THCFitResult(
        X=X,
        W=W,
        relative_error=relerr,
        objective=float(best.fun),
        nit=int(best.nit),
        success=bool(best.success),
        message=str(best.message),
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


def _synthetic_self_test() -> int:
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
        verbose=True,
    )
    print(f"synthetic relative reconstruction error = {fit.relative_error:.6e}")
    # This is a nonconvex fit, so use a modest test threshold.
    return 0 if fit.relative_error < 5.0e-4 else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit molecular THC factors X,W from eri[p,r,q,s]=(pr|qs)."
    )
    parser.add_argument("input", nargs="?", help="ERI .npy or .npz file")
    parser.add_argument("--key", default="eri", help="ERI key for .npz input")
    parser.add_argument("--rank", type=int, help="THC rank / number of alpha points")
    parser.add_argument("--output", default="thc_factors.npz", help="output .npz")
    parser.add_argument("--starts", type=int, default=4, help="random starts")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rcond", type=float, default=1.0e-10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return _synthetic_self_test()
    if args.input is None or args.rank is None:
        parser.error("input and --rank are required unless --self-test is used")

    eri = load_eri(args.input, key=args.key)
    fit = fit_thc_from_eri(
        eri,
        args.rank,
        n_starts=args.starts,
        maxiter=args.maxiter,
        seed=args.seed,
        rcond=args.rcond,
        verbose=True,
    )

    np.savez(
        args.output,
        X=fit.X,
        W=fit.W,
        relative_error=np.array(fit.relative_error),
        eri_order=np.array("prqs"),
    )
    print(f"saved {args.output}")
    print(f"X shape = {fit.X.shape}")
    print(f"W shape = {fit.W.shape}")
    print(f"relative ERI reconstruction error = {fit.relative_error:.8e}")
    print(f"optimizer success = {fit.success}: {fit.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
