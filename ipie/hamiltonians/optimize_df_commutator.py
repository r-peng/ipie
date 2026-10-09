#!/usr/bin/env python3
"""Commutator-regularized compressed double factorization for molecular ERIs.

Representation (input ordering eri[p,r,q,s] = (pr|qs)):

    eri_fit[p,r,q,s]
      = sum_t sum_kl U[t,p,k] U[t,r,k] Z[t,k,l]
                         U[t,q,l] U[t,s,l]

Each U[t] is orthogonal and each Z[t] is a general real symmetric matrix.
No low-rank constraint is imposed on Z[t].

The objective is

    L = 0.5 ||Vfit - V||_F^2 / ||V||_F^2
        + comm_lambda * Cw_proxy
        + 0.5 * rho_z * sum_t ||Z[t]||_F^2 / ||V||_F^2,

where Cw_proxy uses q[t,k] = sum_l |Z[t,k,l]| and the projector
commutator geometry between different leaves.

Optimization alternates:
  (1) exact/CG regularized least-squares solve for full symmetric Z at fixed U;
  (2) orthogonal-manifold gradient steps for U at fixed Z and frozen q.

This is a reference implementation intended for small-to-medium tests. For large
systems, initialize from Cholesky factors and avoid constructing dense M^2 x M^2
pair-space ERI matrices.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, cg


@dataclass
class DFResult:
    U: np.ndarray
    Z: np.ndarray
    eri_relative_error: float
    eri_objective: float
    comm_objective: float
    z_regularizer: float
    total_objective: float
    outer_iterations: int


def _check_eri(eri: np.ndarray, sym_tol: float = 1e-8) -> np.ndarray:
    eri = np.asarray(eri)
    if eri.ndim != 4:
        raise ValueError(f"ERI must be rank 4, got shape {eri.shape}")
    m = eri.shape[0]
    if eri.shape != (m, m, m, m):
        raise ValueError(f"ERI must have shape (M,M,M,M), got {eri.shape}")
    if np.iscomplexobj(eri):
        if np.max(np.abs(eri.imag)) > sym_tol:
            raise NotImplementedError("This reference code supports real ERIs only.")
        eri = eri.real
    eri = np.asarray(eri, dtype=np.float64)
    V = eri.reshape(m * m, m * m)
    rel = np.linalg.norm(V - V.T) / max(np.linalg.norm(V), 1.0)
    if rel > sym_tol:
        raise ValueError(
            "Expected eri[p,r,q,s]=(pr|qs), symmetric under (p,r)<->(q,s); "
            f"relative pair-space asymmetry={rel:.3e}"
        )
    return eri


def build_A(Ut: np.ndarray) -> np.ndarray:
    """A[(p,r),k] = U[p,k] U[r,k]."""
    m = Ut.shape[0]
    return np.einsum("pk,rk->prk", Ut, Ut, optimize=True).reshape(m * m, m)


def reconstruct_pair(U: np.ndarray, Z: np.ndarray) -> np.ndarray:
    nt, m, _ = U.shape
    out = np.zeros((m * m, m * m), dtype=np.float64)
    for t in range(nt):
        A = build_A(U[t])
        out += A @ Z[t] @ A.T
    return 0.5 * (out + out.T)


def reconstruct_eri(U: np.ndarray, Z: np.ndarray) -> np.ndarray:
    m = U.shape[1]
    return reconstruct_pair(U, Z).reshape(m, m, m, m)


def activities(Z: np.ndarray) -> np.ndarray:
    """q[t,k] = sum_l |Z[t,k,l]|."""
    return np.sum(np.abs(Z), axis=2)


def commutator_proxy(U: np.ndarray, q: Optional[np.ndarray] = None) -> tuple[float, float, float]:
    """Return (normalized weighted proxy, numerator, denominator).

    If q is None, all mode weights are one.
    """
    nt, m, _ = U.shape
    if q is None:
        q = np.ones((nt, m), dtype=np.float64)
    num = 0.0
    den = 0.0
    for t in range(nt):
        for s in range(t + 1, nt):
            S = U[t].T @ U[s]
            S2 = S * S
            geom = 2.0 * S2 * (1.0 - S2)
            W = np.outer(q[t], q[s])
            num += float(np.sum(W * geom))
            den += float(np.sum(W))
    val = num / den if den > 0.0 else 0.0
    return val, num, den


def geometric_commutator(U: np.ndarray) -> float:
    return commutator_proxy(U, q=None)[1]


def objective_parts(V: np.ndarray, U: np.ndarray, Z: np.ndarray,
                    comm_lambda: float, rho_z: float,
                    q_for_comm: Optional[np.ndarray] = None) -> dict:
    vnorm2 = float(np.vdot(V, V).real)
    Vfit = reconstruct_pair(U, Z)
    E = Vfit - V
    eri_obj = 0.5 * float(np.vdot(E, E).real) / vnorm2
    if q_for_comm is None:
        q_for_comm = activities(Z)
    comm_obj = commutator_proxy(U, q_for_comm)[0]
    zreg = 0.5 * rho_z * float(np.vdot(Z, Z).real) / vnorm2
    total = eri_obj + comm_lambda * comm_obj + zreg
    return {
        "eri_objective": eri_obj,
        "comm_objective": comm_obj,
        "z_regularizer": zreg,
        "total_objective": total,
        "Vfit": Vfit,
        "E": E,
    }


def initialize_from_cholesky(chol: np.ndarray, nt: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Initialize U,Z from Cholesky-like symmetric factors L_t.

    Each selected factor contributes vec(L_t) vec(L_t)^T.
    Factors are selected by descending Frobenius norm if nt is smaller.
    """
    chol = np.asarray(chol, dtype=np.float64)
    if chol.ndim != 3 or chol.shape[1] != chol.shape[2]:
        raise ValueError("chol must have shape (nchol,M,M)")
    chol = 0.5 * (chol + chol.transpose(0, 2, 1))
    nchol, m, _ = chol.shape
    if nt is None:
        nt = nchol
    if nt <= 0 or nt > nchol:
        raise ValueError(f"nt must satisfy 1 <= nt <= nchol={nchol}")
    order = np.argsort(np.linalg.norm(chol.reshape(nchol, -1), axis=1))[::-1][:nt]
    U = np.empty((nt, m, m), dtype=np.float64)
    Z = np.empty((nt, m, m), dtype=np.float64)
    for i, idx in enumerate(order):
        lam, vec = np.linalg.eigh(chol[idx])
        U[i] = vec
        Z[i] = np.outer(lam, lam)
    return U, Z


def initialize_from_eri(V: np.ndarray, m: int, nt: int) -> tuple[np.ndarray, np.ndarray]:
    """Dense X-DF-style initialization from pair-space eigendecomposition.

    Intended for small tests. Negative pair eigenvalues are supported by carrying
    their sign into Z.
    """
    if nt <= 0 or nt > m * m:
        raise ValueError("invalid nt")
    evals, evecs = np.linalg.eigh(0.5 * (V + V.T))
    order = np.argsort(np.abs(evals))[::-1][:nt]
    U = np.empty((nt, m, m), dtype=np.float64)
    Z = np.empty((nt, m, m), dtype=np.float64)
    for i, j in enumerate(order):
        g = float(evals[j])
        M = evecs[:, j].reshape(m, m)
        M = 0.5 * (M + M.T)
        # If the input has exact 8-fold symmetry, the pair eigenvectors can be
        # chosen symmetric. Renormalize after symmetrization for robustness.
        nrm = np.linalg.norm(M)
        if nrm < 1e-14:
            U[i] = np.eye(m)
            Z[i] = 0.0
            continue
        M /= nrm
        L = np.sqrt(abs(g)) * M
        lam, vec = np.linalg.eigh(L)
        U[i] = vec
        Z[i] = np.sign(g) * np.outer(lam, lam)
    return U, Z


def solve_Z_cg(V: np.ndarray, U: np.ndarray, rho_z: float,
               Z0: Optional[np.ndarray] = None,
               tol: float = 1e-10, maxiter: int = 5000,
               verbose: bool = False) -> np.ndarray:
    """Solve the regularized least-squares normal equations for full Z.

    The linear operator is
        H_t[Z] = sum_s K_ts Z_s K_ts^T + rho_z Z_t,
    K_ts = (U_t^T U_s) ** 2 elementwise.
    """
    nt, m, _ = U.shape
    K = [[None for _ in range(nt)] for _ in range(nt)]
    for t in range(nt):
        for s in range(nt):
            S = U[t].T @ U[s]
            K[t][s] = S * S

    B = np.empty((nt, m, m), dtype=np.float64)
    for t in range(nt):
        A = build_A(U[t])
        B[t] = A.T @ V @ A
        B[t] = 0.5 * (B[t] + B[t].T)

    nvar = nt * m * m

    def matvec(x: np.ndarray) -> np.ndarray:
        X = x.reshape(nt, m, m)
        Y = np.zeros_like(X)
        for t in range(nt):
            acc = rho_z * X[t]
            for s in range(nt):
                acc = acc + K[t][s] @ X[s] @ K[t][s].T
            Y[t] = 0.5 * (acc + acc.T)
        return Y.ravel()

    op = LinearOperator((nvar, nvar), matvec=matvec, dtype=np.float64)
    b = B.ravel()
    x0 = None if Z0 is None else np.asarray(Z0, dtype=np.float64).ravel()
    try:
        x, info = cg(op, b, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter)
    except TypeError:
        # Compatibility with older scipy.
        x, info = cg(op, b, x0=x0, tol=tol, maxiter=maxiter)
    if info != 0 and verbose:
        print(f"warning: CG info={info}")
    Z = x.reshape(nt, m, m)
    Z = 0.5 * (Z + Z.transpose(0, 2, 1))
    return Z


def eri_gradient_U(V: np.ndarray, U: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Euclidean gradient of normalized ERI loss with respect to each U[t]."""
    nt, m, _ = U.shape
    vnorm2 = float(np.vdot(V, V).real)
    Vfit = reconstruct_pair(U, Z)
    E = Vfit - V
    loss = 0.5 * float(np.vdot(E, E).real) / vnorm2
    G = np.zeros_like(U)
    for t in range(nt):
        A = build_A(U[t])
        grad_A = 2.0 * (E @ A @ Z[t]) / vnorm2
        for k in range(m):
            M = grad_A[:, k].reshape(m, m)
            G[t, :, k] = (M + M.T) @ U[t, :, k]
    return G, loss, E


def comm_gradient_U(U: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float]:
    """Euclidean gradient of normalized weighted commutator proxy for frozen q."""
    nt, m, _ = U.shape
    G = np.zeros_like(U)
    _, _, den = commutator_proxy(U, q)
    if den <= 0.0:
        return G, 0.0
    value, _, _ = commutator_proxy(U, q)
    for t in range(nt):
        for s in range(t + 1, nt):
            S = U[t].T @ U[s]
            W = np.outer(q[t], q[s]) / den
            D = W * (4.0 * S * (1.0 - 2.0 * S * S))
            G[t] += U[s] @ D.T
            G[s] += U[t] @ D
    return G, value


def _skew(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A - A.T)


def optimize_U_manifold(V: np.ndarray, U: np.ndarray, Z: np.ndarray,
                        q: np.ndarray, comm_lambda: float,
                        maxiter: int = 100, grad_tol: float = 1e-8,
                        step0: float = 1.0, armijo: float = 1e-4,
                        min_step: float = 1e-10,
                        verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Optimize orthogonal U at fixed Z and frozen q by manifold gradient descent."""
    U = np.array(U, copy=True)

    def frozen_objective(Ux: np.ndarray) -> tuple[float, float, float]:
        parts = objective_parts(V, Ux, Z, comm_lambda=0.0, rho_z=0.0,
                                q_for_comm=q)
        eri = parts["eri_objective"]
        comm = commutator_proxy(Ux, q)[0]
        return eri + comm_lambda * comm, eri, comm

    last = None
    for it in range(maxiter):
        Geri, eri_loss, _ = eri_gradient_U(V, U, Z)
        Gcomm, comm_loss = comm_gradient_U(U, q)
        G = Geri + comm_lambda * Gcomm

        H = np.empty_like(U)
        grad2 = 0.0
        for t in range(U.shape[0]):
            H[t] = _skew(U[t].T @ G[t])
            grad2 += float(np.vdot(H[t], H[t]).real)
        gradnorm = np.sqrt(grad2)
        f0 = eri_loss + comm_lambda * comm_loss
        last = {"objective": f0, "eri": eri_loss, "comm": comm_loss,
                "gradnorm": gradnorm, "nit": it}
        if verbose:
            print(f"    U iter {it:3d}: f={f0:.6e} eri={eri_loss:.6e} "
                  f"comm={comm_loss:.6e} |grad|={gradnorm:.3e}")
        if gradnorm < grad_tol:
            break

        step = step0
        accepted = False
        while step >= min_step:
            Utrial = np.empty_like(U)
            for t in range(U.shape[0]):
                Utrial[t] = U[t] @ expm(-step * H[t])
            ftrial, _, _ = frozen_objective(Utrial)
            if ftrial <= f0 - armijo * step * grad2:
                U = Utrial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            if verbose:
                print("    line search failed; stopping U step")
            break
    if last is None:
        f0, eri_loss, comm_loss = frozen_objective(U)
        last = {"objective": f0, "eri": eri_loss, "comm": comm_loss,
                "gradnorm": np.nan, "nit": 0}
    return U, last


def optimize_df_commutator(eri: np.ndarray, nt: int, *,
                           chol: Optional[np.ndarray] = None,
                           comm_lambda: float = 0.0,
                           rho_z: float = 1e-10,
                           outer_iters: int = 12,
                           u_maxiter: int = 100,
                           u_grad_tol: float = 1e-8,
                           u_step0: float = 1.0,
                           cg_tol: float = 1e-10,
                           cg_maxiter: int = 5000,
                           outer_tol: float = 1e-8,
                           verbose: bool = True) -> DFResult:
    """Alternating commutator-regularized compressed DF optimization."""
    eri = _check_eri(eri)
    m = eri.shape[0]
    V = eri.reshape(m * m, m * m)
    vnorm = np.linalg.norm(V)
    if vnorm == 0.0:
        raise ValueError("zero ERI tensor")

    if chol is not None:
        U, Z = initialize_from_cholesky(chol, nt=nt)
    else:
        U, Z = initialize_from_eri(V, m, nt)

    # Release Z from rank-one immediately.
    Z = solve_Z_cg(V, U, rho_z, Z0=Z, tol=cg_tol,
                   maxiter=cg_maxiter, verbose=verbose)

    prev_total = None
    nouter_done = 0
    for outer in range(outer_iters):
        nouter_done = outer + 1
        q = activities(Z)
        before = objective_parts(V, U, Z, comm_lambda, rho_z, q_for_comm=q)
        if verbose:
            print(
                f"outer {outer + 1}/{outer_iters} BEFORE U: "
                f"total={before['total_objective']:.6e} "
                f"eri={before['eri_objective']:.6e} "
                f"comm={before['comm_objective']:.6e} "
                f"zreg={before['z_regularizer']:.6e}"
            )

        U, _ = optimize_U_manifold(
            V, U, Z, q, comm_lambda,
            maxiter=u_maxiter, grad_tol=u_grad_tol,
            step0=u_step0, verbose=verbose,
        )

        Z = solve_Z_cg(V, U, rho_z, Z0=Z, tol=cg_tol,
                       maxiter=cg_maxiter, verbose=verbose)
        q_new = activities(Z)
        after = objective_parts(V, U, Z, comm_lambda, rho_z,
                                q_for_comm=q_new)
        if verbose:
            print(
                f"outer {outer + 1}/{outer_iters} AFTER Z: "
                f"total={after['total_objective']:.6e} "
                f"eri={after['eri_objective']:.6e} "
                f"comm={after['comm_objective']:.6e} "
                f"zreg={after['z_regularizer']:.6e}"
            )

        total = after["total_objective"]
        if prev_total is not None:
            relchg = abs(total - prev_total) / max(abs(prev_total), 1e-16)
            if verbose:
                print(f"  outer relative objective change = {relchg:.3e}")
            if relchg < outer_tol:
                break
        prev_total = total

    final = objective_parts(V, U, Z, comm_lambda, rho_z)
    relerr = float(np.linalg.norm(final["Vfit"] - V) / np.linalg.norm(V))
    return DFResult(
        U=U,
        Z=Z,
        eri_relative_error=relerr,
        eri_objective=float(final["eri_objective"]),
        comm_objective=float(final["comm_objective"]),
        z_regularizer=float(final["z_regularizer"]),
        total_objective=float(final["total_objective"]),
        outer_iterations=nouter_done,
    )


def load_array(path: str, key: Optional[str] = None) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix == ".npz":
        data = np.load(p)
        if key is None:
            if len(data.files) != 1:
                raise ValueError(f"{p} has keys {data.files}; specify --eri-key/--chol-key")
            key = data.files[0]
        return data[key]
    if p.suffix in (".pkl", ".pickle"):
        with open(p, "rb") as fh:
            data = pickle.load(fh)
        if key is None:
            if isinstance(data, np.ndarray):
                return data
            raise ValueError("pickle contains a mapping; specify a key")
        return np.asarray(data[key])
    raise ValueError("supported inputs: .npy, .npz, .pkl/.pickle")


def save_result(path: str, result: DFResult, comm_lambda: float, rho_z: float) -> None:
    np.savez(
        path,
        U=result.U,
        Z=result.Z,
        eri_relative_error=np.array(result.eri_relative_error),
        eri_objective=np.array(result.eri_objective),
        comm_objective=np.array(result.comm_objective),
        z_regularizer=np.array(result.z_regularizer),
        total_objective=np.array(result.total_objective),
        outer_iterations=np.array(result.outer_iterations),
        comm_lambda=np.array(comm_lambda),
        rho_z=np.array(rho_z),
        eri_order=np.array("prqs"),
    )


def save_grouped_pickle(path: str, U: np.ndarray, Z: np.ndarray) -> None:
    """Save leaves in the format consumed by QCSOR.decompose_h2."""
    grouped = [
        {"X": np.array(U[t], copy=True),
         "W": np.array(Z[t], copy=True),
         "isometry": True}
        for t in range(U.shape[0])
    ]
    with open(path, "wb") as fh:
        pickle.dump({"grouped": grouped}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _random_orthogonal(rng: np.random.Generator, m: int) -> np.ndarray:
    Q, R = np.linalg.qr(rng.standard_normal((m, m)))
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]


def _gradient_self_test() -> float:
    rng = np.random.default_rng(123)
    m, nt = 3, 2
    U = np.stack([_random_orthogonal(rng, m) for _ in range(nt)])
    Z = rng.standard_normal((nt, m, m))
    Z = 0.5 * (Z + Z.transpose(0, 2, 1))
    V = reconstruct_pair(U, Z) + 0.03 * rng.standard_normal((m*m, m*m))
    V = 0.5 * (V + V.T)
    q = activities(Z)
    lam = 0.17

    Geri, _, _ = eri_gradient_U(V, U, Z)
    Gcomm, _ = comm_gradient_U(U, q)
    G = Geri + lam * Gcomm

    # Test a tangent direction U_t exp(eps K_t).
    K = np.stack([_skew(rng.standard_normal((m, m))) for _ in range(nt)])
    analytic = 0.0
    for t in range(nt):
        analytic += float(np.sum(G[t] * (U[t] @ K[t])))

    def f(eps: float) -> float:
        Ut = np.empty_like(U)
        for t in range(nt):
            Ut[t] = U[t] @ expm(eps * K[t])
        parts = objective_parts(V, Ut, Z, 0.0, 0.0, q_for_comm=q)
        return parts["eri_objective"] + lam * commutator_proxy(Ut, q)[0]

    eps = 1e-6
    numeric = (f(eps) - f(-eps)) / (2.0 * eps)
    err = abs(analytic - numeric)
    print(f"gradient self-test: analytic={analytic:.12e} numeric={numeric:.12e} abs_err={err:.3e}")
    return err


def _zsolve_self_test() -> float:
    rng = np.random.default_rng(7)
    m, nt = 3, 2
    U = np.stack([_random_orthogonal(rng, m) for _ in range(nt)])
    Ztrue = rng.standard_normal((nt, m, m))
    Ztrue = 0.5 * (Ztrue + Ztrue.transpose(0, 2, 1))
    V = reconstruct_pair(U, Ztrue)
    Zfit = solve_Z_cg(V, U, rho_z=1e-12, tol=1e-12, maxiter=5000)
    rel = np.linalg.norm(reconstruct_pair(U, Zfit) - V) / np.linalg.norm(V)
    print(f"Z-solve self-test relative reconstruction error={rel:.3e}")
    return rel


def _optimization_self_test() -> float:
    rng = np.random.default_rng(19)
    m, nt = 3, 2
    U0 = np.stack([_random_orthogonal(rng, m) for _ in range(nt)])
    Z0 = rng.standard_normal((nt, m, m))
    Z0 = 0.5 * (Z0 + Z0.transpose(0, 2, 1))
    eri = reconstruct_pair(U0, Z0).reshape(m, m, m, m)
    res = optimize_df_commutator(
        eri, nt,
        comm_lambda=1e-4,
        rho_z=1e-10,
        outer_iters=3,
        u_maxiter=15,
        verbose=False,
    )
    print(f"optimization self-test: relerr={res.eri_relative_error:.3e} comm={res.comm_objective:.3e}")
    return res.eri_relative_error


def self_test() -> int:
    e1 = _gradient_self_test()
    e2 = _zsolve_self_test()
    e3 = _optimization_self_test()
    ok = (e1 < 1e-6) and (e2 < 1e-7) and np.isfinite(e3)
    print("SELF TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Commutator-regularized compressed double factorization with full symmetric Z."
    )
    parser.add_argument("input", nargs="?", help="ERI input (.npy/.npz/.pkl), ordering eri[p,r,q,s]")
    parser.add_argument("--eri-key", default="eri", help="ERI key for npz/pickle")
    parser.add_argument("--nt", type=int, help="number of DF leaves")
    parser.add_argument("--chol", default=None, help="optional Cholesky factor file for initialization")
    parser.add_argument("--chol-key", default="chol", help="Cholesky key for npz/pickle")
    parser.add_argument("--comm-lambda", type=float, default=1e-4,
                        help="weight of normalized activity-weighted commutator proxy")
    parser.add_argument("--rho-z", type=float, default=1e-10,
                        help="L2 magnitude regularization of full Z; does not impose low rank")
    parser.add_argument("--outer-iters", type=int, default=12)
    parser.add_argument("--u-maxiter", type=int, default=100)
    parser.add_argument("--u-grad-tol", type=float, default=1e-8)
    parser.add_argument("--u-step0", type=float, default=1.0)
    parser.add_argument("--cg-tol", type=float, default=1e-10)
    parser.add_argument("--cg-maxiter", type=int, default=5000)
    parser.add_argument("--outer-tol", type=float, default=1e-8)
    parser.add_argument("--output", default="df_comm_opt.npz")
    parser.add_argument("--grouped-pkl", default=None,
                        help="optional QCSOR-compatible grouped pickle output")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.input is None or args.nt is None:
        parser.error("input and --nt are required unless --self-test is used")

    eri = load_array(args.input, args.eri_key)
    chol = None
    if args.chol is not None:
        chol = load_array(args.chol, args.chol_key)

    result = optimize_df_commutator(
        eri,
        args.nt,
        chol=chol,
        comm_lambda=args.comm_lambda,
        rho_z=args.rho_z,
        outer_iters=args.outer_iters,
        u_maxiter=args.u_maxiter,
        u_grad_tol=args.u_grad_tol,
        u_step0=args.u_step0,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        outer_tol=args.outer_tol,
        verbose=not args.quiet,
    )

    save_result(args.output, result, args.comm_lambda, args.rho_z)
    if args.grouped_pkl is not None:
        save_grouped_pickle(args.grouped_pkl, result.U, result.Z)

    print("\nFinal result")
    print(f"  U shape                = {result.U.shape}")
    print(f"  Z shape                = {result.Z.shape}")
    print(f"  relative ERI error     = {result.eri_relative_error:.8e}")
    print(f"  ERI objective          = {result.eri_objective:.8e}")
    print(f"  commutator proxy       = {result.comm_objective:.8e}")
    print(f"  Z regularizer          = {result.z_regularizer:.8e}")
    print(f"  total objective        = {result.total_objective:.8e}")
    print(f"  outer iterations       = {result.outer_iterations}")
    print(f"  max orthogonality err  = {max(np.linalg.norm(u.T @ u - np.eye(u.shape[0])) for u in result.U):.3e}")
    print(f"  ||Z||_F                = {np.linalg.norm(result.Z):.8e}")
    print(f"  max |Z_tkl|            = {np.max(np.abs(result.Z)):.8e}")
    print(f"saved {args.output}")
    if args.grouped_pkl is not None:
        print(f"saved {args.grouped_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
