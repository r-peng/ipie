"""Squared commutator losses in mode space, and their pullback.

Every spin block of every channel is diagonal in one basis: a rotated Cholesky
factor (bases 0..r-1) or the h eigenbasis (basis r). Write the block of
channel i in spin sigma as A_i = sum_a kappa_{i,a} P_a with P_a = x_a x_a^T,
and let s_i be the pair scale (1 for loss_kind "raw", the stored Green weight
a_i for "weighted"). Two objectives are supported:

  raw       L_raw = sum_{i<j} ||[u_i,u_j]||_F^2
  weighted  L_w   = sum_{i<j} a_i a_j ||[u_i,u_j]||_F^2   (unnormalized a_i)

with u_i = diag(I+A_i_up, I+A_i_down) and the two spin blocks' squared norms
added. Over the global modes a = basis*n + mode define, per spin,

    q[a]     = sum_i s_i kappa_{i,a}^2                     (diagonal loads)
    O[a,a']  = sum_i s_i kappa_{i,a} kappa_{i,a'}, a!=a'   (same-spin rank-2 blocks;
                                                            -W_{aa'} for canonical channels)
    C        = X_all^T X_all, zero within each basis;  C2 = C*C;  F = C2 - C2*C2
    h[a]     = (C2 O C2^T)[a,a]
    OO       = sum O[a,a'] O[m,m'] C_am C_a'm C_a'm' C_am'.

Then, exactly, L_sigma = q^T F q - 2 q.h - OO, and dL/dq = 2 (F q - h) >= 0 is
the commutator field seen by a mode. Pairs within one basis commute and
contribute nothing, so each basis enters L affinely through its own loads.
The cost is a few dense operations on the (r+1)n x (r+1)n overlap matrix,
independent of the number of channel pairs.

Loads are built from the channel tables, so any channel list is accepted
(including channels.with_channels results). Two legs on the same spin and the
same mode merge into one projector, (1 + d0 P)(1 + d1 P) = 1 + (d0+d1+d0 d1) P.

Entry points: objective_loss / raw_loss (the optimized value), commutator_loss
(diagnostic record with the sor_base-compatible fields), loss_pullback (the
loss with its channel-leg and basis adjoints, used by gradients.py), and the
ModeSpace / channel_loads / offdiagonal_terms building blocks used by
shift.py and partitions.py.
"""

import math

import numpy as np

from .models import LOSS_KINDS, CommutatorLoss, FixedDecomposition
from .problem import _real_array


def loss_kind_name(loss_kind):
    if not isinstance(loss_kind, str) or loss_kind not in LOSS_KINDS:
        raise ValueError(f"loss_kind must be one of {LOSS_KINDS}; got {loss_kind!r}")
    return loss_kind


def _weights(decomposition, normalize_weights):
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    if not isinstance(normalize_weights, (bool, np.bool_)):
        raise TypeError("normalize_weights must be boolean")
    weights = decomposition.coefficients
    if weights.shape != (decomposition.nterms,):
        raise ValueError("Coefficient count does not match channel count")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("All channel coefficients must be finite and strictly positive")
    if normalize_weights and weights.size:
        total = decomposition.denom
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Combined coefficient sum must be finite and positive")
        weights = weights / total
    return weights


def pair_scales(decomposition, loss_kind="weighted"):
    """Per-channel s_i with pair weight s_i*s_j (combined one-body-first order).

    raw: s_i=1. weighted: s_i=a_i, the stored UNNORMALIZED Green weights.
    """
    loss_kind = loss_kind_name(loss_kind)
    weights = _weights(decomposition, False)
    if loss_kind == "raw":
        return np.ones_like(weights)
    return weights


# --------------------------------------------------------------------------- mode space

class ModeSpace:
    """Gauge-dependent dense data for one decomposition's bases.

    Modes are ordered basis by basis: Cholesky factor g owns modes
    [g*n, (g+1)*n), the h eigenbasis owns [r*n, (r+1)*n) (hslice). C holds the
    overlaps between modes of DIFFERENT bases and is zero within a basis.

    one_body_X (2, n, n) gives each spin its own h eigenbasis (a split Cholesky
    shift). C, C2, F are then those of spin up; spin down differs only in the
    h rows and columns, kept as the strips C_down, C2_down, F_down (n, r*n) of
    h-basis rows against factor columns. split is True in that case, even if
    the two bases happen to be equal.
    """

    def __init__(self, X, one_body_X):
        X = np.asarray(X, dtype=np.float64)
        self.nchol, self.n, _ = X.shape
        n, r = self.n, self.nchol
        self.X = X
        self.nmodes = (r+1)*n
        self.hslice = slice(r*n, (r+1)*n)
        one_body_X = np.asarray(one_body_X, dtype=np.float64)
        self.split = one_body_X.ndim == 3
        up = one_body_X[0] if self.split else one_body_X
        Xf = X.transpose(1, 0, 2).reshape(n, r*n)
        self.Xall = np.concatenate([Xf, up], axis=1)
        basis = np.repeat(np.arange(r+1), n)
        self.cross = basis[:, None] != basis[None, :]
        self.C = np.where(self.cross, self.Xall.T @ self.Xall, 0.0)
        self.C2 = self.C*self.C
        self.F = self.C2 - self.C2*self.C2
        if self.split:
            self.X_down = one_body_X[1]
            self.C_down = self.X_down.T @ Xf
            self.C2_down = self.C_down*self.C_down
            self.F_down = self.C2_down - self.C2_down*self.C2_down

    @classmethod
    def of(cls, decomposition):
        return cls(decomposition.two_body.gauge.X, decomposition.one_body.X)

    def factor(self, g):
        return slice(g*self.n, (g+1)*self.n)

    def _down_product(self, q, cols):
        """(q F_down)[cols] for one spin-down load vector q (nmodes,), from the blocks.

        F_down shares the factor-factor block with F; its h rows are F_down,
        its h columns F_down.T, and its h-h block is zero.
        """
        rn = self.nchol*self.n
        start, stop, _ = cols.indices(self.nmodes)
        out = np.empty(max(stop - start, 0))
        f_stop = min(stop, rn)
        if f_stop > start:
            out[:f_stop-start] = q[:rn] @ self.F[:rn, start:f_stop] + q[rn:] @ self.F_down[:, start:f_stop]
        h_start = max(start, rn)
        if stop > h_start:
            out[h_start-start:] = self.F_down[h_start-rn:stop-rn] @ q[:rn]
        return out

    def field(self, q, h, cols=slice(None)):
        """dL/dq on the selected modes: 2 (F q - h)."""
        if not self.split:
            return 2.0*(q @ self.F[:, cols] - h[:, cols])
        return 2.0*(np.stack([q[0] @ self.F[:, cols], self._down_product(q[1], cols)]) - h[:, cols])

    def loss(self, q, h, OO):
        """q^T F q - 2 q.h - OO summed over spins; roundoff below zero is clamped to 0."""
        if not self.split:
            quadratic = np.sum(q*(q @ self.F))
        else:
            quadratic = np.sum(q[0]*(q[0] @ self.F)) + np.sum(q[1]*self._down_product(q[1], slice(None)))
        value = float(quadratic - 2.0*np.sum(q*h) - np.sum(OO))
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite commutator loss")
        return max(0.0, value)

    def orbital_K(self, Os):
        """K[p,q,r,s] = sum_g sum_{aa'} O_g[a,a'] x_a[p] x_a[q] x_a'[r] x_a'[s]."""
        X = self.X
        return np.einsum("gpa,gqa,gab,grb,gsb->pqrs", X, X, Os, X, X, optimize=True)

    def offdiagonal_terms(self, O):
        """h (2, nmodes) and OO (2,) of the off-diagonal loads (zeros if O is None)."""
        if O is None:
            return np.zeros((2, self.nmodes)), np.zeros(2)
        r, n = self.nchol, self.n
        Y = self.C2[:, :r*n].reshape(self.nmodes, r, n)
        h = np.einsum("arm,srmk,ark->sa", Y, O, Y, optimize=True)
        if self.split:
            # Spin down sees its own h basis in the h rows; O only lives on factor modes.
            Yd = self.C2_down.reshape(n, r, n)
            h[1, self.hslice] = np.einsum("arm,rmk,ark->a", Yd, O[1], Yd, optimize=True)
        OO = np.zeros(2)
        for spin in (0, 1):
            if np.any(O[spin]):
                K = self.orbital_K(O[spin])
                OO[spin] = np.einsum("pqrs,qrsp->", K, K, optimize=True)
        return h, OO


def channel_legs(table):
    """Leg classes of a ChannelTable: (cross_spin, same_spin_distinct, same_spin_repeated) masks."""
    same = table.spins[:, 0] == table.spins[:, 1]
    repeated = same & (table.modes[:, 0] == table.modes[:, 1])
    return ~same, same & ~repeated, repeated


def channel_loads(decomposition, loss_kind, space=None):
    """(q, O) of every channel of the decomposition (any channel list).

    q (2, nmodes) diagonal loads; O (2, r, n, n) symmetric same-spin
    off-diagonal loads with zero diagonal, or None if there are none.
    """
    loss_kind = loss_kind_name(loss_kind)
    one, two = decomposition.one_body.table, decomposition.two_body.table
    r, n, _ = decomposition.two_body.gauge.X.shape
    N = (r+1)*n
    weighted = loss_kind == "weighted"
    q = np.zeros(2*N)
    if len(one):
        s1 = one.a if weighted else np.ones(len(one))
        q += np.bincount(one.spin*N + r*n + one.mode, weights=s1*one.d*one.d, minlength=2*N)
    O = None
    if len(two):
        s = two.a if weighted else np.ones(len(two))
        d0, d1 = two.d[:, 0], two.d[:, 1]
        gm0, gm1 = two.bix*n + two.modes[:, 0], two.bix*n + two.modes[:, 1]
        split, distinct, repeated = channel_legs(two)
        legs = split | distinct
        q += np.bincount(two.spins[legs, 0]*N + gm0[legs], weights=(s*d0*d0)[legs], minlength=2*N)
        q += np.bincount(two.spins[legs, 1]*N + gm1[legs], weights=(s*d1*d1)[legs], minlength=2*N)
        if np.any(repeated):
            kappa = d0[repeated] + d1[repeated] + d0[repeated]*d1[repeated]
            q += np.bincount(two.spins[repeated, 0]*N + gm0[repeated], weights=s[repeated]*kappa*kappa,
                             minlength=2*N)
        if np.any(distinct):
            flat = ((two.spins[distinct, 0]*r + two.bix[distinct])*n + two.modes[distinct, 0])*n \
                + two.modes[distinct, 1]
            O = np.bincount(flat, weights=(s*d0*d1)[distinct], minlength=2*r*n*n).reshape(2, r, n, n)
            O = O + O.transpose(0, 1, 3, 2)
    return q.reshape(2, N), O


def check_bases(decomposition):
    """Orthonormal bases of the right shape: what the mode-space formulas assume."""
    X = _real_array(decomposition.two_body.gauge.X, "Cholesky bases")
    if X.ndim != 3 or X.shape[0] < 1 or X.shape[1] != X.shape[2] or X.shape[1] < 1:
        raise ValueError("Cholesky bases must have shape (r,n,n)")
    n = X.shape[1]
    one_body_X = _real_array(decomposition.one_body.X, "one-body basis")
    if one_body_X.ndim == 3:
        if one_body_X.shape[0] != 2:
            raise ValueError("Per-spin one-body bases must have shape (2,n,n)")
        bases = [(f"one-body basis (spin {s})", one_body_X[s]) for s in (0, 1)]
    else:
        bases = [("one-body basis", one_body_X)]
    bases += [(f"Cholesky basis {g}", factor) for g, factor in enumerate(X)]
    for name, basis in bases:
        if basis.shape != (n, n):
            raise ValueError(f"{name} must have shape {(n,n)}")
        if not np.allclose(basis.T @ basis, np.eye(n), atol=1e-12, rtol=1e-10):
            raise ValueError(f"{name} must have orthonormal columns for the overlap formula")
    one, two = decomposition.one_body.table, decomposition.two_body.table
    if not (np.isfinite(one.d).all() and np.isfinite(two.d).all()):
        raise ValueError("Channel amplitudes must be finite and real")


def _loss(decomposition, loss_kind, space):
    q, O = channel_loads(decomposition, loss_kind, space)
    h, OO = space.offdiagonal_terms(O)
    return space.loss(q, h, OO)


# --------------------------------------------------------------------------- public losses

def objective_loss(decomposition, *, loss_kind="weighted"):
    """The optimized objective for loss_kind, and nothing else.

    raw: L_raw. weighted: sum_{i<j} a_i a_j ||[u_i,u_j]||_F^2 with the stored
    unnormalized a_i.
    """
    if not isinstance(decomposition, FixedDecomposition):
        raise TypeError("decomposition must be a FixedDecomposition")
    loss_kind = loss_kind_name(loss_kind)
    check_bases(decomposition)
    return _loss(decomposition, loss_kind, ModeSpace.of(decomposition))


def raw_loss(decomposition):
    """Only L_raw: objective_loss with loss_kind='raw'."""
    return objective_loss(decomposition, loss_kind="raw")


def commutator_loss(decomposition, *, normalize_weights=True):
    """Every loss at once: L_raw, unnormalized L_w and sor_base-compatible diagnostics.

    normalize_weights affects only weighted_comm2, weight and weighted_mean;
    comm2 and weighted_comm2_unnormalized do not depend on it. weight is the
    sum of w_i w_j over ALL unordered pairs, ((sum w)^2 - sum w^2)/2.
    """
    weights = _weights(decomposition, normalize_weights)
    check_bases(decomposition)
    space = ModeSpace.of(decomposition)
    raw = _loss(decomposition, "raw", space)
    unnormalized = _loss(decomposition, "weighted", space)
    weighted = unnormalized/decomposition.denom**2 if normalize_weights else unnormalized
    first, second = math.fsum(weights), math.fsum(weights*weights)
    weight = max(0.0, 0.5*(first*first - second))
    count = weights.size
    if not all(np.isfinite(value) for value in (raw, weighted, weight, unnormalized)):
        raise FloatingPointError("Overflow in total commutator diagnostics")
    return CommutatorLoss(raw, weighted, weight, weighted/weight if weight > 0 else None,
                          count*(count-1)//2, bool(normalize_weights), unnormalized)


# --------------------------------------------------------------------------- pullback

def loss_pullback(decomposition, loss_kind="weighted", check=True, space=None, *, with_one_body=False):
    """(loss, dL/d channel-leg amplitudes (T,2), dL/dX (r,n,n)).

    gX[g][:, alpha] is the unconstrained partial derivative with respect to
    the factor eigenvector X[g][:, alpha] (same-basis pairs excluded). check=True
    runs check_bases first. space: a ModeSpace already built for these bases
    (GaugeEvaluator reuses one per gauge); None builds it.

    with_one_body=True also returns (g_one, gY): g_one (K,) = dL/d of each
    one-body channel amplitude, and gY = dL/dY for the one-body eigenvectors Y
    (columns), (n, n) for one shared basis or (2, n, n) for per-spin bases (a
    split ModeSpace). They feed the h_sigma pullback of a Cholesky shift.
    """
    loss_kind = loss_kind_name(loss_kind)
    if check:
        check_bases(decomposition)
    space = ModeSpace.of(decomposition) if space is None else space
    q, O = channel_loads(decomposition, loss_kind, space)
    h, OO = space.offdiagonal_terms(O)
    loss = space.loss(q, h, OO)
    G = space.field(q, h)                                            # dL/dq, (2, N)

    two = decomposition.two_body.table
    r, n, N = space.nchol, space.n, space.nmodes
    s = two.a if loss_kind == "weighted" else np.ones(len(two))
    d0, d1 = two.d[:, 0], two.d[:, 1]
    gm0, gm1 = two.bix*n + two.modes[:, 0], two.bix*n + two.modes[:, 1]
    split, distinct, repeated = channel_legs(two)
    g_amplitude = np.zeros((len(two), 2))
    legs = split | distinct
    g_amplitude[legs, 0] = 2.0*(s*d0)[legs]*G[two.spins[legs, 0], gm0[legs]]
    g_amplitude[legs, 1] = 2.0*(s*d1)[legs]*G[two.spins[legs, 1], gm1[legs]]
    if np.any(repeated):
        kappa = d0[repeated] + d1[repeated] + d0[repeated]*d1[repeated]
        common = 2.0*s[repeated]*kappa*G[two.spins[repeated, 0], gm0[repeated]]
        g_amplitude[repeated, 0] = common*(1.0 + d1[repeated])
        g_amplitude[repeated, 1] = common*(1.0 + d0[repeated])

    if space.split:
        gX, gY, dO = _split_basis_pullback(space, q, O)
    else:
        C, C2 = space.C, space.C2
        GC = (np.outer(q[0], q[0]) + np.outer(q[1], q[1]))*(2.0*C - 4.0*C*C2)   # from q^T F(C) q
        gX_offdiagonal = np.zeros((r, n, n))
        dO = None
        if O is not None:
            Yr = C2[:, :r*n].reshape(N, r, n)
            dO = np.zeros_like(O)                                    # dL/dO, entries independent
            for spin in (0, 1):
                Os = O[spin]
                if not np.any(Os):
                    continue
                # -2 q.h, h = diag(C2 O C2^T)
                dO[spin] -= 2.0*np.einsum("xga,xgb->gab", Yr*q[spin][:, None, None], Yr, optimize=True)
                C2O = np.einsum("xgm,gmk->xgk", Yr, Os).reshape(N, r*n)
                GC[:, :r*n] -= 8.0*q[spin][:, None]*C[:, :r*n]*C2O
                # -OO = -sum_{pqrs} K[p,q,r,s] K[q,r,s,p]
                gX_offdiagonal -= _offdiagonal_square_pullback(space, Os, dO, spin)
        GX = space.Xall @ (GC + GC.T)
        gX = GX[:, :r*n].reshape(n, r, n).transpose(1, 0, 2) + gX_offdiagonal
        gY = GX[:, r*n:]
    if O is not None:
        both = dO + dO.transpose(0, 1, 3, 2)
        coupling = both[two.spins[distinct, 0], two.bix[distinct], two.modes[distinct, 0], two.modes[distinct, 1]]
        g_amplitude[distinct, 0] += (s*d1)[distinct]*coupling
        g_amplitude[distinct, 1] += (s*d0)[distinct]*coupling
    if not (np.isfinite(g_amplitude).all() and np.isfinite(gX).all()):
        raise FloatingPointError("Nonfinite loss pullback")
    if not with_one_body:
        return float(loss), g_amplitude, gX
    one = decomposition.one_body.table
    s1 = one.a if loss_kind == "weighted" else np.ones(len(one))
    g_one = 2.0*s1*one.d*G[one.spin, r*n + one.mode]
    if not (np.isfinite(g_one).all() and np.isfinite(gY).all()):
        raise FloatingPointError("Nonfinite one-body pullback")
    return float(loss), g_amplitude, gX, g_one, gY


def _offdiagonal_square_pullback(space, Os, dO, spin):
    """The -OO term of one spin: adds its dL/dO to dO[spin], returns its (positive) factor-basis term.

    OO involves only factor-factor overlaps, so it does not depend on the h bases.
    """
    X = space.X
    K = space.orbital_K(Os)
    Kt = K.transpose(3, 0, 1, 2)                                     # Kt[p,q,r,s] = K[q,r,s,p]
    T = np.einsum("pqrs,gpa,gqa->gars", Kt, X, X, optimize=True)
    dO[spin] -= 2.0*np.einsum("gars,grb,gsb->gab", T, X, X, optimize=True)
    V = np.einsum("gab,grb,gsb->gars", Os, X, X, optimize=True)
    return 8.0*np.einsum("qrsu,gqa,gars->gua", K, X, V, optimize=True)


def _split_basis_pullback(space, q, O):
    """Basis adjoints (gX, gY (2, n, n), dO) for a split ModeSpace, in blocks.

    Both spins share the factor-factor overlaps; spin s has its own h rows
    (h basis Y_s against the factors). With GC_s = dL_s/dC_s, the adjoint of a
    basis vector x_i is sum_j (GC_s + GC_s^T)[i, j] x_j over that spin's
    bases, so the factor-factor blocks of both spins add up (one rn x rn
    array, as in the shared path) and each spin contributes its own h strip.
    """
    r, n = space.nchol, space.n
    rn = r*n
    Xf = space.Xall[:, :rn]
    Y = (space.Xall[:, rn:], space.X_down)
    C, C2 = space.C, space.C2
    Cff, C2ff = C[:rn, :rn], C2[:rn, :rn]
    strips = ((C[rn:, :rn], C2[rn:, :rn]), (space.C_down, space.C2_down))     # (n, rn): h rows, factor columns
    qf, qh = q[:, :rn], q[:, rn:]
    GC_ff = (np.outer(qf[0], qf[0]) + np.outer(qf[1], qf[1]))*(2.0*Cff - 4.0*Cff*C2ff)
    quad_hf = [np.outer(qh[s], qf[s])*(2.0*strips[s][0] - 4.0*strips[s][0]*strips[s][1]) for s in (0, 1)]
    GC_hf = [block.copy() for block in quad_hf]
    gX_offdiagonal = np.zeros((r, n, n))
    dO = None
    if O is not None:
        dO = np.zeros_like(O)
        Yff = C2ff.reshape(rn, r, n)
        for spin in (0, 1):
            Os = O[spin]
            if not np.any(Os):
                continue
            Yh = strips[spin][1].reshape(n, r, n)
            dO[spin] -= 2.0*(np.einsum("xga,xgb->gab", Yff*qf[spin][:, None, None], Yff, optimize=True)
                             + np.einsum("xga,xgb->gab", Yh*qh[spin][:, None, None], Yh, optimize=True))
            GC_ff -= 8.0*qf[spin][:, None]*Cff*np.einsum("xgm,gmk->xgk", Yff, Os).reshape(rn, rn)
            GC_hf[spin] -= 8.0*qh[spin][:, None]*strips[spin][0]*np.einsum("xgm,gmk->xgk", Yh, Os).reshape(n, rn)
            gX_offdiagonal -= _offdiagonal_square_pullback(space, Os, dO, spin)
    # (GC_s + GC_s^T) has [h, f] = GC_hf + quad_hf (GC_s[f, h] is the quadratic part only) and [h, h] = 0.
    sym_hf = [GC_hf[s] + quad_hf[s] for s in (0, 1)]
    GXf = Xf @ (GC_ff + GC_ff.T) + Y[0] @ sym_hf[0] + Y[1] @ sym_hf[1]
    gX = GXf.reshape(n, r, n).transpose(1, 0, 2) + gX_offdiagonal
    gY = np.stack([Xf @ sym_hf[s].T for s in (0, 1)])
    return gX, gY, dO
