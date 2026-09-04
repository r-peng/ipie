"""
GPU-friendly deterministic LAFQMC on a 3-angle Givens grid for H2+ (one
real electron in four spatial orbitals), using EXACT four-corner tetrahedral
deposition in the physical one-electron Hilbert space.

The local Givens chart is

    phi(t1,t2,t3) =
        [cos(t3) cos(t2) cos(t1),
         sin(t1),
         sin(t2) cos(t1),
         sin(t3) cos(t2) cos(t1)].

The midpoint theta grid is the same as in the previous trilinear code.  Given
an exact propagated determinant, we first find its enclosing 3D grid cube.
That cube is split into the standard six Freudenthal tetrahedra.  If the
fractional coordinates f=(f1,f2,f3) satisfy

    f[p0] >= f[p1] >= f[p2],

the containing tetrahedron has the four cube vertices

    000,
    e[p0],
    e[p0] + e[p1],
    111.

These four grid determinants are then used as a (generically nonorthogonal)
basis of the four-dimensional one-electron Hilbert space.  Their coefficients
are obtained analytically from Cramer's rule.  The four 4x4 determinant
numerators are evaluated by a specialized explicit det4 formula, while the
denominator uses one of six closed-form expressions, one for each
Freudenthal tetrahedron type.  No batched linalg.solve or linalg.det is used.
Therefore, for free projection with projective boundary mapping, reconstruction
of every propagated branch is exact up to floating-point roundoff.

Important distinction:
  * The Freudenthal/barycentric construction is used only to choose WHICH four
    local corners are used.
  * The deposition coefficients are physical Hilbert-space coefficients from
    V c = phi_target.  They are not the tetrahedron barycentric coordinates.

At a chart boundary, extended-grid (ghost) vertices are first used in the
analytic four-corner expansion.  Afterward each ghost determinant is mapped to its projectively
equivalent interior grid determinant and the projective sign is applied.

Constrained path is also available.  The exact branch factor

    gbar_ij = a_i <Psi_T|U_i|D_j> / <Psi_T|D_j>

is tested before tetrahedral deposition and the whole branch is removed when
gbar_ij <= cp_tol.  With cp_boundary='projective', every accepted branch is
still represented exactly in Hilbert space.  However, as discussed, finite-
grid CP dynamics can remain representation dependent because the next CP test
acts on the grid determinants separately.

This exact tetrahedral method is a wavefunction representation.  It does not
provide a nonnegative walker-histogram deposition mode, because the exact
four-corner coefficients can have either sign.
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

try:
    from ipie.utils.backend import arraylib as xp
    from ipie.utils.backend import to_host
except ImportError:  # CPU fallback for testing outside an ipie environment.
    xp = np

    def to_host(x):
        return np.asarray(x)


def h2plus_symmetry_basis(trial, thresh: float = 1.0e-12):
    """Return the trial/symmetry-adapted orthogonal basis used previously.

    The normalized trial must have the form [0,a,0,b].  Columns of B are

        q0 = [0,a,0,b]       trial direction,
        q1 = [0,-b,0,a]      same-symmetry tangent direction,
        q2 = [1,0,0,0]       first off-symmetry direction,
        q3 = [0,0,1,0]       second off-symmetry direction.
    """
    t = xp.asarray(trial).copy()
    nrm = xp.linalg.norm(t)
    if float(to_host(nrm)) < thresh:
        raise ValueError("Trial vector has near-zero norm.")
    t /= nrm

    if float(to_host(xp.abs(t[0]))) > thresh or float(to_host(xp.abs(t[2]))) > thresh:
        raise ValueError("Expected a trial orbital of the form [0,a,0,b].")

    a = t[1]
    b = t[3]
    B = xp.zeros((4, 4), dtype=t.dtype)
    B[0, 2] = 1.0
    B[1, 0] = a
    B[1, 1] = -b
    B[2, 3] = 1.0
    B[3, 0] = b
    B[3, 1] = a

    err = xp.linalg.norm(B.T @ B - xp.eye(4, dtype=t.dtype))
    if float(to_host(err)) > 1.0e-10:
        raise RuntimeError(
            f"Constructed basis is not orthonormal; error={float(to_host(err))}."
        )
    return B


class GivensMasterEquation:
    """Exact-tetrahedral H2+ Givens-grid transfer operator.

    Parameters
    ----------
    Ng
        Number of midpoint grid nodes per angle.  Odd Ng places the trial
        theta=(0,0,0) exactly on the grid.
    importance
        If True, store f_j=<Psi_T|D_j> c_j instead of bare coefficients c_j.
    nodal_tol
        Tolerance used to choose a deterministic projective representative.
    constraint_path
        If True, reject exact branches with gbar_ij <= cp_tol.
    cp_tol
        Threshold for the branchwise CP test.  Use 0.0 for the literal sign
        test.
    cp_boundary
        'projective' is recommended for direct branchwise CP and is required
        for exact Hilbert-space deposition of accepted branches.  'absorbing'
        is retained only for comparisons and drops ghost tetra vertices.
    check_tetra_solve
        If True, track the maximum local four-corner reconstruction residual
        and the maximum absolute tetra coefficient during each mat_vec.  Useful
        for validation; it adds some overhead.
    """

    def __init__(
        self,
        Ng: int = 33,
        importance: bool = True,
        nodal_tol: float = 1.0e-13,
        constraint_path: bool = False,
        cp_tol: float = 0.0,
        cp_boundary: str = "projective",
        check_tetra_solve: bool = False,
    ):
        self.importance = bool(importance)
        self.Ng = int(Ng)
        self.nodal_tol = float(nodal_tol)
        self.constraint_path = bool(constraint_path)
        self.cp_tol = float(cp_tol)
        self.cp_boundary = str(cp_boundary).lower()
        self.check_tetra_solve = bool(check_tetra_solve)

        if self.constraint_path and not self.importance:
            raise ValueError("constraint_path=True currently requires importance=True.")
        if self.cp_boundary not in {"projective", "absorbing"}:
            raise ValueError("cp_boundary must be 'projective' or 'absorbing'.")
        if self.cp_tol < 0.0:
            raise ValueError("cp_tol must be nonnegative.")
        if self.Ng < 3:
            raise ValueError("Ng must be at least 3.")
        if self.Ng % 2 == 0:
            print("WARNING: Ng is even, so theta=0 is not a grid point.")

        self.dtheta = xp.pi / self.Ng
        self.theta0 = -0.5 * xp.pi + 0.5 * self.dtheta
        self.theta_vals = self.theta0 + xp.arange(self.Ng) * self.dtheta
        self.Nh = self.Ng**3

        print(
            f"Givens grid: Ng={self.Ng}, Nh={self.Nh}, "
            f"dtheta={float(to_host(self.dtheta))}"
        )

        p1, p2, p3 = xp.indices((self.Ng, self.Ng, self.Ng), dtype=xp.int64)
        theta = xp.stack(
            [self.theta_vals[p1], self.theta_vals[p2], self.theta_vals[p3]], axis=-1
        ).reshape(-1, 3)
        self.mos_local = self.givens_to_local_vec(theta)
        del p1, p2, p3, theta

        self.B = None
        self.mos = None
        self.trial = None
        self.trial_mos = None
        self.Us = None
        self.aUs = None
        self.gtrial = None
        self.ham = None
        self.last_cp_stats = None
        self.last_tetra_stats = None
        self.last_initial_info = None

        self._eye3_int = xp.eye(3, dtype=xp.int64)

        # Same exact one-ghost-layer projective boundary map as the trilinear code.
        self._build_extended_boundary_map()

    def flatten_idx(self, i1, i2, i3):
        return (i1 * self.Ng + i2) * self.Ng + i3

    def flat2idx(self, ix):
        i1 = ix // (self.Ng * self.Ng)
        rem = ix % (self.Ng * self.Ng)
        i2 = rem // self.Ng
        i3 = rem % self.Ng
        return i1, i2, i3

    @staticmethod
    def givens_to_local_vec(theta):
        """Convert (...,3) Givens angles to normalized local vectors (...,4)."""
        t1 = theta[..., 0]
        t2 = theta[..., 1]
        t3 = theta[..., 2]

        c1, s1 = xp.cos(t1), xp.sin(t1)
        c2, s2 = xp.cos(t2), xp.sin(t2)
        c3, s3 = xp.cos(t3), xp.sin(t3)

        out = xp.empty(theta.shape[:-1] + (4,), dtype=theta.dtype)
        out[..., 0] = c3 * c2 * c1
        out[..., 1] = s1
        out[..., 2] = s2 * c1
        out[..., 3] = s3 * c2 * c1
        return out

    def _canonical_sign(self, u):
        """Choose a deterministic representative of the projective orbital."""
        s = xp.ones(u.shape[:-1], dtype=u.dtype)
        x0 = u[..., 0]
        s = xp.where(x0 < -self.nodal_tol, -s, s)

        undecided = xp.abs(x0) <= self.nodal_tol
        for k in range(1, 4):
            xk = u[..., k]
            choose = undecided & (xp.abs(xk) > self.nodal_tol)
            s = xp.where(choose & (xk < 0.0), -s, s)
            undecided = undecided & ~choose
        return s

    def local_vec_to_givens_principal(self, u):
        """Convert canonical normalized local vectors to [-pi/2,pi/2]^3."""
        x0 = u[..., 0]
        x1 = u[..., 1]
        x2 = u[..., 2]
        x3 = u[..., 3]

        t1 = xp.arcsin(xp.clip(x1, -1.0, 1.0))
        rem1_sq = xp.maximum(1.0 - x1 * x1, 0.0)
        rem1 = xp.sqrt(rem1_sq)

        safe_rem1 = xp.where(rem1 > self.nodal_tol, rem1, 1.0)
        y2 = xp.clip(x2 / safe_rem1, -1.0, 1.0)
        t2 = xp.where(rem1 > self.nodal_tol, xp.arcsin(y2), 0.0)

        rem2_sq = xp.maximum(rem1_sq - x2 * x2, 0.0)
        rem2 = xp.sqrt(rem2_sq)
        t3_raw = xp.arctan2(x3, xp.maximum(x0, 0.0))
        t3 = xp.where(rem2 > self.nodal_tol, t3_raw, 0.0)

        half_pi = 0.5 * xp.pi
        theta = xp.stack([t1, t2, t3], axis=-1)
        return xp.clip(theta, -half_pi, half_pi)

    def _vectors_to_principal(self, vecs, return_local: bool = False):
        """Return principal angles and signed norms for original-basis vectors.

        If return_local=True, also return the exact canonical normalized local
        vectors.  Using these as the RHS of the tetra solve avoids rebuilding
        the target from angles near coordinate singularities.
        """
        if self.B is None:
            raise RuntimeError("Call build() before propagating vectors.")

        norms = xp.linalg.norm(vecs, axis=-1)
        if bool(to_host(xp.any(norms < self.nodal_tol))):
            raise ValueError("Encountered a near-zero propagated vector.")

        u_orig = vecs / norms[..., None]
        u_local = xp.einsum("...p,pq->...q", u_orig, self.B, optimize=True)
        sign = self._canonical_sign(u_local)
        u_local = u_local * sign[..., None]
        signed_norm = norms * sign
        theta = self.local_vec_to_givens_principal(u_local)
        if return_local:
            return theta, signed_norm, u_local
        return theta, signed_norm

    def _build_extended_boundary_map(self):
        """Precompute exact projective mapping for indices -1,...,Ng."""
        t0 = time.time()
        ext_n = self.Ng + 2
        p = xp.arange(-1, self.Ng + 1, dtype=xp.int64)
        p1, p2, p3 = xp.meshgrid(p, p, p, indexing="ij")
        theta_ext = xp.stack(
            [
                self.theta0 + p1 * self.dtheta,
                self.theta0 + p2 * self.dtheta,
                self.theta0 + p3 * self.dtheta,
            ],
            axis=-1,
        ).reshape(-1, 3)

        v_ext = self.givens_to_local_vec(theta_ext)
        sign = self._canonical_sign(v_ext)
        v_can = v_ext * sign[:, None]
        theta_can = self.local_vec_to_givens_principal(v_can)

        q = (theta_can - self.theta0) / self.dtheta
        p_can = xp.rint(q).astype(xp.int64)
        max_grid_err = xp.max(xp.abs(q - p_can))
        if float(to_host(max_grid_err)) > 5.0e-8:
            raise RuntimeError(
                "Boundary map did not land on midpoint-grid nodes; "
                f"max index error={float(to_host(max_grid_err))}."
            )
        if bool(to_host(xp.any((p_can < 0) | (p_can >= self.Ng)))):
            raise RuntimeError("Boundary canonicalization produced an out-of-range index.")

        mapped = self.flatten_idx(p_can[:, 0], p_can[:, 1], p_can[:, 2])
        recon = sign[:, None] * self.mos_local[mapped]
        max_vec_err = xp.max(xp.linalg.norm(v_ext - recon, axis=1))
        if float(to_host(max_vec_err)) > 2.0e-10:
            raise RuntimeError(
                "Boundary map failed determinant reconstruction; "
                f"max vector error={float(to_host(max_vec_err))}."
            )

        self._ext_n = ext_n
        self._bc_index = mapped.astype(xp.int64, copy=False)
        self._bc_sign = sign
        del p, p1, p2, p3, theta_ext, v_ext, v_can, theta_can, q, p_can, recon

        print(
            "boundary map built: "
            f"max vector error={float(to_host(max_vec_err)):.3e}, "
            f"time={time.time() - t0:.3f} s"
        )

    def _lookup_boundary(self, p1, p2, p3):
        """Map extended integer nodes (-1...Ng) to interior nodes and signs."""
        e1 = p1 + 1
        e2 = p2 + 1
        e3 = p3 + 1
        flat = (e1 * self._ext_n + e2) * self._ext_n + e3
        return self._bc_index[flat], self._bc_sign[flat]

    def _target_cell(self, theta):
        """Return lower extended cube index and fractional cube coordinate."""
        q = (theta - self.theta0) / self.dtheta
        lo = xp.floor(q).astype(xp.int64)
        lo = xp.clip(lo, -1, self.Ng - 1)
        frac = xp.clip(q - lo, 0.0, 1.0)
        return lo, frac

    def _freudenthal_tetrahedron(self, lo, frac):
        """Choose the unique (up to measure-zero ties) tetrahedron containing target.

        Every unit cube is split into six tetrahedra along its 000--111 body
        diagonal.  Let p0,p1,p2 be the coordinate indices sorted so that

            frac[p0] >= frac[p1] >= frac[p2].

        The selected cube-corner bit vectors are

            000,
            e[p0],
            e[p0] + e[p1],
            111.

        Parameters
        ----------
        lo, frac : (...,3)

        Returns
        -------
        p : (...,4,3) int
            Extended-grid indices of the four tetrahedron vertices.
        order : (...,3) int
            Descending ordering of the fractional coordinates.
        """
        order = xp.argsort(-frac, axis=-1)
        e0 = self._eye3_int[order[..., 0]]
        e1 = self._eye3_int[order[..., 1]]
        z = xp.zeros_like(lo)
        o = xp.ones_like(lo)
        bits = xp.stack([z, e0, e0 + e1, o], axis=-2)
        p = lo[..., None, :] + bits
        return p, order

    @staticmethod
    def _det4(p, q, r, s):
        """Explicit determinant det([p q r s]) for (...,4) column vectors.

        This is fully elementwise and avoids xp.linalg.det on a batch of 4x4
        matrices.
        """
        p01 = p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
        p02 = p[..., 0] * q[..., 2] - p[..., 2] * q[..., 0]
        p03 = p[..., 0] * q[..., 3] - p[..., 3] * q[..., 0]
        p12 = p[..., 1] * q[..., 2] - p[..., 2] * q[..., 1]
        p13 = p[..., 1] * q[..., 3] - p[..., 3] * q[..., 1]
        p23 = p[..., 2] * q[..., 3] - p[..., 3] * q[..., 2]

        r01 = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
        r02 = r[..., 0] * s[..., 2] - r[..., 2] * s[..., 0]
        r03 = r[..., 0] * s[..., 3] - r[..., 3] * s[..., 0]
        r12 = r[..., 1] * s[..., 2] - r[..., 2] * s[..., 1]
        r13 = r[..., 1] * s[..., 3] - r[..., 3] * s[..., 1]
        r23 = r[..., 2] * s[..., 3] - r[..., 3] * s[..., 2]

        return (
            p01 * r23
            - p02 * r13
            + p03 * r12
            + p12 * r03
            - p13 * r02
            + p23 * r01
        )

    def _analytic_tetra_denominator(self, lo, order, dtype):
        """Closed-form det([v0 v1 v2 v3]) for the six tetrahedron types.

        Let the lower cube angles be (a,b,c), the upper values be
        (A,B,C)=(a+h,b+h,c+h), and h=dtheta.  For the six descending
        fractional-coordinate orders, the selected vertices and determinants
        are

          012: 000,100,110,111   +sin(h)^3 cos(A)^2 cos(B)
          021: 000,100,101,111   -sin(h)^3 cos(b) cos(A)^2
          102: 000,010,110,111   -sin(h)^3 cos(a) cos(A) cos(B)
          120: 000,010,011,111   +sin(h)^3 cos(a)^2 cos(B)
          201: 000,001,101,111   +sin(h)^3 cos(a) cos(b) cos(A)
          210: 000,001,011,111   -sin(h)^3 cos(a)^2 cos(b)

        The formulas remain valid for the one-layer extended/ghost cubes used
        at the projective chart boundary.
        """
        lower = self.theta0 + lo.astype(dtype) * self.dtheta
        a = lower[..., 0]
        b = lower[..., 1]
        h = xp.asarray(self.dtheta, dtype=dtype)
        A = a + h
        B = b + h

        ca = xp.cos(a)
        cb = xp.cos(b)
        cA = xp.cos(A)
        cB = xp.cos(B)
        sh3 = xp.sin(h) ** 3

        # A compact unique integer for each permutation of (0,1,2).
        # 012->5, 021->7, 102->11, 120->15, 201->19, 210->21.
        code = order[..., 0] * 9 + order[..., 1] * 3 + order[..., 2]

        d012 = sh3 * cA * cA * cB
        d021 = -sh3 * cb * cA * cA
        d102 = -sh3 * ca * cA * cB
        d120 = sh3 * ca * ca * cB
        d201 = sh3 * ca * cb * cA
        d210 = -sh3 * ca * ca * cb

        den = xp.where(
            code == 5,
            d012,
            xp.where(
                code == 7,
                d021,
                xp.where(
                    code == 11,
                    d102,
                    xp.where(code == 15, d120, xp.where(code == 19, d201, d210)),
                ),
            ),
        )
        return den

    def _solve_exact_tetra(self, target_local, p, lo, order):
        """Analytic exact coefficients for the four selected extended vertices.

        No numerical linear solve is performed.  Cramer's rule is evaluated
        using explicit det4 numerators and the specialized closed-form
        denominator for the selected Freudenthal tetrahedron.

        Parameters
        ----------
        target_local : (...,4)
        p            : (...,4,3)
        lo           : (...,3)
            Lower extended-grid cube index.
        order        : (...,3)
            Descending order of fractional coordinates from
            _freudenthal_tetrahedron.

        Returns
        -------
        c : (...,4)
            Coefficients satisfying sum_k c_k v_k = target_local.
        corner_local : (...,4,4)
            The four extended-grid local vectors v_k, stored by corner.
        """
        theta_corner = self.theta0 + p.astype(target_local.dtype) * self.dtheta
        corner_local = self.givens_to_local_vec(theta_corner)

        v0 = corner_local[..., 0, :]
        v1 = corner_local[..., 1, :]
        v2 = corner_local[..., 2, :]
        v3 = corner_local[..., 3, :]
        u = target_local

        den = self._analytic_tetra_denominator(lo, order, target_local.dtype)

        n0 = self._det4(u, v1, v2, v3)
        n1 = self._det4(v0, u, v2, v3)
        n2 = self._det4(v0, v1, u, v3)
        n3 = self._det4(v0, v1, v2, u)
        c = xp.stack((n0, n1, n2, n3), axis=-1) / den[..., None]
        return c, corner_local

    def build(self, ham, trial, check: bool = True, **_ignored):
        """Attach Hamiltonian and construct the trial-adapted physical basis."""
        self.ham = ham
        self.trial = xp.asarray(trial).copy()
        self.trial /= xp.linalg.norm(self.trial)
        self.M = int(self.trial.size)
        if self.M != 4:
            raise ValueError("This implementation assumes one electron in four real orbitals.")

        self.B = h2plus_symmetry_basis(self.trial)
        self.mos = self.mos_local @ self.B.T
        self.trial_mos = self.mos @ self.trial

        if bool(to_host(xp.any(self.trial_mos <= 0.0))):
            raise RuntimeError("Interior midpoint grid contains a non-positive trial overlap.")

        self.Us = xp.asarray([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
        self.aUs = xp.einsum("a,apq->pq", ham.a, self.Us, optimize=True)
        self.gtrial = self.trial @ self.aUs

        if check:
            overlap_err = xp.max(xp.abs(self.mos_local[:, 0] - self.trial_mos))
            print(f"max trial-overlap identity error={float(to_host(overlap_err)):.3e}")

    def mat_vec(
        self,
        psi,
        term_chunk: int = 4,
        source_chunk: int = 8192,
        synchronize: bool = False,
    ):
        """Apply one free or constrained-path exact-tetra transfer step.

        Compared with the previous exact-tetra implementation, this version
        performs no batched 4x4 solve.  The exact coefficients use explicit
        Cramer numerators and one of six analytic denominator formulas.
        """
        if self.Us is None:
            raise RuntimeError("Call build() first.")
        psi = xp.asarray(psi)
        if psi.size != self.Nh:
            raise ValueError(f"psi has size {psi.size}, expected {self.Nh}.")
        if term_chunk < 1 or source_chunk < 1:
            raise ValueError("term_chunk and source_chunk must be positive.")

        out = xp.zeros(self.Nh, dtype=psi.dtype)
        nterms = int(self.ham.nterms)

        if self.constraint_path:
            cp_total = 0
            cp_rejected = xp.asarray(0, dtype=xp.int64)
            cp_total_flux = xp.asarray(0.0, dtype=psi.dtype)
            cp_rejected_flux = xp.asarray(0.0, dtype=psi.dtype)

        max_solve_res = xp.asarray(0.0, dtype=psi.dtype)
        max_abs_coeff = xp.asarray(0.0, dtype=psi.dtype)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Us = self.Us[a0:a1]
            coeff = xp.asarray(self.ham.a[a0:a1], dtype=psi.dtype)

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                src_mos = self.mos[s0:s1]
                src_psi = psi[s0:s1]
                old_overlap = self.trial_mos[None, s0:s1]

                targets = xp.einsum("apq,sq->asp", Us, src_mos, optimize=True)
                theta, signed_norm, target_local = self._vectors_to_principal(
                    targets, return_local=True
                )
                lo, frac = self._target_cell(theta)
                p, _order = self._freudenthal_tetrahedron(lo, frac)
                tetra_c, corner_local = self._solve_exact_tetra(
                    target_local, p, lo, _order
                )

                if self.check_tetra_solve:
                    recon = xp.einsum("...k,...kp->...p", tetra_c, corner_local, optimize=True)
                    res = xp.linalg.norm(recon - target_local, axis=-1)
                    max_solve_res = xp.maximum(max_solve_res, xp.max(res))
                    max_abs_coeff = xp.maximum(max_abs_coeff, xp.max(xp.abs(tetra_c)))
                    del recon, res

                if self.constraint_path:
                    raw_new_overlap = xp.einsum(
                        "p,asp->as", self.trial, targets, optimize=True
                    )
                    gbar = coeff[:, None] * raw_new_overlap / old_overlap
                    allowed = gbar > self.cp_tol

                    branch_flux = xp.abs(gbar * src_psi[None, :])
                    cp_total += int(allowed.size)
                    cp_rejected += xp.count_nonzero(~allowed)
                    cp_total_flux += xp.sum(branch_flux)
                    cp_rejected_flux += xp.sum(
                        xp.where(allowed, xp.zeros((), dtype=psi.dtype), branch_flux)
                    )

                    base = coeff[:, None] * signed_norm * src_psi[None, :]
                    base = base / old_overlap
                    base = xp.where(allowed, base, xp.zeros((), dtype=psi.dtype))
                else:
                    base = coeff[:, None] * signed_norm * src_psi[None, :]
                    if self.importance:
                        base = base / old_overlap

                # Four exact physical coefficients, then projective boundary mapping.
                for k in range(4):
                    pk = p[..., k, :]
                    p1 = pk[..., 0]
                    p2 = pk[..., 1]
                    p3 = pk[..., 2]
                    ck = tetra_c[..., k]

                    if self.constraint_path and self.cp_boundary == "absorbing":
                        valid = (
                            (p1 >= 0) & (p1 < self.Ng)
                            & (p2 >= 0) & (p2 < self.Ng)
                            & (p3 >= 0) & (p3 < self.Ng)
                        )
                        p1s = xp.clip(p1, 0, self.Ng - 1)
                        p2s = xp.clip(p2, 0, self.Ng - 1)
                        p3s = xp.clip(p3, 0, self.Ng - 1)
                        dest = self.flatten_idx(p1s, p2s, p3s)
                        vals = base * ck
                        if self.importance:
                            vals = vals * self.trial_mos[dest]
                        vals = xp.where(valid, vals, xp.zeros((), dtype=psi.dtype))
                    else:
                        dest, bc_sign = self._lookup_boundary(p1, p2, p3)
                        vals = base * ck * bc_sign
                        if self.importance:
                            vals = vals * self.trial_mos[dest]

                    out += xp.bincount(
                        dest.ravel(), weights=vals.ravel(), minlength=self.Nh
                    )

                del targets, theta, signed_norm, target_local, lo, frac, p
                del tetra_c, corner_local, base, _order
                if self.constraint_path:
                    del raw_new_overlap, gbar, allowed, branch_flux

        if self.constraint_path:
            total_flux = float(to_host(cp_total_flux))
            rejected_flux = float(to_host(cp_rejected_flux))
            rejected = int(to_host(cp_rejected))
            self.last_cp_stats = {
                "total_branches": cp_total,
                "rejected_branches": rejected,
                "rejected_fraction": rejected / max(cp_total, 1),
                "total_abs_flux": total_flux,
                "rejected_abs_flux": rejected_flux,
                "rejected_flux_fraction": rejected_flux / max(total_flux, 1.0e-300),
            }
        else:
            self.last_cp_stats = None

        if self.check_tetra_solve:
            self.last_tetra_stats = {
                "max_local_reconstruction_residual": float(to_host(max_solve_res)),
                "max_abs_tetra_coefficient": float(to_host(max_abs_coeff)),
            }
        else:
            self.last_tetra_stats = None

        if synchronize and hasattr(xp, "cuda"):
            xp.cuda.Stream.null.synchronize()
        return out

    def mat_vec_chunked(
        self, psi, chunk_terms: int = 4, source_chunk: int = 8192, **_ignored
    ):
        return self.mat_vec(
            psi, term_chunk=chunk_terms, source_chunk=source_chunk
        )

    def get_initial_state(self, initial=None, coefficient: float = 1.0):
        """Represent an arbitrary one-electron determinant by exact tetra deposition."""
        if self.B is None or self.mos is None:
            raise RuntimeError("Call build() before get_initial_state().")

        if initial is None:
            initial_vec = self.trial.copy()
        else:
            initial_vec = xp.asarray(initial, dtype=self.mos_local.dtype).copy()

        if initial_vec.ndim != 1 or int(initial_vec.size) != self.M:
            raise ValueError(
                f"initial must be a length-{self.M} orbital vector; got {initial_vec.shape}."
            )

        coeff = xp.asarray(coefficient, dtype=self.mos_local.dtype)
        coeff_host = float(to_host(coeff))
        if not np.isfinite(coeff_host):
            raise ValueError("coefficient must be finite.")
        if self.constraint_path and coeff_host <= 0.0:
            raise ValueError("A constrained-path initial coefficient must be positive.")

        initial_norm = xp.linalg.norm(initial_vec)
        initial_norm_host = float(to_host(initial_norm))
        if initial_norm_host < self.nodal_tol:
            raise ValueError("Initial determinant has near-zero norm.")

        raw_overlap = xp.dot(self.trial, initial_vec)
        raw_overlap_before = float(to_host(raw_overlap))
        phase_flipped = False

        if self.constraint_path:
            overlap_scale = max(initial_norm_host, 1.0)
            if abs(raw_overlap_before) <= self.cp_tol * overlap_scale:
                raise ValueError(
                    "The initial determinant is orthogonal (within cp_tol) to the trial."
                )
            if raw_overlap_before < 0.0:
                initial_vec *= -1.0
                raw_overlap *= -1.0
                phase_flipped = True

        theta_b, signed_norm_b, local_b = self._vectors_to_principal(
            initial_vec[None, :], return_local=True
        )
        theta = theta_b[0]
        signed_norm = signed_norm_b[0]
        target_local = local_b[0]
        lo, frac = self._target_cell(theta[None, :])
        p, order = self._freudenthal_tetrahedron(lo, frac)
        tetra_c, _corner_local = self._solve_exact_tetra(
            target_local[None, :], p, lo, order
        )
        p = p[0]
        tetra_c = tetra_c[0]

        psi = xp.zeros(self.Nh, dtype=self.mos_local.dtype)
        base = coeff * signed_norm

        for k in range(4):
            p1, p2, p3 = p[k, 0], p[k, 1], p[k, 2]
            ck = tetra_c[k]

            if self.constraint_path and self.cp_boundary == "absorbing":
                valid = (
                    (p1 >= 0) & (p1 < self.Ng)
                    & (p2 >= 0) & (p2 < self.Ng)
                    & (p3 >= 0) & (p3 < self.Ng)
                )
                if not bool(to_host(valid)):
                    continue
                dest = self.flatten_idx(p1, p2, p3)
                val = base * ck
            else:
                dest, bc_sign = self._lookup_boundary(p1, p2, p3)
                val = base * ck * bc_sign

            if self.importance:
                val *= self.trial_mos[dest]
            psi[dest] += val

        # Check the actual physical initialization error after all boundary maps.
        bare = psi / self.trial_mos if self.importance else psi
        recon = bare @ self.mos
        target_phys = coeff * initial_vec
        init_err = xp.linalg.norm(recon - target_phys)

        self.last_initial_info = {
            "raw_overlap_before_orientation": raw_overlap_before,
            "oriented_overlap": float(to_host(xp.dot(self.trial, initial_vec))),
            "phase_flipped_for_cp": phase_flipped,
            "initial_norm": initial_norm_host,
            "coefficient": coeff_host,
            "theta": to_host(theta).copy(),
            "freudenthal_order": to_host(order[0]).copy(),
            "tetra_coefficients": to_host(tetra_c).copy(),
            "physical_reconstruction_error": float(to_host(init_err)),
        }
        return psi

    def reconstruct_many_body(self, weight):
        """Return the represented physical one-electron four-vector."""
        weight = xp.asarray(weight)
        bare = weight / self.trial_mos if self.importance else weight
        return bare @ self.mos

    def physical_norm(self, weight):
        return xp.linalg.norm(self.reconstruct_many_body(weight))

    def exact_G_on_many_body(self, vec):
        """Apply Gbar=sum_i a_i U_i directly in the four-dimensional space."""
        return self.aUs @ xp.asarray(vec)

    def energy(self, weight):
        bare = weight / self.trial_mos if self.importance else weight
        denom = xp.dot(self.trial_mos, bare)
        psi_mb = bare @ self.mos
        num = xp.dot(psi_mb, self.gtrial)
        energy = self.ham.Lambda[-1] * (1.0 - num / denom)
        return energy, denom

    def normalize(self, weight, mode: str = "auto"):
        mode = str(mode).lower()
        if mode == "auto":
            mode = "population" if self.constraint_path else "physical"
        if mode == "population":
            nrm = xp.sum(xp.abs(weight))
        elif mode == "physical":
            nrm = self.physical_norm(weight)
        else:
            raise ValueError("normalize mode must be 'auto', 'population', or 'physical'.")
        if float(to_host(nrm)) < self.nodal_tol:
            raise RuntimeError("State has near-zero normalization.")
        return weight / nrm

    def diagnostics(self, step, psi, Gpsi):
        E_i, denom = self.energy(psi)
        psi_mb = self.reconstruct_many_body(psi)
        proj_mb = self.reconstruct_many_body(Gpsi)
        exact_mb = self.aUs @ psi_mb

        abs_err = xp.linalg.norm(proj_mb - exact_mb)
        rel_err = abs_err / xp.maximum(xp.linalg.norm(exact_mb), self.nodal_tol)
        trial_err = xp.abs(xp.dot(self.trial, proj_mb - exact_mb))

        msg = (
            f"step={step}, E={float(to_host(E_i)):.15f}, "
            f"trial_defect={float(to_host(trial_err)):.3e}, "
            f"abs_wfn_defect={float(to_host(abs_err)):.3e}, "
            f"rel_wfn_defect={float(to_host(rel_err)):.3e}, "
            f"denom={float(to_host(denom)):.6e}"
        )
        if self.last_cp_stats is not None:
            msg += (
                f", cp_reject={self.last_cp_stats['rejected_fraction']:.6e}, "
                f"cp_reject_flux={self.last_cp_stats['rejected_flux_fraction']:.6e}"
            )
        if self.last_tetra_stats is not None:
            msg += (
                f", tetra_res={self.last_tetra_stats['max_local_reconstruction_residual']:.3e}, "
                f"max|tetra_c|={self.last_tetra_stats['max_abs_tetra_coefficient']:.3e}"
            )
        print(msg)

    def exact_trial_energy_under_G(self):
        denom = xp.dot(self.trial, self.trial)
        num = xp.dot(self.gtrial, self.trial)
        return self.ham.Lambda[-1] * (1.0 - num / denom)

    def run(
        self,
        start: int,
        stop: int,
        psi,
        print_every: int = 10,
        normalize_every: int = 10,
        term_chunk: int = 4,
        source_chunk: int = 8192,
        normalize_mode: str = "auto",
    ):
        for istep in range(start, stop):
            Gpsi = self.mat_vec(
                psi, term_chunk=term_chunk, source_chunk=source_chunk
            )
            if print_every > 0 and istep % print_every == 0:
                self.diagnostics(istep, psi, Gpsi)
            psi = Gpsi
            if normalize_every > 0 and istep % normalize_every == 0:
                psi = self.normalize(psi, mode=normalize_mode)
        return psi


# Example:
#
# me = GivensMasterEquation(
#     Ng=65,
#     importance=True,
#     constraint_path=False,       # free projection
#     cp_boundary="projective",
#     check_tetra_solve=True,  # verifies analytic coefficients by reconstruction
# )
# me.build(ham, trial)
# psi = me.get_initial_state()
# psi = me.run(
#     0, 200, psi,
#     print_every=10,
#     normalize_every=10,
#     term_chunk=4,
#     source_chunk=8192,
# )
#
# For direct branchwise constrained path:
#
# me = GivensMasterEquation(
#     Ng=65,
#     importance=True,
#     constraint_path=True,
#     cp_tol=0.0,
#     cp_boundary="projective",
# )
