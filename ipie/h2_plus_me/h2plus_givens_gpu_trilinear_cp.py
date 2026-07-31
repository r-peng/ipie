"""
GPU-friendly 3-angle Givens-grid master equation with free-projection and constrained-path trilinear deposition.

The code is specialized to one electron in four real orbitals.  The local
Givens chart is

    phi_local(t1,t2,t3) =
        [cos(t3) cos(t2) cos(t1),
         sin(t1),
         sin(t2) cos(t1),
         sin(t3) cos(t2) cos(t1)].

The first local basis vector is the trial orbital.  For a trial of the form
[0,a,0,b], the physical basis is chosen as

    B = [[0,  0, 1, 0],
         [a, -b, 0, 0],
         [0,  0, 0, 1],
         [b,  a, 0, 0]].

Each exact target U_i |phi_j> is canonicalized into the principal chart and
then deposited onto the 2^3 neighboring midpoint-grid nodes.

Free projection uses the projective boundary identification: a ghost node is
mapped to the equivalent interior determinant, including its overall sign.

Constrained-path propagation first evaluates the exact importance-sampled
branch factor

    gbar_ij = a_i <Psi_T|U_i|phi_j> / <Psi_T|phi_j>

and deletes the entire branch whenever gbar_ij <= cp_tol.  By default, the
remaining branch is deposited with an absorbing nodal boundary: interpolation
corners outside the principal chart are discarded rather than wrapped to the
opposite side.  This is the grid analogue of a Dirichlet constrained-path
boundary.  An optional projective mode is provided for controlled comparisons.
"""

from __future__ import annotations

import itertools
import time
from typing import Optional, Tuple

import numpy as np

try:
    from ipie.utils.backend import arraylib as xp
    from ipie.utils.backend import to_host
except ImportError:  # CPU fallback for testing outside an ipie environment.
    xp = np

    def to_host(x):
        return np.asarray(x)


def h2plus_symmetry_basis(trial, thresh: float = 1.0e-12):
    """Return the requested trial/symmetry-adapted orthogonal basis.

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
        raise RuntimeError(f"Constructed basis is not orthonormal; error={float(to_host(err))}.")
    return B


class GivensMasterEquation:
    """One-electron, four-orbital Givens-grid transfer operator.

    Parameters
    ----------
    Ng
        Number of midpoint grid nodes per angle.  Odd Ng puts the trial state
        theta=(0,0,0) exactly on the grid.
    importance
        If True, evolve f(theta)=<Psi_T|phi(theta)> psi(theta), matching the
        representation used in the original script.
    nodal_tol
        Tolerance used only to choose a deterministic sign when the propagated
        orbital has essentially zero overlap with the trial.
    """

    def __init__(
        self,
        Ng: int = 33,
        importance: bool = True,
        nodal_tol: float = 1.0e-13,
        constraint_path: bool = False,
        cp_tol: float = 1.0e-14,
        cp_boundary: str = "absorbing",
        cp_deposition: str = "wavefunction",
    ):
        self.importance = bool(importance)
        self.Ng = int(Ng)
        self.nodal_tol = float(nodal_tol)
        self.constraint_path = bool(constraint_path)
        self.cp_tol = float(cp_tol)
        self.cp_boundary = str(cp_boundary).lower()
        self.cp_deposition = str(cp_deposition).lower()

        if self.constraint_path and not self.importance:
            raise ValueError(
                "constraint_path=True currently requires importance=True."
            )
        if self.cp_boundary not in {"absorbing", "projective"}:
            raise ValueError("cp_boundary must be 'absorbing' or 'projective'.")
        if self.cp_deposition not in {"wavefunction", "walker"}:
            raise ValueError("cp_deposition must be 'wavefunction' or 'walker'.")
        if self.cp_deposition == "walker" and self.cp_boundary == "projective":
            raise ValueError(
                "cp_deposition='walker' must use cp_boundary='absorbing'."
            )
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

        # Interior midpoint-grid determinants in local coordinates.
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
        self.last_initial_info = None

        # Extended-grid boundary lookup, built after the interior grid exists.
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
        """Choose a deterministic projective representative for local vectors.

        The primary convention is u[...,0] >= 0.  At the nodal surface, where
        u[...,0] is numerically zero, the first nonzero later component is made
        positive.  Returns signs in {+1,-1} such that u*sign is canonical.
        """
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
        """Convert canonical normalized local vectors to the closed principal chart.

        Unlike the old nearest-grid routine, this does not clip angles to the
        first/last midpoint.  It returns angles in the closed interval
        [-pi/2,pi/2], which is required for boundary-aware interpolation.
        """
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
        # If rem2=0, t3 is redundant; choose t3=0 deterministically.
        t3_raw = xp.arctan2(x3, xp.maximum(x0, 0.0))
        t3 = xp.where(rem2 > self.nodal_tol, t3_raw, 0.0)

        half_pi = 0.5 * xp.pi
        theta = xp.stack([t1, t2, t3], axis=-1)
        return xp.clip(theta, -half_pi, half_pi)

    def _vectors_to_principal(self, vecs) -> Tuple[object, object]:
        """Return principal angles and signed norms for original-basis vectors."""
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
        return theta, signed_norm

    def _build_extended_boundary_map(self):
        """Precompute exact mapping for indices -1,...,Ng in every coordinate.

        A trilinear stencil for a target inside the closed chart needs at most
        one ghost layer.  For every extended-grid node, this routine:

          1. constructs its determinant from the possibly out-of-range angles;
          2. fixes the projective sign;
          3. converts it back to principal angles;
          4. identifies the exactly matching interior midpoint node.

        Therefore simultaneous face crossings, edges, and corners are handled
        by the geometry of the determinant rather than by composing ad hoc rules.
        """
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
            raise RuntimeError("Boundary canonicalization produced an out-of-range interior index.")

        mapped = self.flatten_idx(p_can[:, 0], p_can[:, 1], p_can[:, 2])

        # Verify v_ext = sign * v_interior after mapping.
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
        """Return lower extended indices and interpolation fractions."""
        q = (theta - self.theta0) / self.dtheta
        lo = xp.floor(q).astype(xp.int64)
        lo = xp.clip(lo, -1, self.Ng - 1)
        frac = xp.clip(q - lo, 0.0, 1.0)
        return lo, frac

    def build(self, ham, trial, check: bool = True, **_ignored):
        """Attach the Hamiltonian and construct the trial-adapted physical basis."""
        self.ham = ham
        self.trial = xp.asarray(trial).copy()
        self.trial /= xp.linalg.norm(self.trial)
        self.M = int(self.trial.size)
        if self.M != 4:
            raise ValueError("This implementation assumes one electron in four real orbitals.")

        self.B = h2plus_symmetry_basis(self.trial)
        self.mos = self.mos_local @ self.B.T
        self.trial_mos = self.mos @ self.trial

        # All midpoint nodes are inside the positive-overlap chart.
        if bool(to_host(xp.any(self.trial_mos <= 0.0))):
            raise RuntimeError("Interior midpoint grid contains a non-positive trial overlap.")

        self.Us = xp.asarray([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
        self.aUs = xp.einsum("a,apq->pq", ham.a, self.Us, optimize=True)
        self.gtrial = self.trial @ self.aUs

        if check:
            # Because B[:,0] is exactly the trial, the trial overlap must equal
            # the first local coefficient c1*c2*c3.
            overlap_err = xp.max(xp.abs(self.mos_local[:, 0] - self.trial_mos))
            print(f"max trial-overlap identity error={float(to_host(overlap_err)):.3e}")

    def mat_vec(
        self,
        psi,
        term_chunk: int = 8,
        source_chunk: int = 65536,
        corner_batch: int = 4,
        synchronize: bool = False,
    ):
        """Apply the free or constrained-path trilinear transfer operator.

        For constrained path, the exact branch factor

            gbar_ij = a_i <Psi_T|U_i|phi_j> / <Psi_T|phi_j>

        is evaluated before interpolation.  The whole branch is removed when
        gbar_ij <= cp_tol; the constraint is never applied corner by corner.

        cp_deposition='wavefunction' interpolates the propagated determinant
        and is recommended for energies and reconstructed wavefunctions.
        cp_deposition='walker' distributes gbar directly as a positive walker
        histogram.
        """
        if self.Us is None:
            raise RuntimeError("Call build() first.")
        psi = xp.asarray(psi)
        if psi.size != self.Nh:
            raise ValueError(f"psi has size {psi.size}, expected {self.Nh}.")
        if term_chunk < 1:
            raise ValueError("term_chunk must be at least 1.")
        if source_chunk < 1:
            raise ValueError("source_chunk must be at least 1.")
        if corner_batch < 1 or corner_batch > 8:
            raise ValueError("corner_batch must be between 1 and 8.")

        out = xp.zeros(self.Nh, dtype=psi.dtype)
        nterms = int(self.ham.nterms)
        corners = tuple(itertools.product((0, 1), repeat=3))

        if self.constraint_path:
            cp_total = 0
            cp_rejected = xp.asarray(0, dtype=xp.int64)
            cp_total_flux = xp.asarray(0.0, dtype=psi.dtype)
            cp_rejected_flux = xp.asarray(0.0, dtype=psi.dtype)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Us = self.Us[a0:a1]
            coeff = xp.asarray(self.ham.a[a0:a1], dtype=psi.dtype)

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                src_mos = self.mos[s0:s1]
                src_psi = psi[s0:s1]
                old_overlap = self.trial_mos[None, s0:s1]

                # Exact unnormalized targets, shape (nterm,nsource,4).
                targets = xp.einsum("apq,sq->asp", Us, src_mos, optimize=True)
                theta, signed_norm = self._vectors_to_principal(targets)
                lo, frac = self._target_cell(theta)

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

                    if self.cp_deposition == "walker":
                        base = xp.where(
                            allowed,
                            gbar * src_psi[None, :],
                            xp.zeros((), dtype=psi.dtype),
                        )
                        multiply_destination_overlap = False
                    else:
                        base = coeff[:, None] * signed_norm * src_psi[None, :]
                        base = base / old_overlap
                        base = xp.where(
                            allowed, base, xp.zeros((), dtype=psi.dtype)
                        )
                        multiply_destination_overlap = True
                else:
                    base = coeff[:, None] * signed_norm * src_psi[None, :]
                    if self.importance:
                        base = base / old_overlap
                    multiply_destination_overlap = self.importance

                dest_batch = []
                vals_batch = []
                for icorner, bits in enumerate(corners):
                    b0, b1, b2 = bits
                    p1 = lo[..., 0] + b0
                    p2 = lo[..., 1] + b1
                    p3 = lo[..., 2] + b2

                    w1 = frac[..., 0] if b0 else (1.0 - frac[..., 0])
                    w2 = frac[..., 1] if b1 else (1.0 - frac[..., 1])
                    w3 = frac[..., 2] if b2 else (1.0 - frac[..., 2])
                    interp = w1 * w2 * w3

                    if self.constraint_path and self.cp_boundary == "absorbing":
                        valid = (
                            (p1 >= 0) & (p1 < self.Ng)
                            & (p2 >= 0) & (p2 < self.Ng)
                            & (p3 >= 0) & (p3 < self.Ng)
                        )
                        p1_safe = xp.clip(p1, 0, self.Ng - 1)
                        p2_safe = xp.clip(p2, 0, self.Ng - 1)
                        p3_safe = xp.clip(p3, 0, self.Ng - 1)
                        dest = self.flatten_idx(p1_safe, p2_safe, p3_safe)
                        vals = base * interp
                        if multiply_destination_overlap:
                            vals = vals * self.trial_mos[dest]
                        vals = xp.where(
                            valid, vals, xp.zeros((), dtype=psi.dtype)
                        )
                    else:
                        dest, bc_sign = self._lookup_boundary(p1, p2, p3)
                        vals = base * interp * bc_sign
                        if multiply_destination_overlap:
                            vals = vals * self.trial_mos[dest]

                    dest_batch.append(dest.ravel())
                    vals_batch.append(vals.ravel())
                    flush = (
                        len(dest_batch) == corner_batch
                        or icorner == len(corners) - 1
                    )
                    if flush:
                        dest_all = xp.concatenate(dest_batch)
                        vals_all = xp.concatenate(vals_batch)
                        out += xp.bincount(
                            dest_all, weights=vals_all, minlength=self.Nh
                        )
                        dest_batch.clear()
                        vals_batch.clear()
                        del dest_all, vals_all

                del targets, theta, signed_norm, lo, frac, base
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

        if synchronize and hasattr(xp, "cuda"):
            xp.cuda.Stream.null.synchronize()
        return out

    def mat_vec_chunked(
        self, psi, chunk_terms: int = 8, source_chunk: int = 65536,
        corner_batch: int = 4, **_ignored
    ):
        """Compatibility wrapper for the old call site."""
        return self.mat_vec(
            psi, term_chunk=chunk_terms, source_chunk=source_chunk,
            corner_batch=corner_batch
        )

#    def get_initial_state(self):
#        """Return the trial determinant on the grid.
#
#        For odd Ng this is exactly one central grid node.  For even Ng, the
#        trial is represented by trilinear weights on the eight surrounding
#        nodes.
#        """
#        psi = xp.zeros(self.Nh, dtype=self.mos_local.dtype)
#        q = (xp.zeros(3, dtype=self.mos_local.dtype) - self.theta0) / self.dtheta
#        lo = xp.floor(q).astype(xp.int64)
#        frac = q - lo
#
#        for bits in itertools.product((0, 1), repeat=3):
#            b0, b1, b2 = bits
#            p1, p2, p3 = lo[0] + b0, lo[1] + b1, lo[2] + b2
#            w = (
#                (frac[0] if b0 else 1.0 - frac[0])
#                * (frac[1] if b1 else 1.0 - frac[1])
#                * (frac[2] if b2 else 1.0 - frac[2])
#            )
#            dest, bc_sign = self._lookup_boundary(p1, p2, p3)
#            val = w * bc_sign
#            if self.importance:
#                val *= self.trial_mos[dest]
#            psi[dest] += val
#        return psi

    def get_initial_state(
        self,
        initial=None,
        coefficient: float = 1.0,
        deposition: str = "auto",
    ):
        """Represent an arbitrary one-electron Slater determinant on the grid.

        Parameters
        ----------
        initial
            Length-4 real orbital vector in the original basis.  ``None`` uses
            the trial orbital, preserving the behavior of the previous code.
            The vector need not be normalized; its norm is retained in the
            represented state.
        coefficient
            Overall coefficient multiplying the initial determinant.  For a
            constrained-path run this must be positive.
        deposition
            ``"wavefunction"`` interpolates the determinant itself and is the
            recommended initialization for energies/reconstructed states.
            ``"walker"`` distributes the exact initial importance weight and
            is available only for constrained-path importance-sampled runs.
            ``"auto"`` uses ``self.cp_deposition`` for constrained path and
            ``"wavefunction"`` for free projection.

        Notes
        -----
        In a constrained-path calculation, the initial determinant must have
        nonzero overlap with the trial.  If that overlap is negative, the
        determinant is multiplied by -1 before gridding.  This changes only
        the physically irrelevant global phase of a single initial state and
        places the walker in the positive-overlap sector required by CP.
        """
        if self.B is None or self.mos is None:
            raise RuntimeError("Call build() before get_initial_state().")

        if initial is None:
            initial_vec = self.trial.copy()
        else:
            initial_vec = xp.asarray(initial, dtype=self.mos_local.dtype).copy()

        if initial_vec.ndim != 1 or int(initial_vec.size) != self.M:
            raise ValueError(
                f"initial must be a length-{self.M} orbital vector; "
                f"received shape {initial_vec.shape}."
            )

        coeff = xp.asarray(coefficient, dtype=self.mos_local.dtype)
        coeff_host = float(to_host(coeff))
        if not np.isfinite(coeff_host):
            raise ValueError("coefficient must be finite.")
        if self.constraint_path and coeff_host <= 0.0:
            raise ValueError(
                "A constrained-path initial walker must have a positive coefficient."
            )

        deposition = str(deposition).lower()
        if deposition == "auto":
            deposition = self.cp_deposition if self.constraint_path else "wavefunction"
        if deposition not in {"wavefunction", "walker"}:
            raise ValueError("deposition must be 'auto', 'wavefunction', or 'walker'.")
        if deposition == "walker" and not (self.constraint_path and self.importance):
            raise ValueError(
                "deposition='walker' requires constraint_path=True and importance=True."
            )

        initial_norm = xp.linalg.norm(initial_vec)
        initial_norm_host = float(to_host(initial_norm))
        if initial_norm_host < self.nodal_tol:
            raise ValueError("Initial determinant has near-zero norm.")

        raw_overlap = xp.dot(self.trial, initial_vec)
        raw_overlap_host = float(to_host(raw_overlap))
        phase_flipped = False

        if self.constraint_path:
            # The CP importance ratio is undefined at the trial node.
            overlap_scale = max(initial_norm_host, 1.0)
            if abs(raw_overlap_host) <= self.cp_tol * overlap_scale:
                raise ValueError(
                    "The initial determinant is orthogonal (within cp_tol) to the "
                    "trial state and cannot initialize a constrained-path walk."
                )
            if raw_overlap_host < 0.0:
                initial_vec *= -1.0
                raw_overlap *= -1.0
                raw_overlap_host *= -1.0
                phase_flipped = True

        # Exact continuum target: initial_vec = signed_norm * phi(theta).
        theta_batch, signed_norm_batch = self._vectors_to_principal(
            initial_vec[None, :]
        )
        theta = theta_batch[0]
        signed_norm = signed_norm_batch[0]
        lo, frac = self._target_cell(theta[None, :])
        lo = lo[0]
        frac = frac[0]

        psi = xp.zeros(self.Nh, dtype=self.mos_local.dtype)
        retained_weight = xp.asarray(0.0, dtype=self.mos_local.dtype)

        if deposition == "wavefunction":
            # Bare coefficient multiplying the canonical determinant.
            base = coeff * signed_norm
        else:
            # Exact importance-sampled weight of the single initial walker.
            base = coeff * raw_overlap

        for bits in itertools.product((0, 1), repeat=3):
            b0, b1, b2 = bits
            p1 = lo[0] + b0
            p2 = lo[1] + b1
            p3 = lo[2] + b2
            interp = (
                (frac[0] if b0 else 1.0 - frac[0])
                * (frac[1] if b1 else 1.0 - frac[1])
                * (frac[2] if b2 else 1.0 - frac[2])
            )

            if self.constraint_path and self.cp_boundary == "absorbing":
                valid = (
                    (p1 >= 0) & (p1 < self.Ng)
                    & (p2 >= 0) & (p2 < self.Ng)
                    & (p3 >= 0) & (p3 < self.Ng)
                )
                if not bool(to_host(valid)):
                    continue
                dest = self.flatten_idx(p1, p2, p3)
                val = base * interp
            else:
                dest, bc_sign = self._lookup_boundary(p1, p2, p3)
                val = base * interp
                if deposition == "wavefunction":
                    val *= bc_sign

            if deposition == "wavefunction" and self.importance:
                val *= self.trial_mos[dest]

            psi[dest] += val
            retained_weight += interp

        retained_weight_host = float(to_host(retained_weight))
        self.last_initial_info = {
            "raw_overlap_before_orientation": (
                -raw_overlap_host if phase_flipped else raw_overlap_host
            ),
            "oriented_overlap": raw_overlap_host,
            "phase_flipped_for_cp": phase_flipped,
            "initial_norm": initial_norm_host,
            "coefficient": coeff_host,
            "deposition": deposition,
            "retained_interpolation_weight": retained_weight_host,
            "theta": to_host(theta).copy(),
        }

        if (
            self.constraint_path
            and self.cp_boundary == "absorbing"
            and retained_weight_host < 1.0 - 1.0e-12
        ):
            print(
                "WARNING: the initial determinant lies within one grid cell of "
                "the CP nodal boundary; the absorbing interpolation retained "
                f"{retained_weight_host:.12f} of its stencil weight."
            )

        return psi

    def energy(self, weight):
        psi = weight / self.trial_mos if self.importance else weight
        denom = xp.dot(self.trial_mos, psi)
        psi_mb = psi @ self.mos
        num = xp.dot(psi_mb, self.gtrial)
        energy = self.ham.Lambda[-1] * (1.0 - num / denom)
        return energy, denom

    def normalize(self, weight, mode: str = "auto"):
        """Normalize without changing the stationary eigenvector."""
        mode = str(mode).lower()
        if mode == "auto":
            mode = "population" if self.constraint_path else "physical"

        if mode == "population":
            nrm = xp.sum(xp.abs(weight))
        elif mode == "physical":
            psi = weight / self.trial_mos if self.importance else weight
            psi_mb = psi @ self.mos
            nrm = xp.linalg.norm(psi_mb)
        else:
            raise ValueError(
                "normalize mode must be 'auto', 'population', or 'physical'."
            )

        if float(to_host(nrm)) < self.nodal_tol:
            raise RuntimeError("State has near-zero normalization.")
        return weight / nrm

    def diagnostics(self, step, psi, Gpsi_proj):
        E_i, denom = self.energy(psi)
        bare = psi / self.trial_mos if self.importance else psi
        bare_next = Gpsi_proj / self.trial_mos if self.importance else Gpsi_proj

        psi_mb = bare @ self.mos
        proj_mb = bare_next @ self.mos
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
        term_chunk: int = 8,
        source_chunk: int = 65536,
        corner_batch: int = 4,
        normalize_mode: str = "auto",
    ):
        for i in range(start, stop):
            Gpsi = self.mat_vec(
                psi, term_chunk=term_chunk, source_chunk=source_chunk,
                corner_batch=corner_batch
            )
            if i % print_every == 0:
                self.diagnostics(i, psi, Gpsi)
            psi = Gpsi
            if i % normalize_every == 0:
                psi = self.normalize(psi, mode=normalize_mode)
        return psi


# Example: constrained-path propagation with an absorbing nodal boundary.
#
#   me = GivensMasterEquation(
#       Ng=65,
#       importance=True,
#       constraint_path=True,
#       cp_tol=1.0e-14,
#       cp_boundary="absorbing",
#       cp_deposition="wavefunction",  # or "walker"
#   )
#   me.build(ham, trial)
#   psi = me.get_initial_state()
#   psi = me.run(
#       0, 1000, psi,
#       print_every=10,
#       normalize_every=10,
#       term_chunk=4,
#       source_chunk=32768,
#       corner_batch=4,
#       normalize_mode="auto",
#   )
#
# Set constraint_path=False to recover free projection with the projective
# boundary map.
