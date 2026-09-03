"""Deterministic Givens-grid LAFQMC for H2 / STO-3G with a UHF trial.

System assumed here
-------------------
* two spatial orbitals in an orthonormal one-particle basis,
* one alpha electron and one beta electron,
* real, collinear UHF determinants,
* every LAFQMC one-body propagator preserves spin.

The determinant manifold then factorizes into two one-electron projective
lines.  Relative to the UHF trial orbitals t_a and t_b, a walker is
parameterized by two angles

    d_a(theta_a) = cos(theta_a) t_a + sin(theta_a) q_a
    d_b(theta_b) = cos(theta_b) t_b + sin(theta_b) q_b

with theta_a, theta_b in (-pi/2, pi/2), and

    <Psi_T|D(theta_a,theta_b)> = cos(theta_a) cos(theta_b).

Hence the trial-overlap node is the boundary theta_a = +/-pi/2 or
theta_b = +/-pi/2.

The Hamiltonian object is expected to follow the SumOfRotation interface
used by the user's LAFQMC code:

    ham.nterms
    ham.a
    ham.Lambda[-1]
    ham.get_rotation_matrix(ix) -> [U_alpha, U_beta]

where either spin block can be None; None is interpreted as the identity.
The normalized decomposition convention is the same as in the H2+ grid code,
so the projected energy is

    E = Lambda * (1 - <T|Gbar|Psi>/<T|Psi>)

with Gbar = sum_i a_i U_i.

For direct branchwise constrained-path LAFQMC, use

    constraint_path=True
    cp_boundary='projective'
    cp_deposition='wavefunction'

Then the ONLY CP decision is made on the exact branch

    gbar_ij = a_i <T|U_i D_j> / <T|D_j>

before interpolation.  Branches with gbar_ij <= cp_tol are killed.  All
bilinear interpolation corners of a surviving branch are retained through
the projective boundary map, including the appropriate determinant sign.

The optional cp_boundary='absorbing' mode discards ghost interpolation
corners and is retained only for comparisons with an absorbing-boundary
discretization; it is not the pure branchwise CP rule discussed above.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from ipie.utils.backend import arraylib as xp
    from ipie.utils.backend import to_host
except ImportError:  # CPU fallback for testing outside ipie
    xp = np

    def to_host(x):
        return np.asarray(x)


def _as_orbital(v, name: str):
    """Return a real length-2 orbital vector from shape (2,) or (2,1)."""
    x = xp.asarray(v).squeeze()
    if x.ndim != 1 or int(x.size) != 2:
        raise ValueError(f"{name} must contain one orbital with length 2; got shape {x.shape}.")
    return x.copy()


def _trial_basis(t, thresh: float = 1.0e-14):
    """2x2 proper orthogonal basis B=[t,q], with q orthogonal to t."""
    t = _as_orbital(t, "trial orbital")
    nrm = xp.linalg.norm(t)
    if float(to_host(nrm)) < thresh:
        raise ValueError("Trial orbital has near-zero norm.")
    t = t / nrm
    q = xp.asarray([-t[1], t[0]], dtype=t.dtype)
    B = xp.column_stack((t, q))
    return B


def _parse_trial(trial):
    """Parse a UHF trial as (alpha,beta), each with one occupied orbital.

    Accepted examples
    -----------------
    (trial_a, trial_b), with each shape (2,) or (2,1)
    array with shape (2,2), interpreted as rows [alpha,beta]

    For PySCF UHF MO coefficients, pass explicitly

        trial = (mf.mo_coeff[0][:, 0], mf.mo_coeff[1][:, 0])

    after expressing both orbitals in the same orthonormal basis in which
    ham.get_rotation_matrix acts.
    """
    if isinstance(trial, (tuple, list)) and len(trial) == 2:
        ta = _as_orbital(trial[0], "trial alpha")
        tb = _as_orbital(trial[1], "trial beta")
        return ta, tb

    arr = xp.asarray(trial)
    if arr.shape == (2, 2):
        return arr[0].copy(), arr[1].copy()

    raise ValueError(
        "trial must be (trial_alpha, trial_beta), or an array of shape (2,2) "
        "whose rows are the occupied alpha and beta orbitals."
    )


class UHFH2GivensGridLAFQMC:
    """Two-angle deterministic LAFQMC transfer operator for H2/STO-3G UHF.

    Parameters
    ----------
    Ng
        Number of midpoint nodes along each angle.  Odd Ng puts theta=0
        (the trial determinant) exactly on the grid.
    importance
        If True, store f_j = <Psi_T|D_j> c_j rather than bare coefficients c_j.
    nodal_tol
        Numerical tolerance used only for deterministic orientation exactly at
        the trial-overlap node.
    constraint_path
        If True, kill exact branches with gbar_ij <= cp_tol.
    cp_tol
        Threshold for the branchwise constrained-path test.  Set to 0.0 for
        the literal sign test.
    cp_boundary
        'projective' or 'absorbing'.  For direct branchwise CP together with
        cp_deposition='wavefunction', use 'projective'.
    cp_deposition
        'wavefunction' interpolates the propagated determinant itself.
        'walker' directly interpolates positive importance-sampled branch
        weight and therefore requires cp_boundary='absorbing'.
    """

    def __init__(
        self,
        Ng: int = 65,
        importance: bool = True,
        nodal_tol: float = 1.0e-13,
        constraint_path: bool = False,
        cp_tol: float = 0.0,
        cp_boundary: str = "projective",
        cp_deposition: str = "wavefunction",
    ):
        self.Ng = int(Ng)
        self.importance = bool(importance)
        self.nodal_tol = float(nodal_tol)
        self.constraint_path = bool(constraint_path)
        self.cp_tol = float(cp_tol)
        self.cp_boundary = str(cp_boundary).lower()
        self.cp_deposition = str(cp_deposition).lower()

        if self.Ng < 3:
            raise ValueError("Ng must be at least 3.")
        if self.cp_tol < 0.0:
            raise ValueError("cp_tol must be nonnegative.")
        if self.cp_boundary not in {"projective", "absorbing"}:
            raise ValueError("cp_boundary must be 'projective' or 'absorbing'.")
        if self.cp_deposition not in {"wavefunction", "walker"}:
            raise ValueError("cp_deposition must be 'wavefunction' or 'walker'.")
        if self.constraint_path and not self.importance:
            raise ValueError("constraint_path=True currently requires importance=True.")
        if self.cp_deposition == "walker" and self.cp_boundary == "projective":
            raise ValueError(
                "cp_deposition='walker' uses nonnegative walker weights and cannot "
                "use signed projective interpolation.  Use cp_boundary='absorbing', "
                "or use cp_deposition='wavefunction'."
            )

        if self.Ng % 2 == 0:
            print("WARNING: Ng is even, so the UHF trial theta=(0,0) is not a grid point.")

        self.dtheta = xp.pi / self.Ng
        self.theta0 = -0.5 * xp.pi + 0.5 * self.dtheta
        self.theta_vals = self.theta0 + xp.arange(self.Ng) * self.dtheta
        self.Nh = self.Ng**2

        ia, ib = xp.indices((self.Ng, self.Ng), dtype=xp.int64)
        self.theta_a_grid = self.theta_vals[ia].reshape(-1)
        self.theta_b_grid = self.theta_vals[ib].reshape(-1)
        del ia, ib

        self.ham = None
        self.trial_a = None
        self.trial_b = None
        self.Ba = None
        self.Bb = None
        self.mos_a = None
        self.mos_b = None
        self.trial_overlap = None
        self.Ua = None
        self.Ub = None
        self.trial_GD = None
        self.local_energy = None
        self.last_cp_stats = None
        self.last_initial_info = None

    def flatten_idx(self, ia, ib):
        return ia * self.Ng + ib

    def flat2idx(self, ix):
        return ix // self.Ng, ix % self.Ng

    def _canonicalize_vectors(self, vecs, B):
        """Map real 2-vectors to theta in [-pi/2,pi/2] plus signed norms.

        For each raw vector v,

            v = signed_norm * d(theta)

        where d(theta) has nonnegative overlap with the corresponding trial
        orbital (the first column of B).
        """
        norms = xp.linalg.norm(vecs, axis=-1)
        if bool(to_host(xp.any(norms < self.nodal_tol))):
            raise ValueError("Encountered a near-zero propagated spin orbital.")

        u_orig = vecs / norms[..., None]
        u_local = xp.einsum("...p,pq->...q", u_orig, B, optimize=True)

        x0 = u_local[..., 0]
        x1 = u_local[..., 1]
        sign = xp.ones_like(x0)
        sign = xp.where(x0 < -self.nodal_tol, -sign, sign)

        # At x0=0, fix the otherwise ambiguous projective representative by
        # making x1 positive.
        on_node = xp.abs(x0) <= self.nodal_tol
        sign = xp.where(on_node & (x1 < 0.0), -sign, sign)

        u_local = u_local * sign[..., None]
        signed_norm = norms * sign
        theta = xp.arctan2(u_local[..., 1], xp.maximum(u_local[..., 0], 0.0))
        half_pi = 0.5 * xp.pi
        theta = xp.clip(theta, -half_pi, half_pi)
        return theta, signed_norm

    def _target_cell_1d(self, theta):
        """Return left midpoint-grid index and interpolation fraction.

        theta may reach the closed chart boundary +/-pi/2, so lo can be -1 or
        Ng-1.  Therefore the right corner can be 0 or Ng, i.e. one ghost layer.
        """
        q = (theta - self.theta0) / self.dtheta
        lo = xp.floor(q).astype(xp.int64)
        lo = xp.clip(lo, -1, self.Ng - 1)
        frac = q - lo
        frac = xp.clip(frac, 0.0, 1.0)
        return lo, frac

    def _lookup_projective_boundary(self, pa, pb):
        """Map one ghost layer to the principal grid and return determinant sign.

        For one spin orbital,

            d(theta +/- pi) = -d(theta).

        Thus crossing one angle face contributes -1 to the many-body Slater
        determinant; crossing both faces contributes (+1).
        """
        pa_low = pa < 0
        pa_high = pa >= self.Ng
        pb_low = pb < 0
        pb_high = pb >= self.Ng

        pa_map = xp.where(pa_low, pa + self.Ng, xp.where(pa_high, pa - self.Ng, pa))
        pb_map = xp.where(pb_low, pb + self.Ng, xp.where(pb_high, pb - self.Ng, pb))

        sa = xp.where(pa_low | pa_high, -1.0, 1.0)
        sb = xp.where(pb_low | pb_high, -1.0, 1.0)
        sign = sa * sb

        dest = self.flatten_idx(pa_map, pb_map)
        return dest.astype(xp.int64), sign

    def _read_spin_rotations(self, ham):
        """Read [U_alpha,U_beta] from ham.get_rotation_matrix and fill None by I."""
        eye = xp.eye(2)
        Ua = []
        Ub = []
        for ix in range(int(ham.nterms)):
            blocks = ham.get_rotation_matrix(ix)
            if not isinstance(blocks, (tuple, list)) or len(blocks) != 2:
                raise ValueError(
                    "ham.get_rotation_matrix(ix) must return [U_alpha,U_beta] for "
                    "this UHF implementation."
                )
            ua = eye if blocks[0] is None else xp.asarray(blocks[0])
            ub = eye if blocks[1] is None else xp.asarray(blocks[1])
            if ua.shape != (2, 2) or ub.shape != (2, 2):
                raise ValueError(
                    f"Term {ix}: expected 2x2 spin blocks, got {ua.shape} and {ub.shape}."
                )
            Ua.append(ua)
            Ub.append(ub)
        return xp.stack(Ua), xp.stack(Ub)

    def build(self, ham, trial, check: bool = True):
        """Build the 2D determinant grid and precompute trial-projected G data."""
        self.ham = ham
        ta, tb = _parse_trial(trial)

        # The code assumes the one-particle basis is orthonormal.  Normalize the
        # occupied UHF orbitals in that basis.
        ta = ta / xp.linalg.norm(ta)
        tb = tb / xp.linalg.norm(tb)
        self.trial_a = ta
        self.trial_b = tb
        self.Ba = _trial_basis(ta)
        self.Bb = _trial_basis(tb)

        # Grid spin orbitals in trial-adapted coordinates.
        ca = xp.cos(self.theta_a_grid)
        sa = xp.sin(self.theta_a_grid)
        cb = xp.cos(self.theta_b_grid)
        sb = xp.sin(self.theta_b_grid)

        local_a = xp.stack((ca, sa), axis=-1)
        local_b = xp.stack((cb, sb), axis=-1)
        self.mos_a = local_a @ self.Ba.T
        self.mos_b = local_b @ self.Bb.T

        ova = self.mos_a @ self.trial_a
        ovb = self.mos_b @ self.trial_b
        self.trial_overlap = ova * ovb

        if bool(to_host(xp.any(self.trial_overlap <= 0.0))):
            raise RuntimeError("Midpoint grid contains a non-positive trial determinant overlap.")

        acoeff = xp.asarray(ham.a)
        if self.constraint_path and bool(to_host(xp.any(acoeff < 0.0))):
            raise ValueError(
                "Constrained-path LAFQMC assumes nonnegative decomposition coefficients a_i."
            )

        self.Ua, self.Ub = self._read_spin_rotations(ham)

        # Precompute <T|Gbar|D_j> with Gbar = sum_i a_i U_i.  This lets the
        # mixed estimator use the ORIGINAL Hamiltonian even during a CP run.
        self._precompute_trial_GD()

        if check:
            err_a = xp.max(xp.abs(ova - xp.cos(self.theta_a_grid)))
            err_b = xp.max(xp.abs(ovb - xp.cos(self.theta_b_grid)))
            print(
                "UHF H2 Givens grid: "
                f"Ng={self.Ng}, Nh={self.Nh}, dtheta={float(to_host(self.dtheta)):.8e}"
            )
            print(
                "max overlap identity errors: "
                f"alpha={float(to_host(err_a)):.3e}, beta={float(to_host(err_b)):.3e}"
            )

    def _precompute_trial_GD(self, term_chunk: int = 32, source_chunk: int = 65536):
        out = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
        nterms = int(self.ham.nterms)
        acoeff = xp.asarray(self.ham.a, dtype=self.mos_a.dtype)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Ua = self.Ua[a0:a1]
            Ub = self.Ub[a0:a1]
            coeff = acoeff[a0:a1]

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                da = self.mos_a[s0:s1]
                db = self.mos_b[s0:s1]
                ta = xp.einsum("apq,sq->asp", Ua, da, optimize=True)
                tb = xp.einsum("apq,sq->asp", Ub, db, optimize=True)
                ova = xp.einsum("p,asp->as", self.trial_a, ta, optimize=True)
                ovb = xp.einsum("p,asp->as", self.trial_b, tb, optimize=True)
                out[s0:s1] += xp.einsum("a,as->s", coeff, ova * ovb, optimize=True)

        self.trial_GD = out
        Lambda = self.ham.Lambda[-1]
        self.local_energy = Lambda * (1.0 - self.trial_GD / self.trial_overlap)

    def mat_vec(
        self,
        psi,
        term_chunk: int = 8,
        source_chunk: int = 65536,
        corner_batch: int = 4,
        synchronize: bool = False,
    ):
        """Apply one deterministic LAFQMC transfer step on the theta grid.

        For CP, the exact branch

            gbar_ij = a_i <T|U_i D_j> / <T|D_j>

        is tested BEFORE bilinear interpolation.  The whole branch is rejected
        when gbar_ij <= cp_tol.
        """
        if self.ham is None:
            raise RuntimeError("Call build() before mat_vec().")
        psi = xp.asarray(psi)
        if psi.ndim != 1 or int(psi.size) != self.Nh:
            raise ValueError(f"psi must have shape ({self.Nh},), got {psi.shape}.")
        if term_chunk < 1 or source_chunk < 1:
            raise ValueError("term_chunk and source_chunk must be positive.")
        if corner_batch < 1 or corner_batch > 4:
            raise ValueError("corner_batch must be between 1 and 4.")

        out = xp.zeros(self.Nh, dtype=psi.dtype)
        nterms = int(self.ham.nterms)
        acoeff = xp.asarray(self.ham.a, dtype=psi.dtype)
        corners = tuple(itertools.product((0, 1), repeat=2))

        if self.constraint_path:
            cp_total = 0
            cp_rejected = xp.asarray(0, dtype=xp.int64)
            cp_total_flux = xp.asarray(0.0, dtype=psi.dtype)
            cp_rejected_flux = xp.asarray(0.0, dtype=psi.dtype)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Ua = self.Ua[a0:a1]
            Ub = self.Ub[a0:a1]
            coeff = acoeff[a0:a1]

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                src_a = self.mos_a[s0:s1]
                src_b = self.mos_b[s0:s1]
                src_psi = psi[s0:s1]
                old_overlap = self.trial_overlap[None, s0:s1]

                # Exact unnormalized spin-orbital targets.
                target_a = xp.einsum("apq,sq->asp", Ua, src_a, optimize=True)
                target_b = xp.einsum("apq,sq->asp", Ub, src_b, optimize=True)

                theta_a, signed_norm_a = self._canonicalize_vectors(target_a, self.Ba)
                theta_b, signed_norm_b = self._canonicalize_vectors(target_b, self.Bb)
                signed_norm = signed_norm_a * signed_norm_b

                lo_a, frac_a = self._target_cell_1d(theta_a)
                lo_b, frac_b = self._target_cell_1d(theta_b)

                if self.constraint_path:
                    new_ova = xp.einsum(
                        "p,asp->as", self.trial_a, target_a, optimize=True
                    )
                    new_ovb = xp.einsum(
                        "p,asp->as", self.trial_b, target_b, optimize=True
                    )
                    raw_new_overlap = new_ova * new_ovb
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
                        # Bare wavefunction branch coefficient:
                        #   a_i * signed_norm * c_j,
                        # with c_j = f_j / <T|D_j> under importance sampling.
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

                for icorner, (ba, bb) in enumerate(corners):
                    pa = lo_a + ba
                    pb = lo_b + bb
                    wa = frac_a if ba else (1.0 - frac_a)
                    wb = frac_b if bb else (1.0 - frac_b)
                    interp = wa * wb

                    if self.constraint_path and self.cp_boundary == "absorbing":
                        valid = (
                            (pa >= 0) & (pa < self.Ng)
                            & (pb >= 0) & (pb < self.Ng)
                        )
                        pa_safe = xp.clip(pa, 0, self.Ng - 1)
                        pb_safe = xp.clip(pb, 0, self.Ng - 1)
                        dest = self.flatten_idx(pa_safe, pb_safe)
                        vals = base * interp
                        if multiply_destination_overlap:
                            vals = vals * self.trial_overlap[dest]
                        vals = xp.where(valid, vals, xp.zeros((), dtype=psi.dtype))
                    else:
                        dest, bc_sign = self._lookup_projective_boundary(pa, pb)
                        vals = base * interp * bc_sign
                        if multiply_destination_overlap:
                            vals = vals * self.trial_overlap[dest]

                    dest_batch.append(dest.ravel())
                    vals_batch.append(vals.ravel())
                    flush = len(dest_batch) == corner_batch or icorner == len(corners) - 1
                    if flush:
                        dest_all = xp.concatenate(dest_batch)
                        vals_all = xp.concatenate(vals_batch)
                        out += xp.bincount(
                            dest_all, weights=vals_all, minlength=self.Nh
                        )
                        dest_batch.clear()
                        vals_batch.clear()

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

    def get_initial_state(
        self,
        initial=None,
        coefficient: float = 1.0,
        deposition: str = "auto",
    ):
        """Put an arbitrary real UHF determinant onto the grid by bilinear deposition.

        initial is (alpha_orbital, beta_orbital).  If None, use the UHF trial.
        The orbitals need not be normalized; their product norm is retained.
        """
        if self.ham is None:
            raise RuntimeError("Call build() before get_initial_state().")

        if initial is None:
            va = self.trial_a.copy()
            vb = self.trial_b.copy()
        else:
            va, vb = _parse_trial(initial)

        coeff = xp.asarray(coefficient, dtype=self.mos_a.dtype)
        coeff_host = float(to_host(coeff))
        if not np.isfinite(coeff_host):
            raise ValueError("coefficient must be finite.")
        if self.constraint_path and coeff_host <= 0.0:
            raise ValueError("A constrained-path initial walker must have positive coefficient.")

        deposition = str(deposition).lower()
        if deposition == "auto":
            deposition = self.cp_deposition if self.constraint_path else "wavefunction"
        if deposition not in {"wavefunction", "walker"}:
            raise ValueError("deposition must be 'auto', 'wavefunction', or 'walker'.")
        if deposition == "walker" and not (self.constraint_path and self.importance):
            raise ValueError(
                "deposition='walker' requires constraint_path=True and importance=True."
            )

        na = xp.linalg.norm(va)
        nb = xp.linalg.norm(vb)
        determinant_norm = na * nb
        if float(to_host(determinant_norm)) < self.nodal_tol:
            raise ValueError("Initial determinant has near-zero norm.")

        raw_overlap = xp.dot(self.trial_a, va) * xp.dot(self.trial_b, vb)
        raw_overlap_host = float(to_host(raw_overlap))
        phase_flipped = False

        if self.constraint_path:
            overlap_scale = max(float(to_host(determinant_norm)), 1.0)
            if abs(raw_overlap_host) <= self.cp_tol * overlap_scale:
                raise ValueError(
                    "The initial determinant is orthogonal (within cp_tol) to the UHF trial."
                )
            if raw_overlap_host < 0.0:
                # Flipping one occupied spin orbital changes only the global sign
                # of this single determinant and orients it into the positive CP sector.
                va *= -1.0
                raw_overlap *= -1.0
                raw_overlap_host *= -1.0
                phase_flipped = True

        theta_a, sn_a = self._canonicalize_vectors(va[None, :], self.Ba)
        theta_b, sn_b = self._canonicalize_vectors(vb[None, :], self.Bb)
        theta_a = theta_a[0]
        theta_b = theta_b[0]
        signed_norm = sn_a[0] * sn_b[0]

        lo_a, frac_a = self._target_cell_1d(theta_a)
        lo_b, frac_b = self._target_cell_1d(theta_b)

        psi = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
        retained_weight = xp.asarray(0.0, dtype=self.mos_a.dtype)

        if deposition == "walker":
            base = coeff * raw_overlap
        else:
            base = coeff * signed_norm

        for ba, bb in itertools.product((0, 1), repeat=2):
            pa = lo_a + ba
            pb = lo_b + bb
            wa = frac_a if ba else (1.0 - frac_a)
            wb = frac_b if bb else (1.0 - frac_b)
            interp = wa * wb

            if self.constraint_path and self.cp_boundary == "absorbing":
                valid = (
                    (pa >= 0) & (pa < self.Ng)
                    & (pb >= 0) & (pb < self.Ng)
                )
                if not bool(to_host(valid)):
                    continue
                dest = self.flatten_idx(pa, pb)
                val = base * interp
            else:
                dest, bc_sign = self._lookup_projective_boundary(pa, pb)
                val = base * interp
                if deposition == "wavefunction":
                    val *= bc_sign

            if deposition == "wavefunction" and self.importance:
                val *= self.trial_overlap[dest]

            psi[dest] += val
            retained_weight += interp

        self.last_initial_info = {
            "raw_overlap_before_orientation": (
                -raw_overlap_host if phase_flipped else raw_overlap_host
            ),
            "oriented_overlap": raw_overlap_host,
            "phase_flipped_for_cp": phase_flipped,
            "determinant_norm": float(to_host(determinant_norm)),
            "coefficient": coeff_host,
            "deposition": deposition,
            "retained_interpolation_weight": float(to_host(retained_weight)),
            "theta": np.asarray([float(to_host(theta_a)), float(to_host(theta_b))]),
        }
        return psi

    def reconstruct_many_body(self, weight):
        """Return the 2x2 alpha-beta CI coefficient matrix represented by the grid.

        C[p,q] is the coefficient of a_p^dagger b_q^dagger |0>.  Unlike H2+,
        this matrix need not have rank one, so the grid wavefunction generally
        cannot be compressed to a single Slater determinant.
        """
        weight = xp.asarray(weight)
        bare = weight / self.trial_overlap if self.importance else weight
        C = xp.einsum(
            "j,jp,jq->pq", bare, self.mos_a, self.mos_b, optimize=True
        )
        return C

    def physical_norm(self, weight):
        C = self.reconstruct_many_body(weight)
        return xp.linalg.norm(C)

    def energy(self, weight):
        """Mixed projected energy of the ORIGINAL Hamiltonian decomposition."""
        weight = xp.asarray(weight)
        bare = weight / self.trial_overlap if self.importance else weight
        denom = xp.dot(bare, self.trial_overlap)
        num = xp.dot(bare, self.trial_GD)
        E = self.ham.Lambda[-1] * (1.0 - num / denom)
        return E, denom

    def exact_G_on_many_body(self, C):
        """Apply the normalized free Gbar=sum_i a_i U_i to a 2x2 CI matrix."""
        C = xp.asarray(C)
        if C.shape != (2, 2):
            raise ValueError("C must have shape (2,2).")
        term_C = xp.einsum("aip,pq,ajq->aij", self.Ua, C, self.Ub, optimize=True)
        return xp.einsum("a,aij->ij", xp.asarray(self.ham.a), term_C, optimize=True)

    def normalize(self, weight, mode: str = "auto"):
        mode = str(mode).lower()
        if mode == "auto":
            mode = "population" if self.constraint_path else "physical"
        if mode == "population":
            nrm = xp.sum(xp.abs(weight))
        elif mode == "physical":
            nrm = self.physical_norm(weight)
        else:
            raise ValueError("mode must be 'auto', 'population', or 'physical'.")
        if float(to_host(nrm)) < self.nodal_tol:
            raise RuntimeError("State has near-zero normalization.")
        return weight / nrm

    def branch_sign_statistics(self, weight=None, term_chunk: int = 32, source_chunk: int = 65536):
        """Compute b_plus, b_minus, f, and s on every source grid determinant.

        Here gbar_ij includes a_i, consistent with mat_vec:

            b_plus[j]  = sum_{i:gbar>0} gbar_ij
            b_minus[j] = -sum_{i:gbar<0} gbar_ij
            f[j]       = b_minus/(b_plus+b_minus)
            s[j]       = (b_minus-b_plus)/(b_plus+b_minus)

        If weight is supplied, also return the flux-weighted mean f, which is
        the same quantity as rejected_flux_fraction (up to cp_tol handling).
        """
        bplus = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
        bminus = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
        acoeff = xp.asarray(self.ham.a, dtype=self.mos_a.dtype)
        nterms = int(self.ham.nterms)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Ua = self.Ua[a0:a1]
            Ub = self.Ub[a0:a1]
            coeff = acoeff[a0:a1]

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                da = self.mos_a[s0:s1]
                db = self.mos_b[s0:s1]
                old = self.trial_overlap[None, s0:s1]
                ta = xp.einsum("apq,sq->asp", Ua, da, optimize=True)
                tb = xp.einsum("apq,sq->asp", Ub, db, optimize=True)
                ova = xp.einsum("p,asp->as", self.trial_a, ta, optimize=True)
                ovb = xp.einsum("p,asp->as", self.trial_b, tb, optimize=True)
                gbar = coeff[:, None] * (ova * ovb) / old
                bplus[s0:s1] += xp.sum(xp.where(gbar > 0.0, gbar, 0.0), axis=0)
                bminus[s0:s1] += -xp.sum(xp.where(gbar < 0.0, gbar, 0.0), axis=0)

        denom = bplus + bminus
        f = xp.where(denom > 0.0, bminus / denom, 0.0)
        s = xp.where(denom > 0.0, (bminus - bplus) / denom, 0.0)

        result = {"b_plus": bplus, "b_minus": bminus, "f": f, "s": s}
        if weight is not None:
            wabs = xp.abs(xp.asarray(weight))
            flux_weight = wabs * denom
            z = xp.sum(flux_weight)
            result["flux_weighted_f"] = xp.sum(flux_weight * f) / xp.maximum(z, 1.0e-300)
        return result

    def diagnostics(self, step, psi):
        E, denom = self.energy(psi)
        nrm = self.physical_norm(psi)
        msg = (
            f"step={step}, E={float(to_host(E)):.15f}, "
            f"denom={float(to_host(denom)):.6e}, "
            f"physical_norm={float(to_host(nrm)):.6e}"
        )
        if self.last_cp_stats is not None:
            msg += (
                f", cp_reject={self.last_cp_stats['rejected_fraction']:.6e}, "
                f"cp_reject_flux={self.last_cp_stats['rejected_flux_fraction']:.6e}"
            )
        print(msg)

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
        for istep in range(start, stop):
            Gpsi = self.mat_vec(
                psi,
                term_chunk=term_chunk,
                source_chunk=source_chunk,
                corner_batch=corner_batch,
            )
            psi = Gpsi
            if normalize_every > 0 and (istep + 1) % normalize_every == 0:
                psi = self.normalize(psi, mode=normalize_mode)
            if print_every > 0 and (istep + 1) % print_every == 0:
                self.diagnostics(istep + 1, psi)
        return psi


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# # trial_a and trial_b must be expressed in the SAME ORTHONORMAL one-particle
# # basis in which ham.get_rotation_matrix(ix) acts.  For a PySCF UHF object,
# # raw AO coefficients are S-orthonormal rather than Euclidean-orthonormal, so
# # transform them consistently before using this grid code.
#
# trial = (trial_a, trial_b)
#
# me = UHFH2GivensGridLAFQMC(
#     Ng=65,
#     importance=True,
#     constraint_path=True,
#     cp_tol=0.0,
#     cp_boundary="projective",
#     cp_deposition="wavefunction",
# )
# me.build(ham, trial)
# psi = me.get_initial_state()
# psi = me.run(
#     0, 1000, psi,
#     print_every=10,
#     normalize_every=10,
#     term_chunk=8,
#     source_chunk=65536,
#     corner_batch=4,
# )
#
# E, denom = me.energy(psi)
# C = me.reconstruct_many_body(psi)  # 2x2 CI coefficient matrix
# print("E =", E)
# print("C =\n", C)
