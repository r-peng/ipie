"""Deterministic walker-grid LAFQMC for H2 / STO-3G with a UHF trial.

This version intentionally has NO wavefunction-deposition mode.

Representation
--------------
The system has two spatial orbitals, one alpha electron and one beta electron,
and real spin-preserving one-body LAFQMC propagators.  Relative to the trial
orbitals t_a and t_b, a grid determinant is

    d_a(theta_a) = cos(theta_a) t_a + sin(theta_a) q_a
    d_b(theta_b) = cos(theta_b) t_b + sin(theta_b) q_b

with theta_a, theta_b in (-pi/2, pi/2).  Every principal-grid determinant has
positive trial overlap

    O_j = <Psi_T|D_j> = cos(theta_a) cos(theta_b) > 0.

The stored array ``psi[j]`` is ALWAYS an importance-sampled walker weight f_j,
not a bare wavefunction coefficient.  If one wants the wavefunction represented
by the histogram, its grid coefficient is c_j = f_j / O_j.

Branch update
-------------
For source determinant D_j and LAFQMC branch i,

    gbar_ij = a_i <Psi_T|U_i D_j> / <Psi_T|D_j>.

The exact propagated walker weight is

    f'_ij = gbar_ij f_j.

That weight is binned onto the neighboring determinant-grid points using
nonnegative bilinear binning weights.

Free projection
---------------
Negative walker weights ARE allowed.  No branch is rejected.  If gbar_ij < 0,
the descendant walker simply has negative weight.  The angular grid is treated
projectively: a ghost bin is wrapped to the opposite chart edge.  IMPORTANT:
there is NO extra projective sign on f.  The importance weight f=<T|D>c is gauge
invariant under D -> -D, because both <T|D> and c change sign.

Constrained path
----------------
Negative walker weights are NOT allowed.  A branch is retained only if

    gbar_ij > cp_tol.

Starting from nonnegative walker weights, all retained branch weights remain
nonnegative.  The CP chart boundary is absorbing: interpolation weight assigned
to ghost bins is discarded rather than wrapped across the trial node.

Thus the two modes are deliberately:

    constraint_path=False : signed walkers + projective wrapping
    constraint_path=True  : positive walkers + branch rejection + absorbing node

There is no ``deposition='wavefunction'`` option anywhere in this file.
"""

from __future__ import annotations

import itertools
import numpy as np

try:
    from ipie.utils.backend import arraylib as xp
    from ipie.utils.backend import to_host
except ImportError:  # CPU fallback for standalone testing
    xp = np

    def to_host(x):
        return np.asarray(x)


def _as_orbital(v, name: str):
    """Return a real length-2 orbital from shape (2,) or (2,1)."""
    x = xp.asarray(v).squeeze()
    if x.ndim != 1 or int(x.size) != 2:
        raise ValueError(f"{name} must contain one orbital of length 2; got {x.shape}.")
    return x.copy()


def _trial_basis(t, thresh: float = 1.0e-14):
    """Return the proper orthogonal 2x2 basis B=[t,q]."""
    t = _as_orbital(t, "trial orbital")
    nrm = xp.linalg.norm(t)
    if float(to_host(nrm)) < thresh:
        raise ValueError("Trial orbital has near-zero norm.")
    t = t / nrm
    q = xp.asarray([-t[1], t[0]], dtype=t.dtype)
    return xp.column_stack((t, q))


def _parse_trial(trial):
    """Parse a UHF determinant as (alpha_orbital, beta_orbital)."""
    if isinstance(trial, (tuple, list)) and len(trial) == 2:
        return (
            _as_orbital(trial[0], "trial alpha"),
            _as_orbital(trial[1], "trial beta"),
        )

    arr = xp.asarray(trial)
    if arr.shape == (2, 2):
        return arr[0].copy(), arr[1].copy()

    raise ValueError(
        "trial must be (trial_alpha, trial_beta), or an array of shape (2,2)."
    )


class UHFH2GivensGridLAFQMC:
    """Two-angle deterministic WALKER-grid LAFQMC for H2/STO-3G.

    Parameters
    ----------
    Ng
        Number of midpoint nodes along each Givens angle.  Odd Ng puts the
        trial determinant theta=(0,0) exactly on the grid.
    constraint_path
        False: free signed-walker propagation.
        True: constrained-path positive-walker propagation.
    cp_tol
        CP branch threshold.  In CP mode retain only gbar_ij > cp_tol.
        The literal sign constraint is cp_tol=0.
    nodal_tol
        Numerical tolerance for detecting a zero propagated orbital and an
        initial determinant on the trial-overlap node.

    Notes
    -----
    The walker representation requires importance sampling and therefore this
    class fixes that convention internally.  ``psi`` always stores f_j, not c_j.
    """

    def __init__(
        self,
        Ng: int = 65,
        constraint_path: bool = False,
        cp_tol: float = 0.0,
        nodal_tol: float = 1.0e-13,
    ):
        self.Ng = int(Ng)
        self.constraint_path = bool(constraint_path)
        self.cp_tol = float(cp_tol)
        self.nodal_tol = float(nodal_tol)
        self.importance = True  # fixed by construction

        if self.Ng < 3:
            raise ValueError("Ng must be at least 3.")
        if self.cp_tol < 0.0:
            raise ValueError("cp_tol must be nonnegative.")
        if self.Ng % 2 == 0:
            print("WARNING: Ng is even, so theta=(0,0) is not a grid point.")

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
        self.last_free_stats = None
        self.last_initial_info = None

    def flatten_idx(self, ia, ib):
        return ia * self.Ng + ib

    def flat2idx(self, ix):
        return ix // self.Ng, ix % self.Ng

    def _canonicalize_vectors(self, vecs, B):
        """Map real 2-vectors to the positive-overlap projective chart.

        Returns
        -------
        theta
            Canonical angle in [-pi/2, pi/2].
        signed_norm
            Scalar s*||v|| such that raw v = signed_norm*d(theta).
            It is retained for diagnostics; walker propagation itself uses gbar
            and does not need this scalar.
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

        on_node = xp.abs(x0) <= self.nodal_tol
        sign = xp.where(on_node & (x1 < 0.0), -sign, sign)

        u_local = u_local * sign[..., None]
        signed_norm = norms * sign
        theta = xp.arctan2(u_local[..., 1], xp.maximum(u_local[..., 0], 0.0))
        theta = xp.clip(theta, -0.5 * xp.pi, 0.5 * xp.pi)
        return theta, signed_norm

    def _target_cell_1d(self, theta):
        """Return lower midpoint-grid index and linear binning fraction."""
        q = (theta - self.theta0) / self.dtheta
        lo = xp.floor(q).astype(xp.int64)
        lo = xp.clip(lo, -1, self.Ng - 1)
        frac = xp.clip(q - lo, 0.0, 1.0)
        return lo, frac

    def _wrap_projective_boundary(self, pa, pb):
        """Wrap one ghost layer onto the principal projective grid.

        No determinant/projective sign is returned here.  For walker deposition
        the stored importance weight f=<T|D>c is gauge invariant under D -> -D,
        so wrapping a projective representative does NOT change f.
        """
        pa_map = xp.where(pa < 0, pa + self.Ng, xp.where(pa >= self.Ng, pa - self.Ng, pa))
        pb_map = xp.where(pb < 0, pb + self.Ng, xp.where(pb >= self.Ng, pb - self.Ng, pb))
        return self.flatten_idx(pa_map, pb_map).astype(xp.int64)

    def _read_spin_rotations(self, ham):
        """Read [U_alpha,U_beta] from ham and replace None by identity."""
        eye = xp.eye(2)
        Ua = []
        Ub = []
        for ix in range(int(ham.nterms)):
            blocks = ham.get_rotation_matrix(ix)
            if not isinstance(blocks, (tuple, list)) or len(blocks) != 2:
                raise ValueError(
                    "ham.get_rotation_matrix(ix) must return [U_alpha,U_beta]."
                )
            ua = eye if blocks[0] is None else xp.asarray(blocks[0])
            ub = eye if blocks[1] is None else xp.asarray(blocks[1])
            if ua.shape != (2, 2) or ub.shape != (2, 2):
                raise ValueError(
                    f"Term {ix}: expected 2x2 spin blocks, got {ua.shape}, {ub.shape}."
                )
            Ua.append(ua)
            Ub.append(ub)
        return xp.stack(Ua), xp.stack(Ub)

    def build(self, ham, trial, check: bool = True):
        """Build the determinant grid and precompute mixed-estimator data."""
        self.ham = ham
        ta, tb = _parse_trial(trial)
        ta = ta / xp.linalg.norm(ta)
        tb = tb / xp.linalg.norm(tb)

        self.trial_a = ta
        self.trial_b = tb
        self.Ba = _trial_basis(ta)
        self.Bb = _trial_basis(tb)

        ca = xp.cos(self.theta_a_grid)
        sa = xp.sin(self.theta_a_grid)
        cb = xp.cos(self.theta_b_grid)
        sb = xp.sin(self.theta_b_grid)
        self.mos_a = xp.stack((ca, sa), axis=-1) @ self.Ba.T
        self.mos_b = xp.stack((cb, sb), axis=-1) @ self.Bb.T

        ova = self.mos_a @ self.trial_a
        ovb = self.mos_b @ self.trial_b
        self.trial_overlap = ova * ovb

        if bool(to_host(xp.any(self.trial_overlap <= 0.0))):
            raise RuntimeError("Principal midpoint grid contains non-positive trial overlap.")

        acoeff = xp.asarray(ham.a)
        if self.constraint_path and bool(to_host(xp.any(acoeff < 0.0))):
            raise ValueError(
                "CP walker propagation assumes nonnegative decomposition coefficients a_i."
            )

        self.Ua, self.Ub = self._read_spin_rotations(ham)
        self._precompute_trial_GD()

        if check:
            err_a = xp.max(xp.abs(ova - xp.cos(self.theta_a_grid)))
            err_b = xp.max(xp.abs(ovb - xp.cos(self.theta_b_grid)))
            mode = "CP positive walkers / absorbing boundary" if self.constraint_path else "free signed walkers / projective boundary"
            print(
                f"UHF H2 Givens WALKER grid: Ng={self.Ng}, Nh={self.Nh}, "
                f"dtheta={float(to_host(self.dtheta)):.8e}"
            )
            print(f"mode: {mode}")
            print(
                "max overlap identity errors: "
                f"alpha={float(to_host(err_a)):.3e}, "
                f"beta={float(to_host(err_b)):.3e}"
            )

    def _precompute_trial_GD(self, term_chunk: int = 32, source_chunk: int = 65536):
        """Precompute <T|Gbar|D_j> and local energies on grid determinants."""
        out = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
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
        """Apply one walker-grid LAFQMC step.

        Free mode
        ---------
        Every branch is kept and

            f' = gbar * f

        may have either sign.  Ghost interpolation bins wrap projectively with
        NO extra sign on f.

        CP mode
        -------
        Keep only gbar > cp_tol.  With nonnegative input weights all output
        weights remain nonnegative.  Ghost interpolation bins are absorbed.
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

        if self.constraint_path:
            min_psi = float(to_host(xp.min(psi)))
            if min_psi < -100.0 * np.finfo(float).eps:
                raise ValueError(
                    f"CP input contains a negative walker weight ({min_psi:.3e})."
                )

        out = xp.zeros(self.Nh, dtype=psi.dtype)
        acoeff = xp.asarray(self.ham.a, dtype=psi.dtype)
        nterms = int(self.ham.nterms)
        corners = tuple(itertools.product((0, 1), repeat=2))

        if self.constraint_path:
            cp_total = 0
            cp_rejected = xp.asarray(0, dtype=xp.int64)
            cp_total_flux = xp.asarray(0.0, dtype=psi.dtype)
            cp_rejected_flux = xp.asarray(0.0, dtype=psi.dtype)
        else:
            free_total_abs_flux = xp.asarray(0.0, dtype=psi.dtype)
            free_negative_abs_flux = xp.asarray(0.0, dtype=psi.dtype)

        for a0 in range(0, nterms, term_chunk):
            a1 = min(a0 + term_chunk, nterms)
            Ua = self.Ua[a0:a1]
            Ub = self.Ub[a0:a1]
            coeff = acoeff[a0:a1]

            for s0 in range(0, self.Nh, source_chunk):
                s1 = min(s0 + source_chunk, self.Nh)
                src_a = self.mos_a[s0:s1]
                src_b = self.mos_b[s0:s1]
                src_f = psi[s0:s1]
                old_overlap = self.trial_overlap[None, s0:s1]

                target_a = xp.einsum("apq,sq->asp", Ua, src_a, optimize=True)
                target_b = xp.einsum("apq,sq->asp", Ub, src_b, optimize=True)

                # Canonical target location on the positive-overlap projective chart.
                theta_a, _ = self._canonicalize_vectors(target_a, self.Ba)
                theta_b, _ = self._canonicalize_vectors(target_b, self.Bb)
                lo_a, frac_a = self._target_cell_1d(theta_a)
                lo_b, frac_b = self._target_cell_1d(theta_b)

                # Exact importance-sampled branch factor, evaluated BEFORE
                # canonicalization/deposition.
                new_ova = xp.einsum(
                    "p,asp->as", self.trial_a, target_a, optimize=True
                )
                new_ovb = xp.einsum(
                    "p,asp->as", self.trial_b, target_b, optimize=True
                )
                raw_new_overlap = new_ova * new_ovb
                gbar = coeff[:, None] * raw_new_overlap / old_overlap

                branch_f = gbar * src_f[None, :]

                if self.constraint_path:
                    allowed = gbar > self.cp_tol
                    branch_flux = xp.abs(branch_f)
                    cp_total += int(allowed.size)
                    cp_rejected += xp.count_nonzero(~allowed)
                    cp_total_flux += xp.sum(branch_flux)
                    cp_rejected_flux += xp.sum(
                        xp.where(allowed, xp.zeros((), dtype=psi.dtype), branch_flux)
                    )
                    base = xp.where(
                        allowed, branch_f, xp.zeros((), dtype=psi.dtype)
                    )
                else:
                    base = branch_f
                    abs_flux = xp.abs(branch_f)
                    free_total_abs_flux += xp.sum(abs_flux)
                    free_negative_abs_flux += xp.sum(
                        xp.where(branch_f < 0.0, abs_flux, xp.zeros((), dtype=psi.dtype))
                    )

                dest_batch = []
                vals_batch = []

                for icorner, (ba, bb) in enumerate(corners):
                    pa = lo_a + ba
                    pb = lo_b + bb
                    wa = frac_a if ba else (1.0 - frac_a)
                    wb = frac_b if bb else (1.0 - frac_b)
                    interp = wa * wb

                    if self.constraint_path:
                        # Absorbing CP node: discard any ghost-bin interpolation
                        # weight.  No negative walker weights are generated.
                        valid = (
                            (pa >= 0) & (pa < self.Ng)
                            & (pb >= 0) & (pb < self.Ng)
                        )
                        pa_safe = xp.clip(pa, 0, self.Ng - 1)
                        pb_safe = xp.clip(pb, 0, self.Ng - 1)
                        dest = self.flatten_idx(pa_safe, pb_safe)
                        vals = xp.where(
                            valid,
                            base * interp,
                            xp.zeros((), dtype=psi.dtype),
                        )
                    else:
                        # Free walkers live on the projective determinant manifold.
                        # Wrap ghost bins, but DO NOT multiply by a projective sign:
                        # f=<T|D>c is invariant under D -> -D.
                        dest = self._wrap_projective_boundary(pa, pb)
                        vals = base * interp

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

        if self.constraint_path:
            # Every arithmetic contribution is nonnegative in exact arithmetic.
            # Keep a strict diagnostic rather than silently clipping real errors.
            min_out = float(to_host(xp.min(out)))
            if min_out < -100.0 * np.finfo(float).eps:
                raise RuntimeError(
                    f"CP propagation generated a negative grid walker weight {min_out:.3e}."
                )

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
            self.last_free_stats = None
        else:
            total_flux = float(to_host(free_total_abs_flux))
            negative_flux = float(to_host(free_negative_abs_flux))
            self.last_free_stats = {
                "total_abs_flux": total_flux,
                "negative_abs_flux": negative_flux,
                "negative_flux_fraction": negative_flux / max(total_flux, 1.0e-300),
            }
            self.last_cp_stats = None

        if synchronize and hasattr(xp, "cuda"):
            xp.cuda.Stream.null.synchronize()
        return out

    def get_initial_state(self, initial=None, coefficient: float = 1.0):
        """Deposit one initial determinant as WALKER weight only.

        Free mode allows a signed importance weight.  CP mode orients the input
        determinant into the positive trial-overlap sector and requires a
        positive coefficient.
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
            raise ValueError("CP initial walker coefficient must be positive.")

        na = xp.linalg.norm(va)
        nb = xp.linalg.norm(vb)
        determinant_norm = na * nb
        if float(to_host(determinant_norm)) < self.nodal_tol:
            raise ValueError("Initial determinant has near-zero norm.")

        raw_overlap = xp.dot(self.trial_a, va) * xp.dot(self.trial_b, vb)
        raw_overlap_host_before = float(to_host(raw_overlap))
        phase_flipped = False

        if self.constraint_path:
            overlap_scale = max(float(to_host(determinant_norm)), 1.0)
            if abs(raw_overlap_host_before) <= self.cp_tol * overlap_scale:
                raise ValueError(
                    "Initial determinant is on the trial-overlap node within cp_tol."
                )
            if raw_overlap_host_before < 0.0:
                # Global determinant phase is arbitrary.  Orient the initial
                # walker into the positive CP sector.
                va *= -1.0
                raw_overlap *= -1.0
                phase_flipped = True

        raw_overlap_host = float(to_host(raw_overlap))

        theta_a, _ = self._canonicalize_vectors(va[None, :], self.Ba)
        theta_b, _ = self._canonicalize_vectors(vb[None, :], self.Bb)
        theta_a = theta_a[0]
        theta_b = theta_b[0]
        lo_a, frac_a = self._target_cell_1d(theta_a)
        lo_b, frac_b = self._target_cell_1d(theta_b)

        # For a determinant with physical coefficient ``coefficient``, the
        # stored importance walker weight is f=<T|D>*coefficient.
        base = coeff * raw_overlap
        psi = xp.zeros(self.Nh, dtype=self.mos_a.dtype)
        retained = xp.asarray(0.0, dtype=self.mos_a.dtype)

        for ba, bb in itertools.product((0, 1), repeat=2):
            pa = lo_a + ba
            pb = lo_b + bb
            wa = frac_a if ba else (1.0 - frac_a)
            wb = frac_b if bb else (1.0 - frac_b)
            interp = wa * wb

            if self.constraint_path:
                valid = (
                    (pa >= 0) & (pa < self.Ng)
                    & (pb >= 0) & (pb < self.Ng)
                )
                if not bool(to_host(valid)):
                    continue
                dest = self.flatten_idx(pa, pb)
                val = base * interp
            else:
                dest = self._wrap_projective_boundary(pa, pb)
                val = base * interp

            psi[dest] += val
            retained += interp

        if self.constraint_path:
            min_psi = float(to_host(xp.min(psi)))
            if min_psi < -100.0 * np.finfo(float).eps:
                raise RuntimeError("CP initial deposition produced a negative walker weight.")

        self.last_initial_info = {
            "raw_overlap_before_orientation": raw_overlap_host_before,
            "oriented_overlap": raw_overlap_host,
            "phase_flipped_for_cp": phase_flipped,
            "determinant_norm": float(to_host(determinant_norm)),
            "coefficient": coeff_host,
            "retained_binning_weight": float(to_host(retained)),
            "theta": np.asarray([float(to_host(theta_a)), float(to_host(theta_b))]),
            "mode": "cp_positive" if self.constraint_path else "free_signed",
        }
        return psi

    def reconstruct_many_body(self, weight):
        """Reconstruct the approximate 2x2 CI coefficient matrix from walkers."""
        weight = xp.asarray(weight)
        bare = weight / self.trial_overlap
        return xp.einsum(
            "j,jp,jq->pq", bare, self.mos_a, self.mos_b, optimize=True
        )

    def physical_norm(self, weight):
        return xp.linalg.norm(self.reconstruct_many_body(weight))

    def energy(self, weight):
        """Mixed energy of the original Hamiltonian using walker weights.

        Since f_j=<T|D_j>c_j,

            E = sum_j f_j E_L(D_j) / sum_j f_j.

        For free signed walkers the denominator may cross zero, in which case the
        mixed estimator has the expected pole.
        """
        weight = xp.asarray(weight)
        denom = xp.sum(weight)
        num = xp.dot(weight, self.local_energy)
        return num / denom, denom

    def exact_G_on_many_body(self, C):
        """Apply the exact normalized free Gbar=sum_i a_i U_i to a 2x2 CI matrix."""
        C = xp.asarray(C)
        if C.shape != (2, 2):
            raise ValueError("C must have shape (2,2).")
        term_C = xp.einsum("aip,pq,ajq->aij", self.Ua, C, self.Ub, optimize=True)
        return xp.einsum("a,aij->ij", xp.asarray(self.ham.a), term_C, optimize=True)

    def normalize(self, weight, mode: str = "population"):
        """Normalize by walker population (default) or reconstructed physical norm."""
        mode = str(mode).lower()
        if mode == "population":
            # For CP this is simply sum(weight); for free it is the signed-walker
            # L1 population, analogous to normalizing total absolute walker weight.
            nrm = xp.sum(xp.abs(weight))
        elif mode == "physical":
            nrm = self.physical_norm(weight)
        else:
            raise ValueError("mode must be 'population' or 'physical'.")
        if float(to_host(nrm)) < self.nodal_tol:
            raise RuntimeError("State has near-zero normalization.")
        return weight / nrm

    def branch_sign_statistics(
        self, weight=None, term_chunk: int = 32, source_chunk: int = 65536
    ):
        """Compute positive/negative branch factors on each source determinant."""
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
        frac_minus = xp.where(denom > 0.0, bminus / denom, 0.0)
        result = {"b_plus": bplus, "b_minus": bminus, "f_minus": frac_minus}

        if weight is not None:
            wabs = xp.abs(xp.asarray(weight))
            flux_weight = wabs * denom
            z = xp.sum(flux_weight)
            result["flux_weighted_f_minus"] = (
                xp.sum(flux_weight * frac_minus) / xp.maximum(z, 1.0e-300)
            )
        return result

    def diagnostics(self, step, psi):
        E, denom = self.energy(psi)
        nrm = self.physical_norm(psi)
        pop_abs = xp.sum(xp.abs(psi))
        pop_signed = xp.sum(psi)

        msg = (
            f"step={step}, E={float(to_host(E)):.15f}, "
            f"denom={float(to_host(denom)):.6e}, "
            f"physical_norm={float(to_host(nrm)):.6e}, "
            f"pop_abs={float(to_host(pop_abs)):.6e}, "
            f"pop_signed={float(to_host(pop_signed)):.6e}"
        )

        if self.constraint_path and self.last_cp_stats is not None:
            msg += (
                f", cp_reject={self.last_cp_stats['rejected_fraction']:.6e}, "
                f"cp_reject_flux={self.last_cp_stats['rejected_flux_fraction']:.6e}"
            )
        elif (not self.constraint_path) and self.last_free_stats is not None:
            msg += (
                ", negative_flux="
                f"{self.last_free_stats['negative_flux_fraction']:.6e}"
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
        normalize_mode: str = "population",
    ):
        for istep in range(start, stop):
            psi = self.mat_vec(
                psi,
                term_chunk=term_chunk,
                source_chunk=source_chunk,
                corner_batch=corner_batch,
            )
            if normalize_every > 0 and (istep + 1) % normalize_every == 0:
                psi = self.normalize(psi, mode=normalize_mode)
            if print_every > 0 and (istep + 1) % print_every == 0:
                self.diagnostics(istep + 1, psi)
        return psi


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
# Free projection: signed walker weights, no rejection, projective grid wrapping.
#
# me = UHFH2GivensGridLAFQMC(
#     Ng=405,
#     constraint_path=False,
# )
# me.build(ham, trial)
# psi = me.get_initial_state(initial)
# psi = me.run(0, 10000, psi, term_chunk=ham.nterms, source_chunk=me.Nh)
#
# Constrained path: positive walker weights, reject gbar<=0, absorbing node.
#
# me = UHFH2GivensGridLAFQMC(
#     Ng=405,
#     constraint_path=True,
#     cp_tol=0.0,
# )
# me.build(ham, trial)
# psi = me.get_initial_state(initial)
# psi = me.run(0, 10000, psi, term_chunk=ham.nterms, source_chunk=me.Nh)
