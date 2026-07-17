import numpy as np
import h5py, time
from ipie.hamiltonians.sor_base import QCSOR
from ipie.utils.linalg import modified_cholesky

np.set_printoptions(suppress=True, precision=6)

class MasterEquation:
    def __init__(self,Nangle,Nr):
        self.Nr, self.Ntheta, self.Nphi = Nr+1, Nangle * 4, Nangle * 2 + 1
        self.dr, self.dtheta, self.dphi = 1.0 / (self.Nr - 1), 2 * np.pi / self.Ntheta, np.pi / (self.Nphi - 1)
        rs = np.arange(self.Nr) * self.dr
        thetas = np.arange(self.Ntheta) * self.dtheta
        phis = np.arange(self.Nphi) * self.dphi

        print(rs)
        print(thetas)
        print(phis)
        
        print("computing all mos...")
        t0 = time.time()
        ct = np.cos(thetas)
        st = np.sin(thetas)
        sp = np.sin(phis)
        cp = np.cos(phis)
        
        mos_grid = np.empty((self.Nr, self.Ntheta, self.Nphi, 4), dtype=np.float64)
        mos_grid[..., 0] = rs[:, None, None] * ct[None, :, None] * sp[None, None, :]
        mos_grid[..., 1] = rs[:, None, None] * st[None, :, None] * sp[None, None, :]
        mos_grid[..., 2] = rs[:, None, None] * cp[None, None, :]
        mos_grid[..., 3] = np.sqrt(np.maximum(0.0, 1.0 - rs**2))[:, None, None]
        self.Nh = self.Nr * self.Ntheta * self.Nphi
        self.mos = mos_grid.reshape(self.Nh, 4)
        print("computing mo time=", time.time() - t0)

    def build(self, ham, trial, check=False, tol=1e-8):
        self.ham = ham
        self.trial = trial

        print("precomputing Umos...")
        t0 = time.time()
        self.Us = np.asarray([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
        self.Umos = np.einsum("axy,iy->aix", self.Us, self.mos, optimize=True)
        print("computing Umo time=", time.time() - t0)

        #ix_path = f"integrals/R{R:.1f}_{xc}_Umos_ixs_{Nr}_{N}.npy"
        #fac_path = f"integrals/R{R:.1f}_{xc}_Umos_fac_{Nr}_{N}.npy"
        #try:
        #    self.Umos_ixs = np.load(ix_path)
        #    self.Umos_fac = np.load(fac_path)
        #except FileNotFoundError:
        #    print("precomputing index and fac maps...")
        #    t0 = time.time()
        #    self.Umos_ixs, self.Umos_fac = self.mo2ix_vec(Umos)
        #    #np.save(ix_path, Umos_ixs)
        #    #np.save(fac_path, Umos_fac)
        #    print("computing idx map time=", time.time() - t0)
        print("precomputing index and fac maps...")
        t0 = time.time()
        self.Umos_ixs, self.Umos_fac = self.mo2ix_vec(self.Umos)
        print("computing idx map time=", time.time() - t0)

        self.check_projection_error(tol=tol)

        # Precompute the weighted map. This makes each matvec a single bincount.
        self.map_ix = self.Umos_ixs.ravel()
        self.map_fac = (np.asarray(ham.a)[:, None] * self.Umos_fac).ravel()
        
        self.trial_mos = self.mos @ trial
        self.trial_Umos = np.einsum('a,x,aix->i',self.ham.a,trial,self.Umos)
        self.gtrial = np.einsum("a,x,axy->y", np.asarray(self.ham.a), self.trial, self.Us, optimize=True)
        trial_Umos_check = self.mos @ self.gtrial
        err = np.linalg.norm(trial_Umos_check - self.trial_Umos)
        assert err<tol
        print('trial_Umos check=',err)

        if not check:
            return

        t0 = time.time()
        psi_test = np.random.rand(self.Nh) * 2 - 1
        G1 = self.mat_vec(psi_test)
        G2 = self.mat_vec_chunked(psi_test, chunk_terms=32)
        G3 = self.mat_vec_slow(psi_test)
        err1 = np.linalg.norm(G1 - G3)
        err2 = np.linalg.norm(G2 - G3)
        assert err1<tol
        assert err2<tol
        print("check matvec=", err1, np.linalg.norm(G1))
        print("check matvec=", err2, np.linalg.norm(G2))
        print("check vector time=", time.time() - t0)
        raise SystemExit

    def flatten_idx(self, ir, itheta, iphi):
        return (ir * self.Ntheta + itheta) * self.Nphi + iphi

    def flat2idx(self, ix):
        ir = ix // (self.Ntheta * self.Nphi)
        rem = ix % (self.Ntheta * self.Nphi)
        itheta = rem // self.Nphi
        iphi = rem % self.Nphi
        return ir, itheta, iphi

    def mo2ix(self, mo, thresh=1e-8):
        """Scalar version, kept for one-off use and checking."""
        mo = np.array(mo, dtype=np.float64, copy=True)
        norm = np.linalg.norm(mo)
        mo /= norm
        if mo[3] < -thresh:
            norm *= -1.0
            mo *= -1.0
    
        r = np.sqrt(max(0.0, 1.0 - mo[3] ** 2))
        if r < thresh:
            return 0, norm
    
        ir = int(np.rint(r / self.dr) + 1e-6)
        mo /= r
    
        if mo[2] < -1 + thresh:
            return self.flatten_idx(ir, 0, self.Nphi - 1), norm
        if mo[2] > 1 - thresh:
            return self.flatten_idx(ir, 0, 0), norm
    
        iphi = int(np.rint(np.arccos(np.clip(mo[2], -1.0, 1.0)) / self.dphi) + 1e-6)
        theta = np.mod(np.arctan2(mo[1], mo[0]), 2 * np.pi)
        itheta = int(np.rint(theta / self.dtheta) + 1e-6) % self.Ntheta
        return self.flatten_idx(ir, itheta, iphi), norm

    def mo2ix_vec(self, vecs, thresh=1e-8):
        """Vectorized version of mo2ix for shape (..., 4)."""
        orig_shape = vecs.shape[:-1]
        v = vecs.reshape(-1, 4).astype(np.float64, copy=True)
    
        norms = np.linalg.norm(v, axis=1)
        if np.any(norms < thresh):
            raise ValueError("Encountered a near-zero MO vector; cannot map to grid.")
    
        u = v / norms[:, None]
        flip = u[:, 3] < -thresh
        fac = norms.copy()
        fac[flip] *= -1.0
        u[flip] *= -1.0
    
        r = np.sqrt(np.maximum(0.0, 1.0 - u[:, 3] ** 2))
        out_ix = np.zeros(u.shape[0], dtype=np.int64)
    
        active = r >= thresh
        if np.any(active):
            idx = np.nonzero(active)[0]
            rr = r[idx]
            xyz = u[idx, :3] / rr[:, None]
    
            ir = np.rint(rr / self.dr).astype(np.int64)
            ir = np.clip(ir, 0, self.Nr - 1)
    
            z = np.clip(xyz[:, 2], -1.0, 1.0)
            iphi = np.rint(np.arccos(z) / self.dphi).astype(np.int64)
            iphi = np.clip(iphi, 0, self.Nphi - 1)
    
            theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)
            itheta = np.rint(theta / self.dtheta).astype(np.int64) % self.Ntheta
    
            # At the polar caps theta is irrelevant; match the scalar routine.
            itheta[z < -1.0 + thresh] = 0
            itheta[z > 1.0 - thresh] = 0
    
            out_ix[idx] = self.flatten_idx(ir, itheta, iphi)
    
        return out_ix.reshape(orig_shape), fac.reshape(orig_shape)

    def mat_vec(self, psi):
        # Equivalent to:
        # psi_new[Umos_ixs[ix1, ix2]] += ham.a[ix1] * psi[ix2] * Umos_fac[ix1, ix2]
        vals = self.map_fac * np.broadcast_to(psi, self.Umos_fac.shape).ravel()
        return np.bincount(self.map_ix, weights=vals, minlength=self.Nh)
    
    def mat_vec_chunked(self, psi, chunk_terms=512):
        """Lower-memory variant if map_fac is too large for your production run."""
        out = np.zeros(self.Nh, dtype=np.result_type(psi, self.Umos_fac))
        for start in range(0, self.ham.nterms, chunk_terms):
            stop = min(start + chunk_terms, self.ham.nterms)
            vals = (np.asarray(self.ham.a)[start:stop, None] * self.Umos_fac[start:stop]) * psi[None, :]
            out += np.bincount(self.Umos_ixs[start:stop].ravel(), weights=vals.ravel(), minlength=self.Nh)
        return out
    
    def mat_vec_slow(self, psi):
        psi_new = np.zeros(self.Nh)
        for ix1,ai in enumerate(self.ham.a):
            for ix2,moi in enumerate(self.mos):
                #print(ix1,ix2)
                mo_new = np.dot(self.Us[ix1],moi)
                ix,fac = self.mo2ix(mo_new)
                psi_new[ix] += fac*psi[ix2]*ai
        return psi_new

    def check_projection_error(self, tol=1e-8, chunk_terms=256):
        """
        Fast vectorized diagnostic for the mo2ix/mo2ix_vec grid projection.
    
        Checks whether:
            Umos[a, i, :] ~= Umos_fac[a, i] * mos[Umos_ixs[a, i], :]
    
        Returns
        -------
        max_err : float
            Maximum absolute Euclidean projection error.
        rms_err : float
            RMS Euclidean projection error.
        bad_count : int
            Number of projected points with error > tol.
        """
    
        nterms, Nh, norb = self.Umos.shape
    
        max_err = 0.0
        sum_err2 = 0.0
        bad_count = 0
        worst = None
    
        for a0 in range(0, nterms, chunk_terms):
            a1 = min(a0 + chunk_terms, nterms)
    
            ix_chunk = self.Umos_ixs[a0:a1]          # shape (chunk, Nh)
            fac_chunk = self.Umos_fac[a0:a1]         # shape (chunk, Nh)
    
            # mos[ix_chunk] has shape (chunk, Nh, norb)
            reconstructed = self.mos[ix_chunk] * fac_chunk[..., None]
    
            diff = reconstructed - self.Umos[a0:a1]
            err = np.linalg.norm(diff, axis=-1) # shape (chunk, Nh)
    
            local_max = np.max(err)
            if local_max > max_err:
                local_arg = np.unravel_index(np.argmax(err), err.shape)
                worst = (a0 + local_arg[0], local_arg[1], local_max)
                max_err = local_max
    
            sum_err2 += np.sum(err * err)
            bad_count += np.count_nonzero(err > tol)
    
        rms_err = np.sqrt(sum_err2 / (nterms * Nh))
    
        print("projection check:")
        print("  max_err  =", max_err)
        print("  rms_err  =", rms_err)
        print("  bad_count=", bad_count, "out of", nterms * Nh)
    
        if worst is not None:
            a, i, e = worst
            print("  worst term/grid =", a, i)
            print("  worst flat2idx   =", self.flat2idx(i))
            print("  mapped flat2idx  =", self.flat2idx(self.Umos_ixs[a, i]))
            print("  original Umos    =", self.Umos[a, i])
            print("  reconstructed   =", self.Umos_fac[a, i] * self.mos[self.Umos_ixs[a, i]])
    
        if bad_count > 0:
            print(f"WARNING: found {bad_count} projection errors larger than {tol:g}")
    
        return max_err, rms_err, bad_count
    
    def projection_effect_on_energy(self, psi, Gpsi=None, chunk_terms=256, eps=1e-14):
        """
        Diagnose how much grid projection affects the next printed energy.
    
        Your actual printed energy is
    
            E(psi) = Lambda[-1] * (1 - <T|G|psi> / <T|psi>)
    
        where <T|G|psi> is computed using self.trial_Umos @ psi.
    
        The projection error does NOT enter this numerator directly at the same step.
        It enters because the next state is projected:
    
            psi_next_proj = P G psi
    
        instead of the unprojected off-grid state
    
            psi_next_exact = G psi.
    
        This function compares
    
            E_next_exact = E(G psi)
            E_next_proj  = E(P G psi)
    
        without explicitly storing the off-grid state.
        """
    
        if Gpsi is None:
            Gpsi = self.mat_vec(psi)
    
        Lambda = self.ham.Lambda[-1]
    
        # Current denominator and numerator.
        den_curr = np.dot(self.trial_mos, psi)
        num_curr = np.dot(self.trial_Umos, psi)
        E_curr = Lambda * (1.0 - num_curr / den_curr) if abs(den_curr) > eps else np.nan
    
        # Projected next-state quantities.
        den_next_proj = np.dot(self.trial_mos, Gpsi)
        num_next_proj = np.dot(self.trial_Umos, Gpsi)
        E_next_proj = Lambda * (1.0 - num_next_proj / den_next_proj) if abs(den_next_proj) > eps else np.nan
    
        # Exact unprojected next denominator:
        #
        #   <T | G psi> = num_curr
        #
        den_next_exact = num_curr
    
        # Exact unprojected next numerator:
        #
        #   <T | G | G psi>
        #
        # The unprojected G psi is sum_{a,i} ham.a[a] psi[i] |U_a m_i>.
        # For any orbital v, <T|G|v> = self.gtrial @ v.
        #
        # Therefore:
        #
        #   num_next_exact =
        #       sum_{a,i} ham.a[a] psi[i] * (self.gtrial @ Umos[a,i])
        #
        num_next_exact = 0.0
    
        # Projection geometry diagnostics.
        weighted_resid2 = 0.0
        weighted_exact2 = 0.0
        max_pointwise_resid = 0.0
        max_weighted_resid = 0.0
        worst = None
    
        ham_a = np.asarray(self.ham.a)
    
        for a0 in range(0, self.ham.nterms, chunk_terms):
            a1 = min(a0 + chunk_terms, self.ham.nterms)
    
            U_chunk = self.Umos[a0:a1]          # shape (chunk, Nh, 4)
            ix_chunk = self.Umos_ixs[a0:a1]     # shape (chunk, Nh)
            fac_chunk = self.Umos_fac[a0:a1]    # shape (chunk, Nh)
            a_chunk = ham_a[a0:a1]              # shape (chunk,)
    
            # <T|G| U_a m_i>
            exact_next_overlap = np.einsum("x,aix->ai", self.gtrial, U_chunk, optimize=True)
    
            coeff = a_chunk[:, None] * psi[None, :]
            num_next_exact += np.sum(coeff * exact_next_overlap)
    
            # Geometric projection residual:
            #
            #   U_a m_i - fac[a,i] * m_{mapped(a,i)}
            #
            reconstructed = fac_chunk[..., None] * self.mos[ix_chunk]
            resid = U_chunk - reconstructed
    
            resid_norm = np.linalg.norm(resid, axis=-1)
            exact_norm = np.linalg.norm(U_chunk, axis=-1)
    
            weight_abs = np.abs(coeff)
    
            weighted_resid2 += np.sum((weight_abs * resid_norm) ** 2)
            weighted_exact2 += np.sum((weight_abs * exact_norm) ** 2)
    
            local_max = np.max(resid_norm)
            if local_max > max_pointwise_resid:
                local_arg = np.unravel_index(np.argmax(resid_norm), resid_norm.shape)
                aa = a0 + local_arg[0]
                ii = local_arg[1]
                max_pointwise_resid = local_max
                worst = (aa, ii)
    
            local_weighted = np.max(weight_abs * resid_norm)
            max_weighted_resid = max(max_weighted_resid, local_weighted)
    
        E_next_exact = (
            Lambda * (1.0 - num_next_exact / den_next_exact)
            if abs(den_next_exact) > eps
            else np.nan
        )
    
        weighted_geom_error = np.sqrt(weighted_resid2 / max(weighted_exact2, eps))
    
        out = {
            # Current printed energy.
            "E_curr": E_curr,
            "den_curr": den_curr,
            "num_curr": num_curr,
    
            # One-step-ahead comparison.
            "E_next_exact": E_next_exact,
            "E_next_proj": E_next_proj,
            "next_energy_projection_error": E_next_proj - E_next_exact,
    
            # Denominator/numerator errors at the next state.
            "den_next_exact": den_next_exact,
            "den_next_proj": den_next_proj,
            "den_next_error": den_next_proj - den_next_exact,
            "rel_den_next_error": abs(den_next_proj - den_next_exact) / max(abs(den_next_exact), eps),
    
            "num_next_exact": num_next_exact,
            "num_next_proj": num_next_proj,
            "num_next_error": num_next_proj - num_next_exact,
            "rel_num_next_error": abs(num_next_proj - num_next_exact) / max(abs(num_next_exact), eps),
    
            # Geometry-only projection diagnostics.
            "weighted_geom_error": weighted_geom_error,
            "max_pointwise_resid": max_pointwise_resid,
            "max_weighted_resid": max_weighted_resid,
            "worst": worst,
        }
    
        return out

    def get_initial_state(self):
        psi = np.zeros(self.Nh)
        ix, _ = self.mo2ix(self.trial)
        psi[ix] = 1.0
        return psi 

    def energy(self,psi):
        denom = np.dot(self.trial_mos, psi)
        num = np.dot(self.trial_Umos, psi)
        return self.ham.Lambda[-1] * (1.0 - num/denom)

if __name__=='__main__':
    Nangle,Nr = 30,16
    me = MasterEquation(Nangle,Nr)

    R = 1.9
    xc = "b3lyp"
    print(f"############### R={R:.1f} #################")
    with h5py.File(f"integrals/R{R:.1f}_{xc}.h5", "r") as f:
        h1e = f["hcore"][:]
        eri = f["eri"][:]
        mo_coeff = f["mo_coeff"][:]
    trial = mo_coeff[0, :, 0]
    Eref = np.dot(trial,np.dot(h1e,trial))
    Eref += .5*np.einsum('prqs,p,r,q,s->',eri,trial,trial,trial,trial)
    Eref -= .5*np.einsum('prqs,p,s,q,r->',eri,trial,trial,trial,trial)
    print('Eref=',Eref)
    #exit()
    
    nsite = eri.shape[0]
    cmax = nsite**2
    M = eri.reshape((nsite**2, nsite**2))
    print("eri symmetry=", np.linalg.norm(M - M.T))
    chol = modified_cholesky(M, cmax=cmax).reshape(-1, nsite, nsite)
    
    ham = QCSOR(apply_spin_down=False)
    iprint = 1
    #at = 10.*np.ones(nsite)
    #coeff = 5.*np.ones(chol.shape[:2]) 
    #ham.decompose_h1(h1e, at=at, iprint=iprint)
    #ham.decompose_h2(chol, coeff=coeff, iprint=iprint)
    d = 0.02
    ham.decompose_h1(h1e, dt=d, iprint=iprint)
    ham.decompose_h2(chol, di=d, iprint=iprint)
    ham.parse_decomposition()
    print('Lambda=',ham.Lambda)

    check = False
    me.build(ham,trial,check=False,tol=1e-3)
    psi = me.get_initial_state()

    Nstep = 5000 
    E = []
    for i in range(Nstep):
        E.append(me.energy(psi))

        Gpsi = me.mat_vec(psi)
        #diag = me.projection_effect_on_energy(psi,Gpsi=Gpsi)

        psi = Gpsi
        if i%50==0:
            print(
                f"step={i}, "
                f"E={E[-1]}, "
                #f"E_next_proj={diag['E_next_proj']}, "
                #f"E_next_exact={diag['E_next_exact']}, "
                #f"next_proj_err_E={diag['next_energy_projection_error']}, "
                #f"rel_den_next_err={diag['rel_den_next_error']}, "
                #f"rel_num_next_err={diag['rel_num_next_error']}, "
                #f"weighted_geom_err={diag['weighted_geom_error']}, "
                #f"max_pointwise_res={diag['max_pointwise_resid']}"
            )
