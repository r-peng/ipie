import time
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host 
import numpy as np

#np.set_printoptions(suppress=True, precision=6)

class GivensMasterEquation:

    def __init__(self, Ng=33, importance=True):
        self.importance = importance
        self.Ng = int(Ng)
        self.Nh = self.Ng

        # Same midpoint grid used in the AFQMC Fokker-Planck paper:
        self.dtheta = 2. / self.Ng
        self.theta_vals = -1. + (xp.arange(self.Ng) + 0.5) * self.dtheta
        print(f"Givens grid: Ng={self.Ng}, Nh={self.Nh}, xvals={self.theta_vals}")

        self.mos = xp.zeros((self.Nh,4))
        self.mos[:,1] = self.theta_vals
        self.mos[:,3] = xp.sqrt(1.-self.theta_vals**2)

    def local_vec_to_x(self, u, thresh=1e-12):
        """
        Convert normalized local vectors with u[...,0] >= 0 to Givens angles.
        Values very near the nodal boundary u0=0 are clipped to the finite grid domain.
        """
        x0 = u[..., 0]
        x1 = u[..., 1]
        x2 = u[..., 2]
        x3 = u[..., 3]
        assert xp.linalg.norm(x0)<thresh
        assert xp.linalg.norm(x2)<thresh
        return x1 

    def project_vecs_kdtree(self, vecs, thresh=1e-12, workers=-1, iprint=0, tree_key=None):
        """
        Globally nearest projection on the discrete Givens grid, using scipy.cKDTree.

        This is optimal for the grid under Euclidean distance of normalized orbitals,
        equivalently maximum determinant overlap for N=1 after sign fixing.
        """
        from scipy.spatial import cKDTree

        if tree_key in self.tree: 
            tree = self.tree[tree_key] 
        else:
            theta_vals = to_host(self.theta_vals) 
            tree = cKDTree(theta_vals.reshape(self.Ng,1))
            self.tree[tree_key] = tree 

        orig_shape = vecs.shape[:-1]
        v = vecs.reshape(-1, self.M)

        norms = xp.linalg.norm(v, axis=1)
        if xp.any(norms < thresh):
            raise ValueError("Encountered near-zero vector during projection.")

        u = v / norms[:, None]
        print(xp.linalg.norm(u[:,0]))
        assert xp.linalg.norm(u[:,0])<thresh
        assert xp.linalg.norm(u[:,2])<thresh

        flip = u[:, 3] < 0.0
        fac = norms.copy()
        fac[flip] *= -1.0
        u[flip] *= -1.0
        theta = u[:,1] 

        dist, ix = tree.query(to_host(theta.reshape(theta.size,1)), k=1, workers=workers)
        ix = xp.asarray(ix)
        if iprint>0:
            dist = xp.asarray(dist)*fac
            print('  max dist=',xp.amax(dist))
            print('  rms dist=',xp.sqrt((dist**2).sum()/dist.size))
        return ix.reshape(orig_shape), fac.reshape(orig_shape)

    def compute_projection_map(self,Us,a,iprint=0,tree_key=None):
        t0 = time.time()
        Umos = xp.einsum("axy,iy->aix", Us, self.mos, optimize=True)
        aUmos = xp.einsum("a,aix->ix", a, Umos, optimize=True)
        if iprint>0:
            print("computing Umo time=", time.time() - t0)

        t0 = time.time()
        Umos_ixs, Umos_fac = self.project_vecs_kdtree(Umos,iprint=iprint,tree_key=tree_key)
        if iprint>0:
            print("projection time=", time.time() - t0)

        # Precompute weighted map for fast matvec.
        Umos_fac *= a[:, None]
        if self.importance:
            Umos_fac *= self.trial_mos[Umos_ixs]
            Umos_fac /= self.trial_mos[None,:]
        return Umos_ixs.ravel(),Umos_fac

    def build(self, ham, trial, check=False, full=True):
        self.ham = ham
        self.trial = trial.copy()
        self.trial /= xp.linalg.norm(self.trial)
        self.M = self.trial.size
        if self.M != 4:
            raise ValueError("This implementation assumes one electron in M=4 real orbitals.")

        # Convert local grid orbitals to the original coefficient basis.
        self.trial_mos = self.mos @ self.trial
        self.Us = xp.asarray([ham.get_rotation_matrix(ix)[0] for ix in range(ham.nterms)])
        self.aUs = xp.einsum('a,axy->xy',self.ham.a,self.Us)
        self.gtrial = xp.dot(self.trial,self.aUs)

        self.tree = dict()
        self.Umos_ixs = None
        self.Umos_fac = None
        if not full:
            print(f'Ng={self.Ng}, use small_mem version.')
            return

        self.Umos_ixs,self.Umos_fac = self.compute_projection_map(self.Us,self.ham.a,iprint=1,tree_key='full')
        if not check:
            return
        t0 = time.time()
        psi_test = xp.random.rand(self.Nh) * 2 - 1
        G1 = self.mat_vec(psi_test)
        G2 = self.mat_vec_chunked(psi_test, chunk_terms=32)
        G3 = self.mat_vec_slow(psi_test)
        print("check matvec=", xp.linalg.norm(G1 - G3), xp.linalg.norm(G1))
        print("check matvec=", xp.linalg.norm(G2 - G3), xp.linalg.norm(G2))
        print("check vector time=", time.time() - t0)
        exit()

    def mat_vec(self, psi):
        vals = (self.Umos_fac * psi[None,:]).ravel()
        return xp.bincount(self.Umos_ixs, weights=vals, minlength=self.Nh)

    def mat_vec_chunked(self, psi, chunk_terms=30, fname=None):
        out = xp.zeros(self.Nh, dtype=psi.dtype)
        for start in range(0, self.ham.nterms, chunk_terms):
            stop = min(start + chunk_terms, self.ham.nterms)
            Us = self.Us[start:stop]
            a = self.ham.a[start:stop]
            if fname is None:
                ixs,fac = self.compute_projection_map(Us,a)
            else:
                try:
                    ixs = xp.asarray(np.load(f'{fname}_ixs_{start}_{stop}.npy'))
                    fac = xp.asarray(np.load(f'{fname}_fac_{start}_{stop}.npy'))
                except FileNotFoundError:
                    ixs,fac = self.compute_projection_map(Us,a,tree_key=(start,stop))
                    np.save(f'{fname}_ixs_{start}_{stop}.npy',to_host(ixs))
                    np.save(f'{fname}_fac_{start}_{stop}.npy',to_host(fac))
            vals = (fac * psi[None, :]).ravel()
            out += xp.bincount(ixs, weights=vals, minlength=self.Nh)
            del ixs,fac,vals
        return out

    def mat_vec_slow(self,psi):
        psi_new = xp.zeros_like(psi)
        sh1,sh2 = self.Umos_fac.shape
        Umos_ixs = self.Umos_ixs.reshape(sh1,sh2)
        fac = self.Umos_fac * psi[None,:]
        for j in range(sh1):
            for i in range(sh2):
                ix = Umos_ixs[j,i]
                psi_new[ix] += fac[j,i]
        return psi_new

    #def check_projection_error(self, tol=1e-8, chunk_terms=256):
    #    nterms, Nh, norb = self.Umos.shape
    #    max_err = 0.0
    #    sum_err2 = 0.0
    #    bad_count = 0
    #    worst = None

    #    for a0 in range(0, nterms, chunk_terms):
    #        a1 = min(a0 + chunk_terms, nterms)
    #        ix_chunk = self.Umos_ixs[a0:a1]
    #        fac_chunk = self.Umos_fac[a0:a1]
    #        reconstructed = self.mos[ix_chunk] * fac_chunk[..., None]
    #        diff = reconstructed - self.Umos[a0:a1]
    #        err = xp.linalg.norm(diff, axis=-1)

    #        local_max = xp.max(err)
    #        if local_max > max_err:
    #            local_arg = xp.unravel_index(xp.argmax(err), err.shape)
    #            worst = (a0 + local_arg[0], local_arg[1], local_max)
    #            max_err = local_max

    #        sum_err2 += xp.sum(err * err)
    #        bad_count += xp.count_nonzero(err > tol)

    #    rms_err = xp.sqrt(sum_err2 / (nterms * Nh))
    #    print("projection check:")
    #    print("  max_err  =", max_err)
    #    print("  rms_err  =", rms_err)
    #    print("  bad_count=", bad_count, "out of", nterms * Nh)

    #    if worst is not None:
    #        a, i, _ = worst
    #        print("  worst term/grid =", a, i)
    #        print("  worst idx       =", self.flat2idx(i))
    #        print("  mapped idx      =", self.flat2idx(self.Umos_ixs[a, i]))
    #        print("  original Umos   =", self.Umos[a, i])
    #        print("  reconstructed  =", self.Umos_fac[a, i] * self.mos[self.Umos_ixs[a, i]])

    #    if bad_count > 0:
    #        print(f"WARNING: found {bad_count} projection errors larger than {tol:g}")

    #    return max_err, rms_err, bad_count

    #def projection_effect_on_actual_energy(self, psi, Gpsi=None, chunk_terms=256, eps=1e-14):
    #    if Gpsi is None:
    #        Gpsi = self.mat_vec(psi)

    #    Lambda = self.ham.Lambda[-1]

    #    den_curr = np.dot(self.trial_mos, psi)
    #    num_curr = np.dot(self.trial_Umos, psi)
    #    E_curr = Lambda * (1.0 - num_curr / den_curr) if abs(den_curr) > eps else xp.nan

    #    den_next_proj = xp.dot(self.trial_mos, Gpsi)
    #    num_next_proj = xp.dot(self.trial_Umos, Gpsi)
    #    E_next_proj = Lambda * (1.0 - num_next_proj / den_next_proj) if abs(den_next_proj) > eps else xp.nan

    #    # Exact unprojected next denominator is <T|G|psi>.
    #    den_next_exact = num_curr

    #    # Exact unprojected next numerator is <T|G^2|psi>.
    #    num_next_exact = 0.0
    #    weighted_resid2 = 0.0
    #    weighted_exact2 = 0.0
    #    max_pointwise_resid = 0.0
    #    max_weighted_resid = 0.0
    #    worst = None

    #    for a0 in range(0, self.ham.nterms, chunk_terms):
    #        a1 = min(a0 + chunk_terms, self.ham.nterms)
    #        U_chunk = self.Umos[a0:a1]
    #        ix_chunk = self.Umos_ixs[a0:a1]
    #        fac_chunk = self.Umos_fac[a0:a1]
    #        a_chunk = self.ham.a[a0:a1]

    #        exact_next_overlap = xp.einsum("x,aix->ai", self.gtrial, U_chunk, optimize=True)
    #        coeff = a_chunk[:, None] * psi[None, :]
    #        num_next_exact += xp.sum(coeff * exact_next_overlap)

    #        reconstructed = fac_chunk[..., None] * self.mos[ix_chunk]
    #        resid = U_chunk - reconstructed
    #        resid_norm = xp.linalg.norm(resid, axis=-1)
    #        exact_norm = xp.linalg.norm(U_chunk, axis=-1)
    #        weight_abs = xp.abs(coeff)

    #        weighted_resid2 += xp.sum((weight_abs * resid_norm) ** 2)
    #        weighted_exact2 += xp.sum((weight_abs * exact_norm) ** 2)

    #        local_max = xp.max(resid_norm)
    #        if local_max > max_pointwise_resid:
    #            local_arg = xp.unravel_index(xp.argmax(resid_norm), resid_norm.shape)
    #            worst = (a0 + local_arg[0], local_arg[1])
    #            max_pointwise_resid = local_max

    #        max_weighted_resid = max(max_weighted_resid, xp.max(weight_abs * resid_norm))

    #    E_next_exact = (
    #        Lambda * (1.0 - num_next_exact / den_next_exact)
    #        if abs(den_next_exact) > eps
    #        else xp.nan
    #    )

    #    return {
    #        "E_curr": E_curr,
    #        "den_curr": den_curr,
    #        "num_curr": num_curr,
    #        "E_next_exact": E_next_exact,
    #        "E_next_proj": E_next_proj,
    #        "next_energy_projection_error": E_next_proj - E_next_exact,
    #        "den_next_exact": den_next_exact,
    #        "den_next_proj": den_next_proj,
    #        "den_next_error": den_next_proj - den_next_exact,
    #        "rel_den_next_error": abs(den_next_proj - den_next_exact) / max(abs(den_next_exact), eps),
    #        "num_next_exact": num_next_exact,
    #        "num_next_proj": num_next_proj,
    #        "num_next_error": num_next_proj - num_next_exact,
    #        "rel_num_next_error": abs(num_next_proj - num_next_exact) / max(abs(num_next_exact), eps),
    #        "weighted_geom_error": np.sqrt(weighted_resid2 / max(weighted_exact2, eps)),
    #        "max_pointwise_resid": max_pointwise_resid,
    #        "max_weighted_resid": max_weighted_resid,
    #        "worst": worst,
    #    }

    def get_initial_state(self):
        psi = xp.zeros(self.Nh)
        ix, _ = self.project_vecs_kdtree(self.trial[None, :])
        print('trial initial state index',ix)
        psi[int(ix[0])] = 1.0
        return psi

    def energy(self, psi):
        if self.importance:
            psi = psi/self.trial_mos
        denom = xp.dot(self.trial_mos, psi)
        num = xp.dot(xp.dot(psi,self.mos), self.gtrial)
        return self.ham.Lambda[-1] * (1.0 - num / denom),denom

    def normalize(self,weight):
        if self.importance:
            psi = weight/self.trial_mos
        else:
            psi = weight
        psi_mb = xp.dot(psi,self.mos)
        norm = xp.linalg.norm(psi_mb)
        return weight/norm

    def diagnostics(self,i,psi,Gpsi_proj):
        E_i,denom = self.energy(psi)

        if self.importance:
            psi = psi/self.trial_mos
            Gpsi_proj = Gpsi_proj/self.trial_mos
        Gpsi_proj_mb = xp.dot(Gpsi_proj,self.mos)
        num_proj = xp.dot(self.trial,Gpsi_proj_mb)

        psi_mb = xp.dot(psi,self.mos)
        Gpsi_unproj_mb = xp.dot(self.aUs,psi_mb) 
        num_unproj = xp.dot(self.trial,Gpsi_unproj_mb)
        assert xp.fabs(self.ham.Lambda[-1]*(1.0-num_unproj/denom)-E_i)<1e-10

        wfn_dist = xp.linalg.norm(Gpsi_proj_mb-Gpsi_unproj_mb)

        norm_proj = xp.linalg.norm(Gpsi_proj_mb)
        Gpsi_proj_mb /= norm_proj
        norm_unproj = xp.linalg.norm(Gpsi_unproj_mb)
        Gpsi_unproj_mb /= norm_unproj 
        wfn_dist_normalized = xp.linalg.norm(Gpsi_proj_mb-Gpsi_unproj_mb)
        print(
            f"step={i},E={E_i},",
            f"dtrial_proj={xp.fabs(num_proj-num_unproj)},",
            f"abs wfn dist={wfn_dist},",
            f"rel wfn dist={wfn_dist/norm_unproj},",
            f'norm proj={norm_proj},',
            f'norm unproj={norm_unproj},',
            f"normalized wfn dist={wfn_dist_normalized},"
            f'Gpsi_proj_mb={Gpsi_proj_mb}'
        )

    def exact_trial_energy_under_G(self):
        """Energy estimator evaluated at the exact trial orbital, not the grid delta."""
        denom = xp.dot(self.trial, self.trial)
        num = xp.dot(self.gtrial, self.trial)
        return self.ham.Lambda[-1] * (1.0 - num / denom)

    def run(self,start,stop,psi,print_every=1,normalize_every=10,chunk=None,fname=None):
        for i in range(start,stop):
            if self.Umos_ixs is None:
                Gpsi_proj = self.mat_vec_chunked(psi,chunk_terms=chunk,fname=fname)  
            else:
                Gpsi_proj = self.mat_vec(psi)  
            #diag = me.projection_effect_on_actual_energy(psi, Gpsi=Gpsi)
            if i%print_every==0:
                self.diagnostics(i,psi,Gpsi_proj)
            psi = Gpsi_proj
            if i%normalize_every==0:
                psi = self.normalize(psi)
        return psi


if __name__ == "__main__":
    import h5py, time
    import numpy as np
    # For rough cost matching with your previous polar grid:
    #   polar (Nangle=20, Nr=10): Nh = 11 * 80 * 41 = 36080
    #   givens Ng=33:             Nh = 33^3 = 35937
    Ng = 45
    projection = "kdtree"  # choose from: "coordinate", "local", "kdtree"
    me = GivensMasterEquation(Ng=Ng)

    R = 1.9
    xc = "b3lyp"
    print(f"############### R={R:.1f} #################")
    with h5py.File(f"integrals/R{R:.1f}_{xc}.h5", "r") as f:
        h1e = f["hcore"][:]
        eri = f["eri"][:]
        mo_coeff = f["mo_coeff"][:]

    trial = mo_coeff[0, :, 0]
    trial = trial / np.linalg.norm(trial)

    Eref = np.dot(trial, h1e @ trial)
    Eref += 0.5 * np.einsum("prqs,p,r,q,s->", eri, trial, trial, trial, trial)
    Eref -= 0.5 * np.einsum("prqs,p,s,q,r->", eri, trial, trial, trial, trial)
    print("Eref=", Eref)
    trial = xp.asarray(trial)

    nsite = eri.shape[0]
    cmax = nsite**2
    M = eri.reshape((nsite**2, nsite**2))
    print("eri symmetry=", np.linalg.norm(M - M.T))
    chol = modified_cholesky(M, cmax=cmax).reshape(-1, nsite, nsite)

    ham = QCSOR(apply_spin_down=False)
    iprint = 1
    at = 10
    ham.decompose_h1(h1e, at=at*np.ones(nsite), iprint=iprint)
    ai = 5
    ham.decompose_h2(chol, coeff=ai*np.ones(chol.shape[:2]), iprint=iprint)
    ham.parse_decomposition()

    me.build(ham, trial, projection=projection, tol=1e-3)
    print("exact trial energy under G=", me.exact_trial_energy_under_G())

    start,stop = 0,1000 
    if start==0:
        psi = me.get_initial_state()
        print("grid initial energy=", me.energy(psi))
        print("initial projection energy error=", me.energy(psi) - me.exact_trial_energy_under_G())
    else:
        psi = np.load(f'at{at}_ai{ai}_Ng{Ng}_stop{start}.npy')
        psi = xp.asarray(psi)
    me.run(start,stop,psi)
    np.save(f'at{at}_ai{ai}_Ng{Ng}_stop{stop}.npy',psi)
