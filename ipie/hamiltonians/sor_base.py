import numpy as np
import scipy,itertools,plum
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        string_act,
        count_double_occupancy, 
)
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host

class BatchTerms:

    def __init__(self,nbasis,chol_basis,rs=[1]):
        self.nbasis = nbasis
        self.chol_basis = chol_basis
        self.rs = rs
        self.p = {r:[] for r in rs}
        self.d = {r:[] for r in rs}
        self.g = {r:[] for r in rs}
        self.a = {r:[] for r in rs}
        self.kix = {r:[] for r in rs}

    def add_term(self,ai,pi,gi):
        r = len(pi)
        self.a[r].append(ai)
        pi = xp.asarray(pi,dtype=int)
        gi = xp.asarray(gi)
        di = xp.exp(gi)-1.
        self.p[r].append(pi)
        self.g[r].append(gi)
        self.d[r].append(di)

    def get_MB_kappa(self,r,i,basis):
        v = self.chol_basis
        if v is None:
            v = np.eye(self.nbasis)
        ps = self.p[r][i]
        gs = self.g[r][i]

        kappa = [None] * 2
        for p,g in zip(ps,gs):
            s,p_ = p//self.nbasis,p%self.nbasis
            if kappa[s] is None:
                kappa[s] = 0
            ks = np.outer(v[:,p_],v[:,p_]*g)
            kappa[s] += quadratic2MB(ks,basis,s) 
        return kappa

    def get_rotation_matrix(self,r,i):
        v = self.chol_basis
        if v is None:
            v = np.eye(self.nbasis)
        ps = self.p[r][i]
        ds = self.d[r][i]

        U = [None] * 2
        for p,d in zip(ps,ds):
            s,p_ = p//self.nbasis,p%self.nbasis
            if U[s] is None:
                U[s] = xp.ones(self.nbasis)
            U[s][p_] = d+1
        for s,Us in enumerate(U):
            if Us is None:
                continue
            U[s] = xp.einsum('xp,yp,p->xy',v,v,Us)
        return U

    def apply_rotation(self,C,r,i):
        ps = self.p[r][i]
        ds = self.d[r][i]
        for p,d in zip(ps,ds):
            s,p_ = p//self.nbasis,p%self.nbasis
            if C[s] is None:
                continue
            if self.chol_basis is None:
                C[s][p_] += d*C[s][p_]
            else:
                vec = self.chol_basis[:,p_]
                C[s] += xp.outer(d*vec,xp.dot(vec,C[s]))
        return C 

class SumOfRotationBase:

    def __init__(self,eps_sq=None):
        self.eps_sq = eps_sq
        self.batches = []
        self.Lambda = xp.zeros(3)

    def decompose_h1(self,h1e,at,thresh=1e-6,iprint=0):
        h1e = xp.asarray(h1e)
        self.nbasis = h1e.shape[0]

        ek,vk = xp.linalg.eigh(h1e) 
        assert at>xp.amax(xp.fabs(ek))
        nonzero_bands = xp.nonzero(ek)[0]
        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        eta = xp.log(1.-ek/at)

        batch = BatchTerms(self.nbasis,vk,rs=[1])
        self.Lambda[0] += at*2*nonzero_bands.size
        for k in nonzero_bands:
            batch.add_term(at,[k],[eta[k]])
            batch.add_term(at,[k+self.nbasis],[eta[k]])
        self.batches.append(batch)
        return ek

    def decompose_hubbard_h2(self,U,gu,iprint=0,nelec=None):
        batch = BatchTerms(self.nbasis,None,rs=[2])
        ai = U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('ai=',ai)
        self.Lambda[1] += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda[1] += U*sum(nelec)/2 
        for i in range(self.nbasis): 
            batch.add_term(ai,[i,i+self.nbasis],[gu,-gu])
            batch.add_term(ai,[i,i+self.nbasis],[-gu,gu])
        self.batches.append(batch)

    def decompose_qc_h2(self,chol,ai,thresh=1e-6,iprint=0):
        chol = xp.asarray(chol)
        aisq = ai**2/2
        if iprint>0:
            print('ai=',ai)
        for i,L in enumerate(chol):
            ek,vk = xp.linalg.eigh(L) 
            assert ai>xp.amax(xp.fabs(ek))
            if iprint>0:
                print(f'chol={i},bands={ek}')
            nonzero_bands = xp.nonzero(ek)[0]
            self.Lambda[1] += (ai*2*nonzero_bands.size)**2/2.
            eta_plus = xp.log(1.+ek/ai) 
            eta_minus = xp.log(1.-ek/ai) 

            batch = BatchTerms(self.nbasis,vk,rs=[1,2])
            for p,q in itertools.product(nonzero_bands,repeat=2):
                gi = [eta_minus[p],eta_plus[q]]
                if p==q:
                    batch.add_term(aisq,[p],[eta_plus[p]+eta_minus[p]])
                    batch.add_term(aisq,[p+self.nbasis],[eta_plus[p]+eta_minus[p]])
                    batch.add_term(aisq,[p,p+self.nbasis],gi)
                    batch.add_term(aisq,[p+self.nbasis,p],gi)
                else:
                    batch.add_term(aisq,[p,q],gi)
                    batch.add_term(aisq,[p,q+self.nbasis],gi)
                    batch.add_term(aisq,[p+self.nbasis,q],gi)
                    batch.add_term(aisq,[p+self.nbasis,q+self.nbasis],gi)
            self.batches.append(batch)

    def parse_decomposition(self,iprint=0):
        self.Lambda[2] = self.Lambda[0] + self.Lambda[1]
        if iprint>0:
            print('Lambda=',self.Lambda)
        self.keys = []
        self.bare_gf = [] 
        self.E1_kix = []
        self.E2_kix = []
        kix = 0
        for d,batch in enumerate(self.batches):
            for r in batch.rs:
                batch.p[r] = xp.stack(batch.p[r])
                batch.d[r] = xp.stack(batch.d[r])
                batch.g[r] = xp.stack(batch.g[r])
                for i,ai in enumerate(batch.a[r]):
                    self.keys.append((d,r,i))
                    self.bare_gf.append(ai)
                    if d==0:
                        self.E1_kix.append(kix)
                    else:
                        self.E2_kix.append(kix)
                    batch.kix[r].append(kix)
                    kix += 1
                batch.kix[r] = xp.asarray(batch.kix[r]) 

        self.key_map = {key:kix for kix,key in enumerate(self.keys)} 
        self.nkeys = len(self.keys)
        self.bare_gf = xp.asarray(self.bare_gf)/self.Lambda[2]
        self.E1_kix = xp.asarray(self.E1_kix)
        self.E2_kix = xp.asarray(self.E2_kix)

    def calc_gf(self,walkers,trial):
        ovlp,R0,R = self.calc_trial_ovlp_ratio(walkers,trial)
        gf = self.bare_gf.reshape(self.nkeys,1) * ovlp
        if R0 is None:
            return gf
        return gf*R0.reshape(1,walkers.nwalkers)/R

    @plum.dispatch
    def update_walkers(self,kixs,walkers:UHFWalkers):
        keys = [self.keys[int(kix)] for kix in to_host(kixs)]
        Ca = walkers.phia.copy()
        if walkers.ndown==0:
            Cb = [None] * walkers.nwalkers
        else:
            Cb = walkers.phib.copy()
        for w,(d,r,i) in enumerate(keys):
            batch = self.batches[d] 
            Cnew = [Ca[w],Cb[w]]
            Cnew = batch.apply_rotation(Cnew,r,i)
            walkers.phia[w] = Cnew[0]
            if walkers.ndown>0:
                walkers.phib[w] = Cnew[1]

    @plum.dispatch
    def update_walkers(self,kixs,walkers:GHFWalkers):
        keys = [self.keys[int(kix)] for kix in to_host(kixs)]
        nb = walkers.nbasis
        C = xp.stack([walkers.phi[:,:nb],walkers.phi[:,nb:]])
        for w,(d,r,i) in enumerate(keys):
            batch = self.batches[d] 
            C[:,w] = batch.apply_rotation(C[:,w],r,i)
        walkers.phi = xp.concatenate([C[0],C[1]],axis=1)

    def local_energy(self,walkers,trial):
        ovlp,R0,_ = self.calc_trial_ovlp_ratio(walkers,trial,compute_R=False)
        E1 = self.Lambda[0]-xp.dot(self.bare_gf[self.E1_kix],ovlp[self.E1_kix])*self.Lambda[2]
        E2 = self.Lambda[1]-xp.dot(self.bare_gf[self.E2_kix],ovlp[self.E2_kix])*self.Lambda[2]
        return E1+E2,E1,E2,R0

    def _get_MB_gf(self,basis):
        H = 0
        print('called')
        for d,batch in enumerate(self.batches): 
            for r in batch.rs:
                for i in range(len(batch.a[r])):
                    kappa = batch.get_MB_kappa(r,i,basis)
                    U = None
                    for spin,k in enumerate(kappa):
                        if k is None:
                            continue
                        Us = scipy.linalg.expm(k)
                        if U is None:
                            U = Us
                        else:
                            U = np.dot(U,Us)
                    kix = self.key_map[d,r,i]
                    H += self.bare_gf[kix]*U
        return H

##### MB helper fxns #####
def quadratic2MB(M,basis,spin,thresh=1e-6):
    if len(M.shape)==1:
        M = np.diag(M)
    basis_map = {cf:i for i,cf in enumerate(basis)}
    H = np.zeros((len(basis),)*2)
    for (p,q) in itertools.product(range(M.shape[0]),repeat=2):
        if np.absolute(M[p,q])<thresh:
            continue
        for ix1,cf1 in enumerate(basis):
            ops = (2*p+spin,'cre'),(2*q+spin,'des')
            cf2,sign = string_act(cf1,ops)
            if cf2 is None:
                continue
            ix2 = basis_map[cf2]
            H[ix2,ix1] += M[p,q]*sign
    return H

def bcs2MB(A,B,u,v,basis=None,basis_map=None): 
    nsite,npair = A.shape
    if basis is None:
        basis = all_bitstrings_list(nsite) 
    if basis_map is None:
        basis_map = {det:ix for ix,det in enumerate(basis)}
    nbasis = len(basis)
    def apply_cre(i,psi,A):
        return apply_a_dag_dense_sign(psi,basis,A[:,i],nsite,det_to_index=basis_map)
    
    psi = np.zeros(nbasis)
    psi[basis_map[0]] = 1.
    for k in range(npair//2-1,-1,-1):
        psi_v = apply_cre(2*k+1,psi,A)
        psi_v = apply_cre(2*k,psi_v,A)
        psi = u[2*k]*psi + v[2*k]*psi_v
    nocc = B.shape[1]
    for k in range(nocc-1,-1,-1):
        psi = apply_cre(k,psi,B)
    return psi,basis,basis_map

def det2MB(B,basis=None,basis_map=None,order=1):
    nsite,nocc = B.shape
    if basis is None:
        basis = all_bitstrings_list(nsite) 
    if basis_map is None:
        basis_map = {det:ix for ix,det in enumerate(basis)}
    nbasis = len(basis)
    def apply_cre(i,psi,A):
        return apply_a_dag_dense_sign(psi,basis,A[:,i],nsite,det_to_index=basis_map)
    psi = np.zeros(nbasis)
    psi[basis_map[0]] = 1.
    ks = range(nocc) if order==1 else range(nocc-1,-1,-1)
    for k in ks:
        psi = apply_cre(k,psi,B)
    return psi,basis,basis_map

def hubbard2MB(h1e,U,nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,1,thresh=thresh)
    for ix,cf in enumerate(basis):
        H[ix,ix] += U*count_double_occupancy(cf,nsite)
    return H,basis,basis_map

def chol2MB(h1e,chol,nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,1,thresh=thresh)
    for i,L in enumerate(chol):
        L_ = quadratic2MB(L,basis,0,thresh=thresh)
        L_ += quadratic2MB(L,basis,1,thresh=thresh)
        H += np.dot(L_,L_)/2.
    return H,basis,basis_map
