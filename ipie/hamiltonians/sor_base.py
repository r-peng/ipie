import numpy as np
import scipy,itertools
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        string_act,
        count_double_occupancy, 
)
from ipie.utils.backend import arraylib as xp

class Rotation:

    def __init__(self,chol_idx,pa=None,ga=None,pb=None,gb=None):
        self.chol_idx = chol_idx
        self.p = [None,None]
        self.g = [None,None]
        self.d = [None,None]
        self.r = [None,None]
        if pa is not None:
            self.p[0] = xp.asarray(pa,dtype=int)
            self.g[0] = xp.asarray(ga)
            self.d[0] = xp.exp(self.g[0])-1.
            self.r[0] = len(pa)
        if pb is not None:
            self.p[1] = xp.asarray(pb,dtype=int)
            self.g[1] = xp.asarray(gb)
            self.d[1] = xp.exp(self.g[1])-1.
            self.r[1] = len(pb)

    def get_MB_kappa(self,v,basis):
        kappa = [None] * 2
        for s in (0,1):
            p,g = self.p[s],self.g[s]
            if p is None:
                continue
            ks = np.einsum('xr,yr,r->xy',v[:,p],v[:,p],g)
            kappa[s] = quadratic2MB(ks,basis,s) 
        return kappa

    def get_rotation_matrix(self,v):
        U = [None] * 2
        for s in (0,1):
            p,d = self.p[s],self.d[s]
            if p is None:
                continue
            diag = xp.ones(v.shape[0])
            diag[p] += d
            U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        return U

class SumOfRotationBase:

    def __init__(self):
        self.chol_basis = []
        self.terms = [] 
        self.bare_gf = []
        self.Lambda = 0. 

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        self.nbasis = self.h1e.shape[0]

        ek,vk = xp.linalg.eigh(self.h1e-self.v0) 
        self.chol_basis.append(vk)
        chol_idx = len(self.chol_basis)-1

        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        assert at>xp.amax(xp.fabs(ek))
        nonzero_bands = xp.nonzero(ek)[0]
        eta = xp.log(1.-ek/at)

        self.bare_gf += [at] * (2*nonzero_bands.size)
        self.Lambda += at*2*nonzero_bands.size
        for k in nonzero_bands:
            self.terms.append(Rotation(chol_idx,pa=[k],ga=[eta[k]]))
            self.terms.append(Rotation(chol_idx,pb=[k],gb=[eta[k]]))
        return ek

    def parse_decomposition(self,iprint=0):
        if iprint>0:
            print('Lambda=',self.Lambda)
        self.chol_basis = xp.asarray(self.chol_basis)
        self.bare_gf = xp.asarray(self.bare_gf)/self.Lambda
        assert self.bare_gf.size==len(self.terms)
        if iprint>0:
            print('normalization=',xp.fabs(self.bare_gf).sum())

    def parse_sampled_rotations(self,kixs):
        dmap = {(s,r):[] for s in (0,1) for r in (1,2)}
        umap = {(s,r):[] for s in (0,1) for r in (1,2)}
        wmap = {(s,r):[] for s in (0,1) for r in (1,2)}
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            for s in (0,1):
                p = term.p[s]
                if p is None:
                    continue
                r = term.r[s]
                dmap[s,r].append(term.d[s])
                umap[s,r].append(self.chol_basis[term.chol_idx][:,p])
                wmap[s,r].append(w)
        for s,r in itertools.product((0,1),(1,2)):
            if len(dmap[s,r])==0:
                dmap[s,r] = None
                umap[s,r] = None
                wmap[s,r] = None
                continue
            dmap[s,r] = xp.asarray(dmap[s,r])
            umap[s,r] = xp.asarray(umap[s,r])
            wmap[s,r] = xp.asarray(wmap[s,r])
        return dmap,umap,wmap

    def parse_sampled_rotations_slow(self,kixs):
        Us = [None] * len(kixs)
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            Us[w] = term.get_rotation_matrix(self.chol_basis[term.chol_idx])
        return Us

    def local_energy(self,walkers):
        D = walkers.D 
        if isinstance(D,list):
            Daa,Dbb = D
            Dab = Dba = None
        else:
            nb = self.nbasis
            Daa,Dab,Dba,Dbb = D[:,:nb,:nb],D[:,:nb,nb:],D[:,nb:,:nb],D[:,nb:,nb:]

        E1 = xp.einsum('ij,wij->w',self.h1e,Daa+Dbb)
        E2 = self.compute_E2(Daa,Dbb,Dab=Dab,Dba=Dba)
        return E1+E2,E1,E2

    def _get_MB_gf(self,basis):
        H = 0
        print('called')
        for ai,term in zip(self.bare_gf,self.terms): 
            kappa = term.get_MB_kappa(self.chol_basis[term.chol_idx],basis)
            U = None
            for spin,k in enumerate(kappa):
                if k is None:
                    continue
                Us = scipy.linalg.expm(k)
                if U is None:
                    U = Us
                else:
                    U = np.dot(U,Us)
            H += ai*U
        return H

class HubbardSOR(SumOfRotationBase):

    def __init__(self,h1e,U):
        super().__init__()
        self.h1e = xp.asarray(h1e)
        self.U = U
        self.v0 = -0.5*U*xp.eye(self.h1e.shape[0]) 

    def decompose_h2(self,gu,iprint=0,nelec=None):
        self.chol_basis.append(xp.eye(self.nbasis))
        chol_idx = len(self.chol_basis)-1

        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('ai=',ai)
        self.bare_gf += [ai] * (2*self.nbasis)
        self.Lambda += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda += self.U*sum(nelec)/2 
        for i in range(self.nbasis): 
            self.terms.append(Rotation(chol_idx,pa=[i],ga=[gu],pb=[i],gb=[-gu]))
            self.terms.append(Rotation(chol_idx,pa=[i],ga=[-gu],pb=[i],gb=[gu]))

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = xp.einsum('wii,wii->w',Daa,Dbb)
        if Dab is None:
            return E2*self.U
        E2 -= xp.einsum('wii,wii->w',Dab,Dba)
        return E2*self.U

class QCSOR(SumOfRotationBase):

    def __init__(self,h1e,chol):
        super().__init__()
        self.h1e = xp.asarray(h1e)
        self.chol = xp.asarray(chol)
        self.v0 = .5*xp.einsum('npr,nrs->ps',self.chol,self.chol) 

    def decompose_h2(self,ai,thresh=1e-6,iprint=0):
        aisq = ai**2/2
        if iprint>0:
            print('ai=',ai)
        for i,L in enumerate(self.chol):
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_idx = len(self.chol_basis)-1

            if iprint>0:
                print(f'chol={i},bands={ek}')
            assert ai>xp.amax(xp.fabs(ek))
            nonzero_bands = xp.nonzero(ek)[0]
            self.Lambda += (ai*2*nonzero_bands.size)**2/2.
            eta_plus = xp.log(1.+ek/ai) 
            eta_minus = xp.log(1.-ek/ai) 

            for p,q in itertools.product(nonzero_bands,repeat=2):
                self.bare_gf += [aisq] * 4
                eta_p = eta_minus[p]
                eta_q = eta_plus[q]
                if p==q:
                    self.terms.append(Rotation(chol_idx,pa=[p],ga=[eta_p+eta_q]))
                    self.terms.append(Rotation(chol_idx,pb=[p],gb=[eta_p+eta_q]))
                else:
                    self.terms.append(Rotation(chol_idx,pa=[p,q],ga=[eta_p,eta_q]))
                    self.terms.append(Rotation(chol_idx,pb=[p,q],gb=[eta_p,eta_q]))
                self.terms.append(Rotation(chol_idx,pa=[p],ga=[eta_p],pb=[q],gb=[eta_q]))
                self.terms.append(Rotation(chol_idx,pa=[q],ga=[eta_q],pb=[p],gb=[eta_p]))

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = 0
        for i,L in enumerate(self.chol):
            DaaL = xp.einsum('wpq,qr->wpr',Daa,L)
            DbbL = xp.einsum('wpq,qr->wpr',Dbb,L)
            E2 += xp.einsum('wpp->w',DaaL+DbbL)**2
            E2 -= xp.einsum('wpq,wqp->w',DaaL,DaaL)
            E2 -= xp.einsum('wpq,wqp->w',DbbL,DbbL)
            if Dab is not None:
                DabL = xp.einsum('wpq,qr->wpr',Dab,L)
                DbaL = xp.einsum('wpq,qr->wpr',Dba,L)
                E2 -= 2.*xp.einsum('wpq,wqp->w',DabL,DbaL)
        return 0.5*E2

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
