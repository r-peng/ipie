import numpy as np
import scipy,itertools
from ipie.config import config
from ipie.hamiltonians.bitstring_utils import * 
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()
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
def hubbard2MB(h1e,U,nelecs,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = np.zeros((len(basis),)*2)
    for ix1,cf1 in enumerate(basis):
        for i,j in itertools.product(range(nsite),repeat=2):
            if np.fabs(h1e[i,j])<thresh:
                continue
            for s in (0,1):
                ops = (2*i+s,'cre'),(2*j+s,'des') 
                cf2,sign = string_act(cf1,ops)
                if cf2 is not None:
                    ix2 = basis_map[cf2]
                    H[ix2,ix1] += h1e[i,j]*sign
        H[ix1,ix1] += U*count_double_occupancy(cf1,nsite)
    return H,basis,basis_map

class Udiag:
    def __init__(self,nbasis,pa=None,ga=None,pb=None,gb=None):
        self.ps = [None] * 2
        self.ds = [None] * 2
        if pa is not None:
            self.ps[0] = xp.asarray(pa)
            self.ds[0] = xp.exp(xp.asarray(ga))-1.
        if pb is not None:
            self.ps[1] = xp.asarray(pb) 
            self.ds[1] = xp.exp(xp.asarray(gb))-1.

        if (pa is not None) and (pb is not None):
            self.p = xp.concatenate([self.ps[0],self.ps[1]+nbasis])
            self.d = xp.concatenate(self.ds)
        elif pa is None:
            self.p = self.ps[1]+nbasis
            self.d = self.ds[1]
        else:
            self.p = self.ps[0]
            self.d = self.ds[0]
        self.n = self.d.size

    def select(self,D,axes):
        for ax in axes:
            D = xp.take(D,self.p,axis=ax)
        return D
    def get_MB_kappa(self,v,basis):
        nsite = v.shape[0]
        kappa = [None] * 2
        for s,p in enumerate(self.ps):
            if p is None:
                continue
            ks = np.outer(v[:,p],v[:,p]*(self.ds[s]+1.))
            kappa[s] = quadratic2MB(ks,basis,s) 
        return kappa
    def get_rotation_matrix(self,v):
        nsite = v.shape[0]
        U = [None] * 2
        for s,p in enumerate(self.ps):
            if p is None:
                continue
            diag = xp.ones(nsite)
            diag[p] += self.ds[s]
            U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        return U
    def apply_rotation(self,phi,v,nsite):
        for s,p in enumerate(self.ps):
            if p is None:
                continue
            if v is None:
                phi[s][p] += self.ds[s]*phi[s][p]
            else:
                vec = xp.take(v,p,axis=1)
                phi[s] += self.ds[s]*xp.outer(vec,xp.dot(vec,phi[s]))
        return phi

class SumOfRotationBase(GenericBase):

    def __init__(self,h1e,U,eps_sq=None):
        super().__init__(h1e)
        self.U = U 
        self.eps_sq = eps_sq

        self.chol_basis = []
        self.terms = []
        self.bare_gf = []
        self.Lambda = np.zeros(3)

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        # hopping
        if RANK>0:
            iprint = 0

        eks,vk = np.linalg.eigh(self.H1) 
        self.chol_basis.append(vk)
        terms = []
        bare_gf = []
        for k,ek in enumerate(eks):
            if np.fabs(ek)<thresh:
                continue
            ak = at
            gk = 1.-ek/ak
            if gk<0:
                raise ValueError
            gk = np.log(gk)
            self.Lambda[0] += 2*ak
            if iprint>0:
                print(f'band={k},ek={ek},gk={gk}')

            terms.append(Udiag(self.nbasis,pa=[k],ga=[gk]))
            bare_gf.append(ak)

            terms.append(Udiag(self.nbasis,pb=[k],gb=[gk]))
            bare_gf.append(ak)
        self.terms.append(terms)
        self.bare_gf.append(bare_gf)
        return eks

    def decompose_h2(self,gu,iprint=0,nelec=None):
        # onsite interaction
        if RANK>0:
            iprint = 0

        self.chol_basis.append(None)
        terms = []
        bare_gf = []
        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('a_U=',ai)
        self.Lambda[1] += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda[1] += self.U*sum(nelec)/2 
        for i in range(self.nbasis): 
            terms.append(Udiag(self.nbasis,pa=[i],pb=[i],ga=[gu],gb=[-gu]))
            bare_gf.append(ai)
            terms.append(Udiag(self.nbasis,pa=[i],pb=[i],ga=[-gu],gb=[gu]))
            bare_gf.append(ai)
        self.terms.append(terms)
        self.bare_gf.append(bare_gf)

    def parse_decomposition(self):
        self.Lambda[2] = self.Lambda[0] + self.Lambda[1]
        self.keys = [(d,i) for d,terms in enumerate(self.terms) for i in range(len(terms))]
        self.bare_gf = np.array([ai for bare_gf in self.bare_gf for ai in bare_gf])/self.Lambda[2]
        self.nkeys = len(self.keys)
        self.key_map = {key:kix for kix,key in enumerate(self.keys)} 
        self.E1_kix = []
        self.E2_kix = []
        for (d,i),kix in self.key_map.items():
            if d==0:
                self.E1_kix.append(kix)
            else:
                self.E2_kix.append(kix)
        self.E1_kix = np.array(self.E1_kix)
        self.E2_kix = np.array(self.E2_kix)

        self.cast_to_cupy()

    def cast_to_cupy(self, verbose=False):
        if not config.get_option("use_gpu"):
            return
        if verbose:
            print(f"# {self.__class__.__name__}: moving SOR arrays to GPU")
        self.H1 = xp.asarray(self.H1)
        self.Lambda = xp.asarray(self.Lambda)
        self.E1_kix = xp.asarray(self.E1_kix)
        self.E2_kix = xp.asarray(self.E2_kix)
        self.bare_gf = xp.asarray(self.bare_gf)
        self.chol_basis = [None if basis is None else xp.asarray(basis) for basis in self.chol_basis]

    def calc_gf(self,walkers,trial):
        ovlp,R0,R = self.calc_trial_ovlp_ratio(walkers,trial)
        gf = self.bare_gf.reshape(self.nkeys,1) * ovlp
        if R0 is None:
            return gf
        return gf*R0.reshape(1,walkers.nwalkers)/R

    def sample_from_gf(self,gf):
        sign = xp.sign(gf)
        nminus = int(to_host(xp.sum(sign < -0.5)))
        #if nminus>0:
        #    print('number of minus=',nminus)
        #    #exit()

        p = xp.fabs(gf)
        b = p.sum(axis=0)
        nwalker = b.size
        p /= b.reshape(1,nwalker)
        cdf = xp.cumsum(p, axis=0)
        sample = xp.random.random(nwalker)
        kixs = xp.sum(cdf < sample.reshape(1,nwalker), axis=0).astype(xp.int64)
        kixs = xp.minimum(kixs, self.nkeys - 1)
        b *= sign[kixs, xp.arange(nwalker)]
        keys = [self.keys[int(kix)] for kix in to_host(kixs)]
        self.b = b
        return keys,b
    
    def local_energy(self,walkers,trial):
        ovlp,R0,_ = self.calc_trial_ovlp_ratio(walkers,trial,compute_R=False)

        E = [xp.dot(self.bare_gf[kixs],ovlp[kixs])*self.Lambda[2] for kixs in (self.E1_kix,self.E2_kix)]
        E1 = self.Lambda[0] - E[0]
        E2 = self.Lambda[1] - E[1]
        if R0 is not None:
            if RANK==0:
                print('R0 mean=',to_host(xp.mean(R0)))
        return E1+E2,E1,E2,R0

    def _get_MB_gf(self,basis):
        H = 0
        for d,terms in enumerate(self.terms): 
            for i,term in enumerate(terms):
                v = self.chol_basis[d]
                if v is None:
                    v = np.eye(self.nbasis)
                kappa = term.get_MB_kappa(v,basis)
                U = None
                for spin,k in enumerate(kappa):
                    if k is None:
                        continue
                    Us = scipy.linalg.expm(k)
                    if U is None:
                        U = Us
                    else:
                        U = np.dot(U,Us)
                kix = self.key_map[d,i]
                H += self.bare_gf[kix]*U
        return H
