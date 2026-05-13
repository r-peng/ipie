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
    def __init__(self,ai,ps,gs):
        self.ai = ai
        self.ps = ps
        self.gs = xp.array(gs)
        self.ds = xp.exp(self.gs)-1
        self.ns = self.ds.size
    def select(self,D,axes):
        for ax in axes:
            D = xp.take(D,self.ps,axis=ax)
        return D
    def apply_rotation(self,phi0,v,nsite):
        phi = [phis.copy() for phis in phi0]
        for p,d in zip(self.ps,self.ds):
            s,ps = p//nsite,p%nsite
            if v is None:
                phi[s][ps] += d*phi0[s][ps]
            else:
                vec = xp.take(v,ps,axis=1)
                phi[s] += d*xp.outer(vec,xp.dot(vec,phi0[s]))
        return phi

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

    def apply_rotation(self,C,UC,r,i):
        ps = self.p[r][i]
        ds = self.d[r][i]
        for p,d in zip(ps,ds):
            s,p_ = p//self.nbasis,p%self.nbasis
            if self.chol_basis is None:
                C[s,p_] += d*UC[s,p_]
            else:
                left = d*self.chol_basis[:,p_]
                right = UC[s,p_]
                C[s] += xp.outer(left,right)
        return C 

class SumOfRotationBase(GenericBase):

    def __init__(self,h1e,U,eps_sq=None):
        super().__init__(h1e)
        self.U = U 
        self.eps_sq = eps_sq

        self.batches = []
        self.Lambda = xp.zeros(3)
        self.H1 = xp.asarray(self.H1)

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        # hopping
        if RANK>0:
            iprint = 0

        eks,vk = np.linalg.eigh(self.H1) 
        batch = BatchTerms(self.nbasis,vk,rs=[1])
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
            batch.add_term(ak,[k],[gk])
            batch.add_term(ak,[k+self.nbasis],[gk])
        self.batches.append(batch)
        return eks

    def decompose_h2(self,gu,iprint=0,nelec=None):
        # onsite interaction
        if RANK>0:
            iprint = 0

        batch = BatchTerms(self.nbasis,None,rs=[2])
        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('a_U=',ai)
        self.Lambda[1] += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda[1] += self.U*sum(nelec)/2 
        for i in range(self.nbasis): 
            batch.add_term(ai,[i,i+self.nbasis],[gu,-gu])
            batch.add_term(ai,[i,i+self.nbasis],[-gu,gu])
        self.batches.append(batch)

    def parse_decomposition(self):
        self.Lambda[2] = self.Lambda[0] + self.Lambda[1]
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

    def sample_from_gf(self,gf):
        sign = xp.sign(gf)
        #nminus = int(to_host(xp.sum(sign < -0.5)))
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
        E1 = self.Lambda[0]-xp.dot(self.bare_gf[self.E1_kix],ovlp[self.E1_kix])*self.Lambda[2]
        E2 = self.Lambda[1]-xp.dot(self.bare_gf[self.E2_kix],ovlp[self.E2_kix])*self.Lambda[2]
        if R0 is not None:
            if RANK==0:
                print('R0 mean=',to_host(xp.mean(R0)))
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
