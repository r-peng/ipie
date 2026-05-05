import numpy as np
import scipy,itertools
from ipie.hamiltonians.bitstring_utils import * 
from ipie.hamiltonians.generic_base import GenericBase
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
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
        self.gs = np.array(gs)
        self.ds = np.exp(self.gs)-1
        self.ns = self.ds.size
    def select(self,D,axes):
        for ax in axes:
            D = np.take(D,self.ps,axis=ax)
        return D
    def get_MB_kappa(self,v,basis):
        nsite = v.shape[0]
        kappa = [None] * 2
        for p,g in zip(self.ps,self.gs):
            s,ps = p//nsite,p%nsite
            if kappa[s] is None:
                kappa[s] = 0
            ks = np.outer(v[:,ps],v[:,ps]*g)
            kappa[s] += quadratic2MB(ks,basis,s) 
        return kappa
    def get_rotation_matrix(self,v):
        nsite = v.shape[0]
        U = [None] * 2
        for p,d in zip(self.ps,self.ds):
            s,ps = p//nsite,p%nsite
            if U[s] is None:
                U[s] = np.ones(nsite)
            U[s][ps] = d+1
        for s,Us in enumerate(U):
            if Us is None:
                continue
            U[s] = np.einsum('xp,yp,p->xy',v,v,Us)
        return U
    def apply_rotation(self,phi0,v,nsite):
        phi = [phis.copy() for phis in phi0]
        for p,d in zip(self.ps,self.ds):
            s,ps = p//nsite,p%nsite
            if v is None:
                phi[s][ps] += d*phi0[s][ps]
            else:
                vec = np.take(v,ps,axis=1)
                phi[s] += d*np.outer(vec,np.dot(vec,phi0[s]))
        return phi

class SumOfRotationBase(GenericBase):

    def __init__(self,h1e,U,eps_sq=None):
        super().__init__(h1e)
        self.U = U 
        self.eps_sq = eps_sq

        self.chol_basis = []
        self.terms = []

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        # hopping
        self.Lambda1 = 0.
        if RANK>0:
            iprint = 0

        eks,vk = np.linalg.eigh(self.H1) 
        self.chol_basis.append(vk)
        terms = []
        for k,ek in enumerate(eks):
            if np.fabs(ek)<thresh:
                continue
            ak = at
            gk = 1.-ek/ak
            if gk<0:
                raise ValueError
            gk = np.log(gk)
            self.Lambda1 += 2*ak
            if iprint>0:
                print(f'band={k},ek={ek},gk={gk}')
            terms.append(Udiag(ak,(k,),(gk,)))
            terms.append(Udiag(ak,(k+self.nbasis,),(gk,)))
        self.terms.append(terms)
        return eks

    def decompose_h2(self,gu,iprint=0,nelec=None):
        # onsite interaction
        self.Lambda2 = 0.
        if RANK>0:
            iprint = 0

        self.chol_basis.append(None)
        terms = []
        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('a_U=',ai)
        self.Lambda2 += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda2 += self.U*sum(nelec)/2 
        for i in range(self.nbasis): 
            terms.append(Udiag(ai,(i,i+self.nbasis,),(gu,-gu,)))
            terms.append(Udiag(ai,(i,i+self.nbasis,),(-gu,gu,)))
        self.terms.append(terms)

    def parse_decomposition(self):
        self.keys = [(d,i) for d,terms in enumerate(self.terms) for i in range(len(terms))]
        self.nkeys = len(self.keys)
        self.key_map = {key:kix for kix,key in enumerate(self.keys)} 

        Lambda = self.Lambda1 + self.Lambda2
        self.bare_gf = np.zeros(self.nkeys) 
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                self.bare_gf[kix] = term.ai/Lambda

    def calc_gf(self,walkers,trial):
        ovlp,R0,R = self.calc_trial_ovlp_ratio(walkers,trial)
        gf = self.bare_gf.reshape(self.nkeys,1) * ovlp
        if R0 is None:
            return gf
        return gf*R0.reshape(1,walkers.nwalkers)/R

    def sample_from_gf(self,gf):
        sign = np.sign(gf)
        s = sign.flatten()
        nminus = len(s[s<-0.5])
        #if nminus>0:
        #    print('number of minus=',nminus)
        #    #exit()

        p = np.fabs(gf)
        b = p.sum(axis=0)
        nwalker = b.size
        p /= b.reshape(1,nwalker)
        keys = [None] * nwalker
        for w in range(nwalker):
            kix = np.random.choice(self.nkeys,p=p[:,w])
            keys[w] = self.keys[kix]
            b[w] *= sign[kix,w] 
        self.b = b
        return keys,b
    
    def local_energy(self,walkers,trial):
        ovlp,R0,_ = self.calc_trial_ovlp_ratio(walkers,trial,compute_R=False)

        E1 = 0
        E2 = 0
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                if d==0:
                    E1 += term.ai * ovlp[kix]
                else:
                    E2 += term.ai * ovlp[kix]
        E1 = self.Lambda1 - E1
        E2 = self.Lambda2 - E2
        if RANK==0:
            print('R0 mean=',np.mean(R0))
        return E1+E2,E1,E2,R0

    def _get_MB_gf(self,basis):
        H = 0
        for vi,terms in zip(self.chol_basis,self.terms): 
            for term in terms:
                kappa = term.get_MB_kappa(vi,basis)
                U = None
                for spin,k in enumerate(kappa):
                    if k is None:
                        continue
                    Us = scipy.linalg.expm(k)
                    if U is None:
                        U = Us
                    else:
                        U = np.dot(U,Us)
                H += term.ai*U
        return H

