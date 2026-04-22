import numpy as np
import scipy,itertools
from ipie.hamiltonians.bitstring_utils import * 
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
    def compute_trial_ovlp(self,D):
        Dj = self.select(D,(1,2))
        M = Dj+np.diag(1./self.ds).reshape(1,self.ns,self.ns)
        det = np.linalg.det(M)
        det *= self.ds.prod()
        return det
    def compute_M(self,D):
        # input matrix dim: walker,p,q
        Dj = self.select(D,(1,2))
        M1 = Dj+np.diag(1./self.ds).reshape(1,self.ns,self.ns)
        M1 = np.linalg.inv(M1)

        M2 = np.einsum('wij,wjk->wik',M1,Dj)
        M2 = np.eye(self.ns).reshape(1,self.ns,self.ns) - M2
        M2 = M2*self.ds.reshape(1,1,self.ns)
        return M1,M2
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
    def apply_rotation(self,phi0,v):
        nsite = v.shape[0]
        phi = [phis.copy() for phis in phi0]
        for p,d in zip(self.ps,self.ds):
            s,ps = p//nsite,p%nsite
            vec = np.take(v,ps,axis=1)
            phi[s] += d*np.outer(vec,np.dot(vec,phi0[s]))
        return phi

