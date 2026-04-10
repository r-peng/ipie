import numpy as np
import scipy,itertools
def quadratic2MB(M,basis,spin,thresh=1e-6):
    from ipie.hamiltonians.bitstring_utils import string_act 
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
    #def _trace2_uhf(self,Aconj1,typ1,Aconj2,typ2):
    #    tr = 0 
    #    for s,info in enumerate(self.info):
    #        if info is None:
    #            continue
    #        ps = info[0]
    #        mat_j = Aconj1[s][:,ps][:,:,ps]
    #        mat = np.einsum('wij,wjk->wik',mat_j,self.M[typ1,s])
    #        mat_j = Aconj2[s][:,ps][:,:,ps]
    #        mat = np.einsum('wij,wjk->wik',mat,mat_j)
    #        mat = np.einsum('wij,wji->w',mat,self.M[typ2,s])
    #        tr += mat 
    #    return tr 
    #def _get_D2(self,CCdU,U,diag=True):
    #    mat_j = CCdU[:,:,self.ps]
    #    mat = np.einsum('wxi,wij->wxj',mat_j,self.M[1])
    #    mat_j = U[:,self.ps]
    #    if diag:
    #        return np.einsum('wxi,xi->wx',mat,mat_j)
    #    else:
    #        return np.einsum('wxi,yi->wxy',mat,mat_j)
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
