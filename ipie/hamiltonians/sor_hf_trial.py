import numpy as np
import scipy,itertools,plum
from ipie.hamiltonians.sor_base import * 
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF

@plum.dispatch
def walkers2uhf(walkers:UHFWalkers):
    return np.array([walkers.phia.real,walkers.phib.real])

@plum.dispatch
def walkers2ghf(walkers:UHFWalkers):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    C = np.zeros((nw,nb*2,nu+nd))
    C[:,:nb,:nu] = walkers.phia.real
    C[:,nb:,nu:] = walkers.phib.real
    return C 

@plum.dispatch
def walkers2ghf(walkers:GHFWalkers):
    return walkers.phi.real

@plum.dispatch
def trial2uhf(trial:SingleDet):
    return np.array([trial.psi0a.real,trial.psi0b.real])

@plum.dispatch
def trial2ghf(trial:SingleDet):
    nb = trial.nbasis
    nu,nd = trial.nelec 
    C = np.zeros((nb*2,nu+nd))
    C[:nb,:nu] = trial.psi0a.real
    C[nb:,nu:] = trial.psi0b.real
    return C 

@plum.dispatch
def trial2ghf(trial:SingleDetGHF):
    return trial.psi0.real

def make_full(Daa,Dbb,Dab=None,Dba=None):
    nw,n1,n2 = Daa.shape 
    D = np.zeros((nw,n1*2,n2*2))
    D[:,:n1,:n2] = Daa
    D[:,n1:,n2:] = Dbb
    if Dab is not None:
        D[:,:n1,n2:] = Dab
    if Dba is not None:
        D[:,n1:,:n2] = Dba
    return D

@plum.dispatch
def _ovlp(walkers:UHFWalkers,inv=True):
    C = walkers2uhf(walkers)
    CdC = np.einsum('swxi,swxj->swij',C,C)
    if inv:
        CdC = np.linalg.inv(CdC)
    return C,CdC

@plum.dispatch
def _ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True):
    C = walkers2uhf(walkers)
    B = trial2uhf(trial)
    CdB = np.einsum('swxi,sxj->swij',C,B)
    if inv:
        CdB = np.linalg.inv(CdB)
    return C,B,CdB 

@plum.dispatch
def _ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True):
    nw = walkers.nwalkers
    nu,nd = trial.nelec
    CdB = np.zeros((nw,nu+nd,nu+nd))
    CdB[:,:nu] = np.einsum('wxi,xj->wij',walkers.phia.real,trial.psi0a.real)
    CdB[:,nu:] = np.einsum('wxi,xj->wij',walkers.phib.real,trial.psi0b.real)
    if inv:
        CdB = np.linalg.inv(CdB)
    return CdB

@plum.dispatch
def _ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True):
    CdB = np.einsum('wxi,xj->wij',walkers.phi.real,trial.psi0.real)
    if inv:
        CdB = np.linalg.inv(CdB)
    return CdB

@plum.dispatch
def _D0(walkers:UHFWalkers,ovlp=False):
    C,CdCinv = _ovlp(walkers)
    D = np.einsum('swxi,swij->swxj',C,CdCinv) 
    D = np.einsum('swxj,swyj->swxy',D,C) 
    if not ovlp:
        return D
    return D,1./np.linalg.det(CdCinv)

@plum.dispatch
def _D0(walkers:UHFWalkers,trial:SingleDet,ovlp=False):
    C,B,CdBinv = _ovlp(walkers,trial)
    D = np.einsum('sxi,swij->swxj',B,CdBinv) 
    D = np.einsum('swxj,swyj->swxy',D,C) 
    if not ovlp:
        return D
    return D,1./np.linalg.det(CdBinv)

@plum.dispatch
def _D0(walkers:UHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = _ovlp(walkers,trial)
    tmp = np.einsum('xi,wij->wxj',trial.psi0.real,CdBinv) 

    nw = walkers.nwalkers
    nu,nd = trial.nelec
    nb = trial.nbasis
    D = np.zeros((nw,nb*2,nb*2))
    D[:,:,:nb] = np.einsum('wxj,wyj->wxy',tmp[:,:,:nu],walkers.phia.real)
    D[:,:,nb:] = np.einsum('wxj,wyj->wxy',tmp[:,:,nu:],walkers.phib.real)
    if not ovlp:
        return D
    return D,1./np.linalg.det(CdBinv)

@plum.dispatch
def _D0(walkers:GHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = _ovlp(walkers,trial)
    D = np.einsum('xi,wij->wxj',trial.psi0,CdBinv) 
    D = np.einsum('wxj,wyj->wxy',D,walkers.phi.real) 
    if not ovlp:
        return D
    return D,1./np.linalg.det(CdBinv)

def conjugate_chol_left(U,D,full=False):
    if U is None:
        if len(D.shape)==4 and full:
            return make_full(D[0],D[1])
        return D

    if len(D.shape)==4:
        UD = np.einsum('xp,swxy->swpy',U,D)
        if full:
            UD = make_full(UD[0],UD[1])
        return UD
    nw,_,sh2 = D.shape
    nb,_ = U.shape
    UD = np.zeros((nw,nb*2,sh2))
    UD[:,:nb] = np.einsum('xp,wxy->wpy',U,D[:,:nb])
    UD[:,nb:] = np.einsum('xp,wxy->wpy',U,D[:,nb:])
    return UD

def conjugate_chol_right(U,D,full=False):
    if U is None:
        if len(D.shape)==4 and full:
            return make_full(D[0],D[1])
        return D

    if len(D.shape)==4:
        DU = np.einsum('swxy,yq->swxq',D,U)
        if full:
            DU = make_full(DU[0],DU[1])
        return DU
    nw,sh1,_ = D.shape
    nb,_ = U.shape
    DU = np.zeros((nw,sh1,nb*2))
    DU[:,:,:nb] = np.einsum('wxy,yq->wxq',D[:,:,:nb],U)
    DU[:,:,nb:] = np.einsum('wxy,yq->wxq',D[:,:,nb:],U)
    return DU

def _trace_regularize(D0):
    if len(D0.shape)==3:
        return np.einsum('wxy->w',D0**2)
    return np.einsum('swxy->w',D0**2)

def _rdm_intermediates(U,D,UD,full=True):
    DU = conjugate_chol_right(U,D)
    if len(D.shape)==3:
        UDDtU = np.einsum('wpx,wqx->wpq',UD,UD)
        UDtDU = np.einsum('wxp,wxq->wpq',DU,DU)
        if U is None:
            UDDt = UDDtU
        else:
            UDDt = np.einsum('wpx,wyx->wpy',UD,D)
        UDDDU = np.einsum('wpx,wxq->wpq',UDDt,DU)
        return UDDtU,UDtDU,UDDDU
    UDDtU = np.einsum('swpx,swqx->swpq',UD,UD)
    UDtDU = np.einsum('swxp,swxq->swpq',DU,DU)
    if U is None:
        UDDt = UDDtU
    else:
        UDDt = np.einsum('swpx,swyx->swpy',UD,D)
    UDDDU = np.einsum('swpx,swxq->swpq',UDDt,DU)
    if full:
        UDDtU = make_full(UDDtU[0],UDDtU[1])
        UDtDU = make_full(UDtDU[0],UDtDU[1])
        UDDDU = make_full(UDDDU[0],UDDDU[1])
    return UDDtU,UDtDU,UDDDU

def _compute_trial_ovlp(term,D):
    ns,ds = term.ns,term.ds
    Dj = term.select(D,(1,2))
    M = Dj+np.diag(1./ds).reshape(1,ns,ns)
    det = np.linalg.det(M)
    return det * ds.prod()

def _compute_M(term,D):
    ns,ds = term.ns,term.ds
    # input matrix dim: walker,p,q
    Dj = term.select(D,(1,2))
    M1 = Dj+np.diag(1./ds).reshape(1,ns,ns)
    M1 = np.linalg.inv(M1)

    M2 = np.einsum('wij,wjk->wik',M1,Dj)
    M2 = np.eye(ns).reshape(1,ns,ns) - M2
    return M1,M2*ds.reshape(1,1,ns)

class SORHFTrial(SumOfRotationBase):

    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        D0 = _D0(walkers,trial)
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False
        R0 = None 
        if compute_R0:
            tr0 = _trace_regularize(D0)
            R0 = 1./np.sqrt(1.+self.eps_sq*tr0)

        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        R = None
        if compute_R:
            R = np.ones((self.nkeys,nw))
        for d,terms in enumerate(self.terms):
            U = self.chol_basis[d]
            UD = conjugate_chol_left(U,D0) 
            UDU = conjugate_chol_right(U,UD,full=True) 
            if compute_R:
                P1,P2,P3 = _rdm_intermediates(U,D0,UD)

            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                if not compute_R:
                    ovlp[kix] = _compute_trial_ovlp(term,UDU)
                    continue

                M1,M2 = _compute_M(term,UDU)
                ovlp[kix] = term.ds.prod()/np.linalg.det(M1)

                tr = tr0.copy()
                P1i = term.select(P1,(1,2))
                P2i = term.select(P2,(1,2))
                UDUi = term.select(UDU,(1,2))

                t = np.einsum('wij,wkj->wik',P1i,M1)
                t -= 2*np.einsum('wij,wkj->wik',UDUi,M2)
                t = np.einsum('wij,wjk->wik',t,P2i)
                t = np.einsum('wij,wji->w',t,M1)
                tr += t 

                m = np.einsum('wij,wkj->wik',M2,M2)
                m += 2*M2
                tr += np.einsum('wij,wji->w',P2i,m)

                P3i = term.select(P3,(1,2))
                tr -= 2*np.einsum('wij,wji->w',P3i,M1)

                R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

    @plum.dispatch
    def update_workers(self,keys,walkers:UHFWalkers):
        for w,(d,i) in enumerate(keys):
            term = self.terms[d][i] 
            phi = [walkers.phia[w],walkers.phib[w]]
            phi = term.apply_rotation(phi,self.chol_basis[d])
            walkers.phia[w] = phi[0]
            walkers.phib[w] = phi[1]

    def _get_trial_ovlp_ratio(self,walkers,trial):
        C = walkers2ghf(walkers)
        B = trial2ghf(trial)
        BdC = np.einsum('xi,wxj->wij',B,C)
        
        BdCinv = np.linalg.inv(BdC)
        D = np.einsum('wxi,wij->wxj',C,BdCinv)
        D = np.einsum('wxi,yi->wxy',D,B)
        tr = np.einsum('wxy->w',D**2)
        R0 = 1./np.sqrt(1.+self.eps_sq*tr)

        detBdC = np.linalg.det(BdC)
        nw = walkers.nwalkers
        nb = trial.nbasis 
        ovlp = np.zeros((self.nkeys,nw)) 
        R = np.ones((self.nkeys,nw))
        for d,terms in enumerate(self.terms):
            v = self.chol_basis[d]
            if v is None:
                v = np.eye(self.nbasis)
            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                U = term.get_rotation_matrix(v)
                Ufull = np.eye(nb*2)
                if U[0] is not None:
                    Ufull[:nb,:nb] = U[0]
                if U[1] is not None:
                    Ufull[nb:,nb:] = U[1]
                C_ = np.einsum('xy,wyi->wxi',Ufull,C) 
                BdC = np.einsum('xi,wxj->wij',B,C_)
                ovlp[kix] = np.linalg.det(BdC)/detBdC 

                BdCinv = np.linalg.inv(BdC)
                D = np.einsum('wxi,wij->wxj',C_,BdCinv)
                D = np.einsum('wxi,yi->wxy',D,B)
                tr = np.einsum('wxy->w',D**2)
                R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

