import numpy as np
import plum
from ipie.hamiltonians.sor_base import SumOfRotationBase
from ipie.hamiltonians.walkers_utils import (
        walkers2uhf,
        walkers2ghf,
        conjugate_chol_left,
        conjugate_chol_right,
        make_full,
)
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF

@plum.dispatch
def trial2uhf(trial:SingleDet):
    return xp.stack((trial.psi0a.real,trial.psi0b.real))

@plum.dispatch
def trial2ghf(trial:SingleDet):
    nb = trial.nbasis
    nu,nd = trial.nelec 
    C = xp.zeros((nb*2,nu+nd))
    C[:nb,:nu] = trial.psi0a.real
    C[nb:,nu:] = trial.psi0b.real
    return C 

@plum.dispatch
def trial2ghf(trial:SingleDetGHF):
    return trial.psi0.real

@plum.dispatch
def _ovlp(walkers:UHFWalkers,inv=True):
    C = walkers2uhf(walkers)
    CdC = xp.einsum('swxi,swxj->swij',C,C)
    if inv:
        CdC = xp.linalg.inv(CdC)
    return C,CdC

@plum.dispatch
def _ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True):
    C = walkers2uhf(walkers)
    B = trial2uhf(trial)
    CdB = xp.einsum('swxi,sxj->swij',C,B)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return C,B,CdB 

@plum.dispatch
def _ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True):
    nw = walkers.nwalkers
    nu,nd = trial.nelec
    CdB = xp.zeros((nw,nu+nd,nu+nd))
    CdB[:,:nu] = xp.einsum('wxi,xj->wij',walkers.phia.real,trial.psi0a.real)
    CdB[:,nu:] = xp.einsum('wxi,xj->wij',walkers.phib.real,trial.psi0b.real)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

@plum.dispatch
def _ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True):
    CdB = xp.einsum('wxi,xj->wij',walkers.phi.real,trial.psi0.real)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

@plum.dispatch
def _D0(walkers:UHFWalkers,ovlp=False):
    C,CdCinv = _ovlp(walkers)
    D = xp.einsum('swxi,swij->swxj',C,CdCinv) 
    D = xp.einsum('swxj,swyj->swxy',D,C) 
    if not ovlp:
        return D
    return D,1./xp.linalg.det(CdCinv)

@plum.dispatch
def _D0(walkers:UHFWalkers,trial:SingleDet,ovlp=False):
    C,B,CdBinv = _ovlp(walkers,trial)
    D = xp.einsum('sxi,swij->swxj',B,CdBinv) 
    D = xp.einsum('swxj,swyj->swxy',D,C) 
    if not ovlp:
        return D
    return D,1./xp.linalg.det(CdBinv)

@plum.dispatch
def _D0(walkers:UHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = _ovlp(walkers,trial)
    tmp = xp.einsum('xi,wij->wxj',trial.psi0.real,CdBinv) 

    nw = walkers.nwalkers
    nu,nd = trial.nelec
    nb = trial.nbasis
    D = xp.zeros((nw,nb*2,nb*2))
    D[:,:,:nb] = xp.einsum('wxj,wyj->wxy',tmp[:,:,:nu],walkers.phia.real)
    D[:,:,nb:] = xp.einsum('wxj,wyj->wxy',tmp[:,:,nu:],walkers.phib.real)
    if not ovlp:
        return D
    return D,1./xp.linalg.det(CdBinv)

@plum.dispatch
def _D0(walkers:GHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = _ovlp(walkers,trial)
    D = xp.einsum('xi,wij->wxj',trial.psi0,CdBinv) 
    D = xp.einsum('wxj,wyj->wxy',D,walkers.phi.real) 
    if not ovlp:
        return D
    return D,1./xp.linalg.det(CdBinv)

def _trace_regularize(D0):
    if len(D0.shape)==3:
        return xp.einsum('wxy->w',D0**2)
    return xp.einsum('swxy->w',D0**2)

def _rdm_intermediates(U,D,UD,full=True):
    DU = conjugate_chol_right(U,D)
    if len(D.shape)==3:
        UDDtU = xp.einsum('wpx,wqx->wpq',UD,UD)
        UDtDU = xp.einsum('wxp,wxq->wpq',DU,DU)
        if U is None:
            UDDt = UDDtU
        else:
            UDDt = xp.einsum('wpx,wyx->wpy',UD,D)
        UDDDU = xp.einsum('wpx,wxq->wpq',UDDt,DU)
        return UDDtU,UDtDU,UDDDU
    UDDtU = xp.einsum('swpx,swqx->swpq',UD,UD)
    UDtDU = xp.einsum('swxp,swxq->swpq',DU,DU)
    if U is None:
        UDDt = UDDtU
    else:
        UDDt = xp.einsum('swpx,swyx->swpy',UD,D)
    UDDDU = xp.einsum('swpx,swxq->swpq',UDDt,DU)
    if full:
        UDDtU = make_full(UDDtU[0],UDDtU[1])
        UDtDU = make_full(UDtDU[0],UDtDU[1])
        UDDDU = make_full(UDDDU[0],UDDDU[1])
    return UDDtU,UDtDU,UDDDU

def _batched_M1(Dj,ds):
    n,r = ds.shape
    idx = xp.arange(r)
    M = Dj.copy()
    M[:,:,idx,idx] += (1./ds)[:,None,:]
    return M

def _batched_trial_ovlp(Dj,ds):
    M = _batched_M1(Dj,ds)
    det = xp.linalg.det(M)
    return det * ds.prod(axis=1)[:,None]

def _batched_M(Dj,ds):
    M1 = _batched_M1(Dj,ds)
    M1 = xp.linalg.inv(M1)

    n,r = ds.shape
    idx = xp.arange(r)
    M2 = -xp.einsum('nwij,nwjk->nwik',M1,Dj)
    M2[:,:,idx,idx] += xp.ones(r) 
    return M1,M2*ds[:,None,None,:]

class SORHFTrial(SumOfRotationBase):

    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        D0 = _D0(walkers,trial)
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False
        R0 = None 
        if compute_R0:
            tr0 = _trace_regularize(D0)
            R0 = 1./xp.sqrt(1.+self.eps_sq*tr0)

        nw = walkers.nwalkers
        ovlp = xp.zeros((self.nkeys,nw)) 
        R = None
        if compute_R:
            R = xp.ones((self.nkeys,nw))
        for d,batch in enumerate(self.batches):
            U = batch.chol_basis
            UD = conjugate_chol_left(U,D0) 
            UDU = conjugate_chol_right(U,UD,full=True) 
            if compute_R:
                P1,P2,P3 = _rdm_intermediates(U,D0,UD)
            for r in batch.rs:
                kix = batch.kix[r]
                ps = batch.p[r]
                ds = batch.d[r]
                udu = xp.stack([UDU[:,pi][:,:,pi] for pi in ps])
                if not compute_R:
                    ovlp[kix] = _batched_trial_ovlp(udu,ds)
                    continue

                M1,M2 = _batched_M(udu,ds)
                ovlp[kix] = ds.prod(axis=1)[:,None]/xp.linalg.det(M1)

                p1 = xp.stack([P1[:,pi][:,:,pi] for pi in ps])
                p2 = xp.stack([P2[:,pi][:,:,pi] for pi in ps])

                t = xp.einsum('nwij,nwkj->nwik',p1,M1)
                t -= 2*xp.einsum('nwij,nwkj->nwik',udu,M2)
                t = xp.einsum('nwij,nwjk->nwik',t,p2)
                t = xp.einsum('nwij,nwji->nw',t,M1)
                tr = tr0[None,:] + t 

                m = xp.einsum('nwij,nwkj->nwik',M2,M2)
                m += 2*M2
                tr += xp.einsum('nwij,nwji->nw',p2,m)

                p3 = xp.stack([P3[:,pi][:,:,pi] for pi in ps])
                tr -= 2*xp.einsum('nwij,nwji->nw',p3,M1)

                R[kix] = 1./xp.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

    def _get_trial_ovlp_ratio(self,walkers,trial):
        print('called')
        C = walkers2ghf(walkers)
        B = trial2ghf(trial)
        BdC = xp.einsum('xi,wxj->wij',B,C)
        
        BdCinv = xp.linalg.inv(BdC)
        D = xp.einsum('wxi,wij->wxj',C,BdCinv)
        D = xp.einsum('wxi,yi->wxy',D,B)
        tr = xp.einsum('wxy->w',D**2)
        R0 = 1./xp.sqrt(1.+self.eps_sq*tr)

        detBdC = xp.linalg.det(BdC)
        nw = walkers.nwalkers
        nb = trial.nbasis 
        ovlp = xp.zeros((self.nkeys,nw)) 
        R = xp.ones((self.nkeys,nw))
        for d,batch in enumerate(self.batches):
            for r in batch.rs:
                for i in range(batch.p[r].shape[0]):
                    kix = self.key_map[d,r,i]
                    U = batch.get_rotation_matrix(r,i)
                    Ufull = xp.eye(nb*2)
                    if U[0] is not None:
                        Ufull[:nb,:nb] = U[0]
                    if U[1] is not None:
                        Ufull[nb:,nb:] = U[1]
                    C_ = xp.einsum('xy,wyi->wxi',Ufull,C) 
                    BdC = xp.einsum('xi,wxj->wij',B,C_)
                    ovlp[kix] = xp.linalg.det(BdC)/detBdC 

                    BdCinv = xp.linalg.inv(BdC)
                    D = xp.einsum('wxi,wij->wxj',C_,BdCinv)
                    D = xp.einsum('wxi,yi->wxy',D,B)
                    tr = xp.einsum('wxy->w',D**2)
                    R[kix] = 1./xp.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

    def _update_walkers_slow(self,kixs,walkers):
        keys = [self.keys[int(kix)] for kix in to_host(kixs)]
        nb = self.nbasis
        C = walkers2ghf(walkers)
        for w,(d,r,i) in enumerate(keys):
            batch = self.batches[d] 
            U = batch.get_rotation_matrix(r,i)
            Ufull = xp.eye(nb*2)
            if U[0] is not None:
                Ufull[:nb,:nb] = U[0]
            if U[1] is not None:
                Ufull[nb:,nb:] = U[1]
            C[w] = xp.einsum('xy,yi->xi',Ufull,C[w]) 
        return C
