import numpy as np
import plum,time
from ipie.utils.backend import arraylib as xp
from ipie.hamiltonians.sor_base import SumOfRotationBase
from ipie.hamiltonians.walkers_utils import (
        walkers2uhf,
        walkers2ghf,
        conjugate_chol_left, 
        conjugate_chol_right,
)
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers

# some helper fxns
def get_bcs_state(h1e,hbcs,thresh=1e-10,iprint=0):
    nsite = h1e.shape[0]

    hbdg = np.zeros((nsite*4,)*2)
    hbdg[:nsite,:nsite] = h1e.copy()
    hbdg[nsite:2*nsite,nsite:2*nsite] = h1e.copy()
    hbdg[2*nsite:3*nsite,2*nsite:3*nsite] = -h1e.T
    hbdg[3*nsite:,3*nsite:] = -h1e.T
    hbdg[:2*nsite,2*nsite:] = hbcs.copy()
    hbdg[2*nsite:,:2*nsite] = -hbcs

    w,v = np.linalg.eigh(hbdg)
    if iprint>0:
        print('bcs energy levels=',w)
    
    U,V = v[:2*nsite,2*nsite:],v[2*nsite:,2*nsite:]
    rho = np.dot(V,V.T)
    n,W = np.linalg.eigh(rho)
    if iprint>0:
        print('occupation=',n)
    nocc = len(n[n>1.-thresh])
    nvir = len(n[n<thresh])
    npair = 2*nsite-nocc-nvir
    assert npair%2==0
    if iprint>0:
        print('number of occupied,pair,virtual=',nocc,npair,nvir)
    A,B = W[:,nvir:nvir+npair],W[:,nvir+npair:]
    assert A.shape[1]==npair
    assert B.shape[1]==nocc
    n = n[nvir:nvir+npair]
    u = np.sqrt(1.-n)
    v = np.sqrt(n)
    if iprint>0:
        print('u=',u)
        print('v=',v)

    U = A*u.reshape(1,npair)
    V = np.zeros((npair,)*2)
    for k in range(npair//2):
        V[2*k,2*k+1] = v[2*k]
        V[2*k+1,2*k] = -v[2*k]
    V = np.dot(A,V)
    Z = np.linalg.inv(np.dot(U.T,V.conj()))
    Z = np.dot(V.conj(),np.dot(Z,V.T.conj()))
    if iprint>0:
        print('Z symmetry=',np.linalg.norm(Z+Z.T))
        print('Z norm=',np.linalg.norm(Z))
    return Z,A,B,u,v 

@plum.dispatch
def _conjugate_walkers_left(walkers:GHFWalkers,ZU):
    return xp.einsum('wxi,xp->wip',walkers.phi.real,ZU)

@plum.dispatch
def _conjugate_walkers_left(walkers:UHFWalkers,ZU):
    nw,nb,_ = walkers.phia.shape
    nu,nd = walkers.nup,walkers.ndown
    _,sh2 = ZU.shape
    t = xp.zeros((nw,nu+nd,sh2))
    t[:,:nu] = xp.einsum('wxi,xq->wiq',walkers.phia.real,ZU[:nb])
    t[:,nu:] = xp.einsum('wxi,xq->wiq',walkers.phib.real,ZU[nb:])
    return t

@plum.dispatch
def _K11(walkers:UHFWalkers,Z):
    CZ = _K12(walkers,Z)
    nw,nocc,_ = CZ.shape
    nb = walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    K = xp.zeros((nw,nocc,nocc))
    K[:,:,:nu] = xp.einsum('wix,wxj->wij',CZ[:,:,:nb],walkers.phia.real)
    K[:,:,nu:] = xp.einsum('wix,wxj->wij',CZ[:,:,nb:],walkers.phib.real)
    return CZ,K

@plum.dispatch
def _K11(walkers:GHFWalkers,Z):
    CZ = _K12(walkers,Z)
    return CZ,xp.einsum('wix,wxj->wij',CZ,walkers.phi.real)

def _K12(walkers,B):
    sh1,sh2 = B.shape
    CB = _conjugate_walkers_left(walkers,B)
    return CB

def _parse(K,n1):
    return K[:,:n1,:n1],K[:,:n1,n1:],K[:,n1:,:n1],K[:,n1:,n1:]

def _block(K11,K12,K21,K22=None):
    nw,r1,c1 = K11.shape
    _,_,c2 = K12.shape
    _,r2,_ = K21.shape
    K = xp.zeros((nw,r1+r2,c1+c2))
    K[:,:r1,:c1] = K11
    K[:,:r1,c1:] = K12
    K[:,r1:,:c1] = K21
    if K22 is not None:
        K[:,r1:,c1:] = K22
    return K

def _KY(CZ,B,k11,k12):
    KY = xp.einsum('wij,wjp->wip',k11,CZ)
    KY -= xp.einsum('wij,pj->wip',k12,B)
    return KY

def _D0(walkers,CZ,B,k11,k12):
    D = _KY(CZ,B,k11,k12)
    return _multiply_walkers_left(walkers,D)

@plum.dispatch
def _multiply_walkers_left(walkers:UHFWalkers,T):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    sh2 = T.shape[-1]
    D = xp.zeros((nw,nb*2,sh2))
    D[:,:nb] = xp.einsum('wpi,wiq->wpq',walkers.phia.real,T[:,:nu])
    D[:,nb:] = xp.einsum('wpi,wiq->wpq',walkers.phib.real,T[:,nu:])
    return D

@plum.dispatch
def _multiply_walkers_left(walkers:GHFWalkers,T):
    return xp.einsum('wpi,wiq->wpq',walkers.phi.real,T)

@plum.dispatch
def _multiply_walkers_right(walkers:UHFWalkers,T):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown 
    sh1 = T.shape[1]
    D = xp.zeros((nw,sh1,nu+nd))
    D[:,:,:nu] = xp.einsum('wip,wpj->wij',T[:,:,:nb],walkers.phia.real)
    D[:,:,nu:] = xp.einsum('wip,wpj->wij',T[:,:,nb:],walkers.phib.real)
    return D

@plum.dispatch
def _multiply_walkers_right(walkers:GHFWalkers,T):
    return xp.einsum('wip,wpj->wij',T,walkers.phi.real)

def _KXXK(walkers,k1):
    XK = -_multiply_walkers_left(walkers,k1)
    KXXK = xp.einsum('wpi,wpj->wij',XK,XK)
    KXX = _multiply_walkers_right(walkers,XK.transpose(0,2,1))
    return KXXK, KXX

def _KYYK(CZ,B,k11,k12,k22):
    KY1 = _KY(CZ,B,k11,k12)
    KY2 = _KY(CZ,B,-k12.transpose(0,2,1),k22)
    KY = -xp.concatenate([KY1,KY2],axis=1)
    return xp.einsum('wip,wjp->wij',KY,KY)

def _rdm_intermediates1(walkers,CZ,B,K):
    Kx,KXX = _KXXK(walkers,xp.concatenate([K[0],K[1]],axis=2))
    Ky = _KYYK(CZ,B,K[0],K[1],K[3])
    n1 = CZ.shape[1]
    Kxy = xp.einsum('wij,wjk->wik',KXX,Ky[:,:n1])
    return [_parse(K,n1) for K in (Kx,Ky,Kxy)] 

@plum.dispatch
def _p0(walkers:UHFWalkers,U):
    C = walkers2uhf(walkers)
    return conjugate_chol_right(U,C.transpose(0,1,3,2),full=True)

@plum.dispatch
def _p0(walkers:GHFWalkers,U):
    C = walkers2ghf(walkers)
    return conjugate_chol_right(U,C.transpose(0,2,1))

def _bK(K,t,p,h,compute11=True,compute12=True,compute21=True,compute22=True):
    k11,k12,k21,k22 = K
    m11 = None
    m12 = None
    m21 = None
    m22 = None
    if compute11:
        m11 = xp.einsum('wip,wij->wpj',t,k11)
        m11 += xp.einsum('pi,wij->wpj',h,k21)
    if compute12:
        m12 = xp.einsum('wip,wij->wpj',t,k12)
        m12 += xp.einsum('pi,wij->wpj',h,k22)
    if compute21:
        m21 = xp.einsum('wip,wij->wpj',p,k11)
    if compute22:
        m22 = xp.einsum('wip,wij->wpj',p,k12)
    return m11,m12,m21,m22

def _Kb(K,t,p,h,compute11=True,compute12=True,compute21=True,compute22=True):
    k11,k12,k21,k22 = K
    m11 = None
    m12 = None
    m21 = None
    m22 = None
    if compute11:
        m11 = xp.einsum('wpj,wjq->wpq',k11,t)
        m11 += xp.einsum('wpj,qj->wpq',k12,h)
    if compute12:
        m12 = xp.einsum('wpj,wjq->wpq',k11,p)
    if compute21:
        m21 = xp.einsum('wpj,wjq->wpq',k21,t)
        m21 += xp.einsum('wpj,qj->wpq',k22,h)
    if compute22:
        m22 = xp.einsum('wpj,wjq->wpq',k21,p)
    return m11,m12,m21,m22

def _Kp(K,p,compute11=True,compute21=True):
    k11,k21 = K
    m11 = None
    m21 = None
    if compute11:
        m11 = xp.einsum('wpi,wiq->wpq',k11,p)
    if compute21:
        m21 = xp.einsum('wpi,wiq->wpq',k21,p)
    return m11,None,m21,None 

def _pK(K,p,compute11=True,compute12=True):
    k11,k12 = K
    m11 = None
    m12 = None
    if compute11:
        m11 = xp.einsum('wip,wij->wpj',k11,p)
    if compute12:
        m12 = xp.einsum('wip,wij->wpj',k12,p)
    return m11,m12,None,None

def _rdm_intermediates2(K,K1,Kx,Ky,Kxy,p,t1,t2,h1,h2):
    kmap = dict()
    k = _bK(Kxy,t1,p,h1)
    Kxy11 = _Kb(k,t1,p,h1)

    k = _bK(Kx,t2,p,h2)
    Kx21 = _Kb(k,t1,p,h1)
    Kx22 = _Kb(k,t2,p,h2,compute21=False)

    K12 = _Kb(K1,t2,p,h2)

    k = _bK(Kx,t1,p,h1)
    Kx11 = _Kb(k,t1,p,h1,compute21=False)

    k = _bK(Ky,t1,p,h1)
    Ky11 = _Kb(k,t1,p,h1,compute21=False)
    Ky1p = _Kp((k[0],k[2]),p) 

    K1p = _Kp((K1[0],K1[2]),p) 

    k = _bK(K,t2,p,h2,compute12=False,compute22=False)
    K2p = _Kp((k[0],k[2]),p) 

    k = _Kp((Ky[0],None),p,compute21=False)
    Kypp = _pK((k[0],k[1]),p,compute12=False) 
    return Kxy11,Kx21,Kx22,K12,Kx11,Ky11,Ky1p,K1p,K2p,Kypp 

def _batch_select(K,ixs,ps):
    ls = [None] * 4
    for ix in ixs:
        Kix = K[ix]
        ls[ix] = xp.stack([Kix[:,pi][:,:,pi] for pi in ps]) 
    return ls

def _batch_block(ls,sign=-1):
    n,nw,r,_ = ls[0].shape
    k = xp.zeros((n,nw,r*2,r*2))
    k[:,:,:r,:r] = ls[0]
    k[:,:,:r,r:] = ls[1]
    if ls[2] is None:
        k[:,:,r:,:r] = ls[1].transpose(0,1,3,2)*sign
    else:
        k[:,:,r:,:r] = ls[2]
    k[:,:,r:,r:] = ls[3]
    return k

def _pf1(K):
    nax = len(K.shape)
    if nax==3:
        a = K[:,0,1]
    elif nax==4:
        a = K[:,:,0,1]
    else:
        raise ValueError
    return a 
    
def _pf2(K):
    nax = len(K.shape)
    if nax==3:
        a = K[:,0,1]
        b = K[:,0,2]
        c = K[:,0,3]
        d = K[:,1,2]
        e = K[:,1,3]
        f = K[:,2,3]
    elif nax==4:
        a = K[:,:,0,1]
        b = K[:,:,0,2]
        c = K[:,:,0,3]
        d = K[:,:,1,2]
        e = K[:,:,1,3]
        f = K[:,:,2,3]
    else:
        raise ValueError
    return a*f-b*e+d*c

def _pf(K,r):
    if r==1:
        return _pf1(K)
    elif r==2:
        return _pf2(K)
    else:
        raise ValueError

class SORHFBTrial(SumOfRotationBase):

    def trial_precompute(self,trial,iprint=0):
        trial.cast_to_cupy()
        t0 = time.time()

        nb = trial.nbasis
        no = trial.nocc
        nchol = len(self.batches)
        Z = trial.Z.reshape(1,nb*2,nb*2)
        B = trial.B.reshape(1,nb*2,no)

        if self.eps_sq is not None:
            ZB = xp.dot(Z[0],B[0]).reshape(1,nb*2,no)
        for d,batch in enumerate(self.batches):
            U = batch.chol_basis
            batch.h1 = conjugate_chol_left(U,B)[0]
            Z1U = conjugate_chol_right(U,Z)
            batch.Z1U = Z1U[0]
            s1 = conjugate_chol_left(U,Z1U)[0]
            batch.c1 = dict()
            for r in batch.rs:
                ps = batch.p[r]
                ds = batch.d[r]
                n = ds.shape[0]
                idx = xp.arange(r)

                c1 = xp.zeros((n,r*2,r*2))
                c1[:,r:,r:] = xp.stack([s1[pi][:,pi] for pi in ps])*ds[:,:,None]*ds[:,None,:]
                c1[:,idx,idx+r] = ds
                c1[:,idx+r,idx] = -ds
                c1 = xp.linalg.inv(c1)
                c1 = 0.5 * (c1-c1.transpose(0,2,1))
                pf_bot = _pf(c1,r)
                batch.c1[r] = c1,pf_bot
            if self.eps_sq is None:
                continue

            batch.h2 = conjugate_chol_left(U,ZB)[0]
            Z2U = xp.einsum('wxy,yp->wxp',Z,batch.Z1U)
            batch.Z2U = Z2U[0]
            s2 = conjugate_chol_left(U,Z2U)[0]
            batch.c2 = dict()
            for r in batch.rs:
                ps = batch.p[r]
                ds = batch.d[r]
                n = ds.shape[0]
                idx = xp.arange(r)

                c2 = xp.zeros((n,r*2,r*2))
                c2[:,r:,r:] = -xp.stack([s2[pi][:,pi] for pi in ps])*ds[:,:,None]*ds[:,None,:]
                c2[:,idx,idx+r] = -ds
                c2[:,idx+r,idx] = -ds
                batch.c2[r] = c2,ds*2+ds**2
        if iprint>0:
            print('precompute time=',time.time()-t0)

    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        B,Z = trial.B,trial.Z
        n1 = walkers.nup + walkers.ndown
        n2 = B.shape[1]
        nb = walkers.nbasis
        nw = walkers.nwalkers

        CZ,K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        K = _block(K11,K12,-K12.transpose(0,2,1))
        K = xp.linalg.inv(K)
        K = _parse(K,n1)
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False
        R0 = None
        if compute_R0:
            D0 = _D0(walkers,CZ,B,K[0],K[1])
            tr0 = xp.einsum('wij->w',D0**2)
            R0 = 1./xp.sqrt(1.+self.eps_sq*tr0)
        R = None
        if compute_R:
            Kx,Ky,Kxy = _rdm_intermediates1(walkers,CZ,B,K)
            R = xp.ones((self.nkeys,nw))

        ovlp = xp.zeros((self.nkeys,nw)) 
        for d,batch in enumerate(self.batches):
            U = batch.chol_basis
            p = _p0(walkers,U)
            t1 =  _conjugate_walkers_left(walkers,batch.Z1U)
            h1 = -batch.h1
            K1 = _bK(K,t1,p,h1)
            K11 = _Kb(K1,t1,p,h1,compute21=False)

            if compute_R:
                t2 =  _conjugate_walkers_left(walkers,batch.Z2U)
                h2 = batch.h2
                ls = _rdm_intermediates2(K,K1,Kx,Ky,Kxy,p,t1,t2,h1,h2)
                Kxy11,Kx21,Kx22,K12,Kx11,Ky11,Ky1p,K1p,K2p,Kypp = ls

            for r in batch.rs:
                kix = batch.kix[r] 
                ps = batch.p[r] 
                c1,pf_bot = batch.c1[r]

                m = _batch_select(K11,(0,1,3),ps)
                m = _batch_block(m,sign=-1)
                m = m + c1[:,None,:,:]
                m = 0.5 * (m-m.transpose(0,1,3,2))
                ovlp[kix] = _pf(m,r)/pf_bot[:,None]

                if not compute_R:
                    continue
                m = xp.linalg.inv(m)
                c2,d2 = batch.c2[r]

                # (0,0,0,1),(0,1,0,0)
                k = _batch_select(Kxy11,(0,1,2,3),ps)
                k = _batch_block(k)
                tr = tr0[None,:] - 2.*xp.einsum('nwij,nwji->nw',k,m)

                # (0,0,1,0)
                k = _batch_select(Kx22,(0,1,3),ps)
                k = _batch_block(k,sign=1)
                tr += xp.einsum('nwij,nji->nw',k,c2)

                # (0,0,1,1),(0,1,1,0)
                k = _batch_select(Kx21,(0,1,2,3),ps)
                k = _batch_block(k)
                tmp = xp.einsum('nwji,njk->nwik',k,c2)
                k = _batch_select(K12,(0,1,2,3),ps)
                k12 = _batch_block(k)
                mk12 = xp.einsum('nwij,nwjk->nwik',m,k12)
                tr -= 2.*xp.einsum('nwij,nwij->nw',tmp,mk12)

                # (0,1,0,1)
                k = _batch_select(Ky11,(0,1,3),ps)
                ky11 = _batch_block(k,sign=1)
                right = -xp.einsum('nwij,nwjk->nwik',m,ky11)
                right = xp.einsum('nwij,nwjk->nwik',right,m) 
                # (0,1,1,1)
                tmp = xp.einsum('nwij,njk->nwik',mk12,c2)
                right += xp.einsum('nwij,nwkj->nwik',tmp,mk12) 
                k = _batch_select(Kx11,(0,1,3),ps)
                k = _batch_block(k,sign=1)
                tr += xp.einsum('nwij,nwji->nw',k,right)

                # (1,0,0,0)
                k = _batch_select(Kypp,[0],ps)[0]
                tr += xp.einsum('nwii,ni->nw',k,d2)

                # (1,0,1,0)
                k = _batch_select(K2p,(0,2),ps)
                k2p = xp.concatenate([k[0],k[2]],axis=2) 
                tmp = xp.einsum('nwji,njk->nwik',k2p,c2) 
                tr += xp.einsum('nwij,nwji,ni->nw',tmp,k2p,d2)

                k = _batch_select(K1p,(0,2),ps)
                k = xp.concatenate([k[0],k[2]],axis=2)
                mk1p = xp.einsum('nwij,nwjk->nwik',m,k)
                # (1,0,0,1),(1,1,0,0)
                k = _batch_select(Ky1p,(0,2),ps)
                left = -2.*xp.concatenate([k[0],k[2]],axis=2) 
                # (1,0,1,1),(1,1,1,0)
                k12c = xp.einsum('nwij,njk->nwik',k12,c2)
                left += 2.*xp.einsum('nwij,nwjk->nwik',k12c,k2p)
                # (1,1,0,1)
                # (1,1,1,1)
                tmp = ky11 + xp.einsum('nwij,nwkj->nwik',k12c,k12)
                left += xp.einsum('nwij,nwjk->nwik',tmp,mk1p)
                tr += xp.einsum('nwji,nwji,ni->nw',left,mk1p,d2)

                R[kix] = 1./xp.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

    def _get_trial_ovlp_ratio(self,walkers,trial):
        from pfapack import pfaffian as pf
        B,Z = trial.B,trial.Z
        n1 = walkers.nup + walkers.ndown
        n2 = B.shape[1]
        nw = walkers.nwalkers

        CZ,K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        K = np.block([[K11,K12],[-K12.transpose(0,2,1),np.zeros((nw,n2,n2))]])
        ovlp0 = np.array([pf.pfaffian(Ki) for Ki in K])

        phia = walkers.phia.copy()
        phib = walkers.phib.copy()
        ovlp = np.zeros((self.nkeys,nw)) 
        R = np.ones((self.nkeys,nw))
        for d,batch in enumerate(self.batches):
            for r in batch.rs:
                for i in range(batch.p[r].shape[0]):
                    kix = self.key_map[d,r,i]
                    walkers.phia = phia
                    walkers.phib = phib
                    U = batch.get_rotation_matrix(r,i)
                    if U[0] is not None:
                        walkers.phia = np.einsum('xy,wyi->wxi',U[0],phia)
                    if U[1] is not None:
                        walkers.phib = np.einsum('xy,wyi->wxi',U[1],phib)
                    CZ,K11 = _K11(walkers,Z)
                    K12 = _K12(walkers,B)
                    K = np.block([[K11,K12],[-K12.transpose(0,2,1),np.zeros((nw,n2,n2))]])
                    ovlp[kix] = np.array([pf.pfaffian(Ki) for Ki in K])/ovlp0

                    K = np.linalg.inv(K)
                    K = _parse(K,n1)
                    D = _D0(walkers,CZ,B,K[0],K[1])
                    tr = np.einsum('wij->w',D**2)
                    R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,None,R
