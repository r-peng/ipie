import numpy as np
import scipy,itertools,plum
from pfapack import pfaffian as pf
from ipie.hamiltonians.sor_hf_trial import * 
from ipie.trial_wavefunction.single_hfb import SingleHFB

# some helper fxns
def get_bcs_state(h1e,hbcs,thresh=1e-10):
    nsite = h1e.shape[0]

    hbdg = np.zeros((nsite*4,)*2)
    hbdg[:nsite,:nsite] = h1e.copy()
    hbdg[nsite:2*nsite,nsite:2*nsite] = h1e.copy()
    hbdg[2*nsite:3*nsite,2*nsite:3*nsite] = -h1e.T
    hbdg[3*nsite:,3*nsite:] = -h1e.T
    hbdg[:2*nsite,2*nsite:] = hbcs.copy()
    hbdg[2*nsite:,:2*nsite] = -hbcs

    w,v = np.linalg.eigh(hbdg)
    if RANK==0:
        print('bcs energy levels=',w)
    
    U,V = v[:2*nsite,2*nsite:],v[2*nsite:,2*nsite:]
    rho = np.dot(V,V.T)
    n,W = np.linalg.eigh(rho)
    if RANK==0:
        print('occupation=',n)
    nocc = len(n[n>1.-thresh])
    nvir = len(n[n<thresh])
    npair = 2*nsite-nocc-nvir
    assert npair%2==0
    if RANK==0:
        print('number of occupied,pair,virtual=',nocc,npair,nvir)
    A,B = W[:,nvir:nvir+npair],W[:,nvir+npair:]
    assert A.shape[1]==npair
    assert B.shape[1]==nocc
    n = n[nvir:nvir+npair]
    u = np.sqrt(1.-n)
    v = np.sqrt(n)
    if RANK==0:
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
    if RANK==0:
        print('Z symmetry=',np.linalg.norm(Z+Z.T))
        print('Z norm=',np.linalg.norm(Z))
    return Z,A,B,u,v 

@plum.dispatch
def _conjugate_walkers_left(walkers:GHFWalkers,ZU):
    return np.einsum('wxi,xp->wip',walkers.phi.real,ZU)

@plum.dispatch
def _conjugate_walkers_left(walkers:UHFWalkers,ZU):
    nw,nb,_ = walkers.phia.shape
    nu,nd = walkers.nup,walkers.ndown
    _,sh2 = ZU.shape
    t = np.zeros((nw,nu+nd,sh2))
    t[:,:nu] = np.einsum('wxi,xq->wiq',walkers.phia.real,ZU[:nb])
    t[:,nu:] = np.einsum('wxi,xq->wiq',walkers.phib.real,ZU[nb:])
    return t

def _K12(walkers,B):
    sh1,sh2 = B.shape
    CB = _conjugate_walkers_left(walkers,B.reshape(1,sh1,sh2))
    return CB[0]

@plum.dispatch
def _K11(walkers:UHFWalkers,Z):
    CZ = _K12(walkers,Z)
    nw,nocc,_ = CZ.shape
    nb = walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    K = np.zeros((nw,nocc,nocc))
    K[:,:,:nu] = np.einsum('wix,wxj->wij',CZ[:,:,:nb],walkers.phia.real)
    K[:,:,nu:] = np.einsum('wix,wxj->wij',CZ[:,:,nb:],walkers.phib.real)
    return CZ,K

@plum.dispatch
def _K11(walkers:GHFWalkers,Z):
    CZ = _K12(walkers,Z)
    return CZ,np.einsum('wix,wxj->wij',CZ,walkers.phi.real)

@plum.dispatch
def _p0(walkers:UHFWalkers,U):
    C = walkers2uhf(walkers)
    return conjugate_chol_right(U,C.transpose(0,1,3,2),full=True)

@plum.dispatch
def _p0(walkers:GHFWalkers,U):
    C = walkers2ghf(walkers)
    return conjugate_chol_right(U,C.transpose(0,2,1))

def assemble2(K11,K12,K21=None,K22=None,sign=-1):
    nw,r1,c1 = K11.shape
    _,_,c2 = K12.shape
    if K21 is None:
        K21 = sign*K12.transpose(0,2,1)
    _,r2,_ = K21.shape
    r = r1+r2
    c = c1+c2

    K = np.zeros((nw,r,c))
    K[:,:r1,:c1] = K11
    K[:,:r1,c1:] = K12
    K[:,r1:,:c1] = K21 
    if K22 is not None:
        K[:,r1:,c1:] = K22
    return K

def block_inv(K11,K12,return_full=True):
    nw,nocc,no = K12.shape
    K = assemble2(K11,K12)
    Kinv = np.linalg.inv(K)
    if return_full:
        return Kinv
    else:
        return Kinv[:,:nocc,:nocc],Kinv[:,:nocc,nocc:],Kinv[:,nocc:,nocc:]

def _conjugate_b(K,b1,b2=None,symm='skew'):
    t1,p1,h1 = b1
    if b2 is None:
        t2,p2,h2 = None,None,None
    else:
        t2,p2,h2 = b2

    if isinstance(K,tuple):
        k11,k12,k22 = K
    else:
        n1 = t1.shape[2]
        k11 = K[:n1,:n1]
        k12 = K[:n1,n1:]
        k21 = K[n1:,:n1]
        k22 = K[n1:,n1:]

    if symm[:4]=='herm':
        sign = 1 
    elif symm[:4]=='skew':
        sign = -1 
    else:
        sign = None
    
    t1k11 = np.einsum('dwip,wij->dwpj',t1,k11)
    h1k21 = np.einsum('dwpi,wji->dwpj',h1,k12)*sign
    t2_ = t1 if t2 is None else t2
    m11 = np.einsum('dwpj,dwjq->dwpq',t1k11+h1k21,t2_)

    t1k12 = np.einsum('dwip,wij->dwpj',t1,k12)
    h1k22 = np.einsum('dpi,wij->dwpj',h1,k22)
    h2_ = h1 if h2 is None else h2
    m11 += np.einsum('dwpj,dqj->dwpq',t1k12+h1k22,h2_)
    p2_ = p1 if p2 is None else p1
    m12 = np.einsum('dwpj,dwjq->dwpq',t1k11+h1k21,p2_)
    if t2 is None and h2 is None and p2 is None:
        m21 = m12.transpose(0,1,3,2)*sign
    else:
        if t2 is None:
            k11t2 = -t1k11.transpose(0,1,3,2)*sign
        else:
            k11t2 = np.einsum('wij,dwjp->dwip',k11,t2)
        if h2 is None:
            k12h2 = h1k21.transpose(0,1,3,2)*sign
        else:
            k12h2 = np.einsum('wij,dwpj->dwip',k12,h2)
        m21 = np.einsum('dwip,dwiq->dwpq',p1,k11t2+k12h2)

    p1k11 = np.einsum('dwip,wij->dwpj',p1,k11)
    p2_ = p1 if p2 is None else p2
    m22 = np.einsum('dwpj,dwjq->dwpq',p1k11,p2_)
    return m11,m12,m21,m22

@plum.dispatch
def _multiply_walkers_left(walkers:UHFWalkers,T):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    sh2 = T.shape[-1]
    D = np.zeros((nw,nb*2,sh2))
    D[:,:nb] = np.einsum('wpi,wiq->wpq',walkers.phia.real,T[:,:nu])
    D[:,nb:] = np.einsum('wpi,wiq->wpq',walkers.phib.real,T[:,nu:])
    return D

@plum.dispatch
def _multiply_walkers_left(walkers:GHFWalkers,T):
    return np.einsum('wpi,wiq->wpq',walkers.phi.real,T)

def _KY(CZ,B,k11,k12):
    KY = np.einsum('wij,wjp->wip',k11,CZ)
    KY -= np.einsum('wij,pj->wip',k12,B)
    return KY

def _D0(walkers,k11,k12,CZ,B):
    D = _KY(CZ,B,k11,k12)
    return _multiply_walkers_left(walkers,D)

@plum.dispatch
def _multiply_walkers_right(walkers:UHFWalkers,T):
    nw,nb = walkers.nwalkers,walkers.nbasis
    sh1 = T.shape[-1]
    D = np.zeros((nw,sh1,nb*2))
    D[:,:,:nb] = np.einsum('wip,wpj->wij',T[:,:,:nb],walkers.phia.real)
    D[:,:,nb:] = np.einsum('wip,wpj->wij',T[:,:,nb:],walkers.phib.real)
    return D

@plum.dispatch
def _multiply_walkers_right(walkers:GHFWalkers,T):
    return np.einsum('wip,wpj->wij',T,walkers.phi.real)

def _XK(walkers,K11,K12):
    CK11 = _multiply_walkers_left(walkers,K11)
    CK12 = _multiply_walkers_left(walkers,K12)
    return -np.stack([CK11,CK12],axis=2)

def _KXXK(walkers,k11,k12):
    XK = _XK(walkers,k11,k12)
    KXXK = np.einsum('wpi,wpj->wij',XK,XK)
    KXX = _multiply_walkers_right(walkers,XK.transpose(0,2,1))
    return KXXK, KXX

def _KYYK(CZ,B,k11,k12,k22):
    KY1 = _KY(CZ,B,k11,k12)
    KY2 = _KY(CZ,B,-k12.transpose(0,2,1),k22)
    KY = -np.stack([KY1,KY2],axis=1)
    return np.einsum('wip,wjp->wij',KY,KY)

class SORHFBTrial(SORHFTrial):

    def trial_precompute(self,trial):
        nb = trial.nbasis
        no = trial.nocc
        B,Z = trial.psi0

        Z = Z.reshape(1,nb*2,nb*2)
        B = B.reshape(1,nb*2,no)

        self.s1 = np.zeros((self.nchol,nb*2,nb*2))
        self.Z1U = np.zeros((self.nchol,nb*2,nb*2))
        self.h1 = np.zeros((self.nchol,nb*2,no))
        if self.eps_sq is not None:
            self.s2 = np.zeros((self.nchol,nb*2,nb*2))
            self.Z2U = np.zeros((self.nchol,nb*2,nb*2))
            self.h2 = np.zeros((self.nchol,nb*2,no))
            ZB = np.dot(Z[0],B[0]).reshape(1,nb*2,nb*2)
        for d,U in enumerate(self.chol_basis):
            Z1U = conjugate_chol_right(U,Z)
            self.s1[d] = conjugate_chol_left(U,Z1U)[:,0]
            self.Z1U[d] = Z1U[:,0]
            self.h1[d] = conjugate_chol_left(U,B)[:,0]
            if self.eps_sq is None:
                continue

            Z2U = np.einsum('wxy,yp->wxp',Z,self.Z1U[d])
            self.s2[d] = conjugate_chol_left(U,Z2U)[:,0]
            self.Z2U[d] = Z2U[:,0]
            self.h2[d] = conjugate_chol_left(U,ZB)[:,0]
    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        B,Z = trial.psi0
        CZ,k11 = _K11(walkers,Z)
        k12 = _K12(walkers,B)
        Kinv = block_inv(k11,k12) 
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False
        R0 = None
        if compute_R0:
            D0 = _D0(walkers,k11,k12,CZ,B)
            R0 = 1./np.sqrt(1.+self.eps_sq*tr0)

        b1 = t1,p,-self.h1
        m11,m12,m22 = _conjugate_b(k11,k12,k22,p,t1,-self.h1,symm='skew')

        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        for d,terms in enumerate(self.terms):
            t1 =  _conjugate_walkers_left(walkers,self.Z1U[d])
            U = self.chol_basis[d]
            p = _p0(walkers,U)
            b = _assemble2(t1,p,K21=-self.h.T)
            bKinv = np.einsum('w')

            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                ns = term.ns
                s = term.select(self.s[d],(0,1))*np.outer(term.ds,term.ds)
                ds = np.diag(term.ds)
                zero = np.zeros((ns,ns))

                c = np.block([[zero,ds],
                              [-ds,s]]) 
                cinv = np.linalg.inv(c)
                pf_bot = pf.pfaffian(cinv)

                M11 = term.select(m11[d],(1,2))
                M12 = term.select(m12[d],(1,2))
                M22 = term.select(m22[d],(1,2))
                top = assemble2(M11,M12,K22=M22) 
                top += cinv.reshape(1,ns*2,ns*2) 
                top = 0.5 * (top-top.transpose(0,2,1))
                for w in range(nw):
                    ovlp[kix,w] = pf.pfaffian(top[w])/pf_bot

        if not compute_R:
            return ovlp,R0,None

        KXXK,KXX = _KXXK(walkers,k11,k12)
        KYYK = _KYYK(CZ,B,k11,k12,k22)
        KXXKYYK = np.einsum('wij,wjk->wik',KXX,KYYK)
        t2 =  _conjugate_walkers_left(walkers,self.Z2U)

        R = np.ones((self.nkeys,nw))
    def _get_trial_ovlp_ratio(self,walkers,trial):
        B,Z = trial.psi0
        K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        K = assemble2(K11,K12) 
        ovlp0 = np.array([pf.pfaffian(Ki) for Ki in K])

        phia = walkers.phia.copy()
        phib = walkers.phib.copy()
        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                walkers.phia = phia
                walkers.phib = phib
                U = term.get_rotation_matrix(self.chol_basis[d])
                if U[0] is not None:
                    walkers.phia = np.einsum('xy,wyi->wxi',U[0],phia)
                if U[1] is not None:
                    walkers.phib = np.einsum('xy,wyi->wxi',U[1],phib)
                K11 = _K11(walkers,Z)
                K12 = _K12(walkers,B)
                K = assemble2(K11,K12) 
                ovlp[kix] = np.array([pf.pfaffian(Ki) for Ki in K])/ovlp0
        return ovlp,None,None
