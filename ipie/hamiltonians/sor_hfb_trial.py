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

def _K12(walkers,B):
    sh1,sh2 = B.shape
    CB = _conjugate_walkers_left(walkers,B.reshape(1,sh1,sh2))
    return CB[0]

def _parse(K,n1):
    return K[:,:n1,:n1],K[:,:n1,n1:],K[:,n1:,:n1],K[:,n1:,n1:]

def _KY(CZ,B,k11,k12):
    KY = np.einsum('wij,wjp->wip',k11,CZ)
    KY -= np.einsum('wij,pj->wip',k12,B)
    return KY

def _D0(walkers,CZ,B,k11,k12):
    D = _KY(CZ,B,k11,k12)
    return _multiply_walkers_left(walkers,D)

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

def _KXXK(walkers,k1):
    XK = -_multiply_walkers_left(walkers,k1)
    KXXK = np.einsum('wpi,wpj->wij',XK,XK)
    KXX = _multiply_walkers_right(walkers,XK.transpose(0,2,1))
    return KXXK, KXX

def _KYYK(CZ,B,k11,k12,k22):
    KY1 = _KY(CZ,B,k11,k12)
    KY2 = _KY(CZ,B,-k12.transpose(0,2,1),k22)
    KY = -np.stack([KY1,KY2],axis=1)
    return np.einsum('wip,wjp->wij',KY,KY)

def _rdm_intermediates1(walkers,CZ,B,K):
    Kx,KXX = _KXXK(walkers,np.stack([K[0],K[1]],axis=2))
    Ky = _KYYK(CZ,B,K[0],K[1],K[3])
    Kxy = np.einsum('wij,wjk->wik',KXX,Ky)

    n1 = CZ.shape[1]
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
        m11 = np.einsum('wip,wij->wpj',t,k11)
        m11 += np.einsum('pi,wij->wpj',h,k21)
    if compute12:
        m12 = np.einsum('wip,wij->wpj',t,k12)
        m12 += np.einsum('pi,wij->wpj',h,k22)
    if compute21:
        m21 = np.einsum('wip,wij->wpj',p,k11)
    if compute22:
        m22 = np.einsum('wip,wij->wpj',p,k12)
    return m11,m12,m21,m22

def _Kb(K,t,p,h,compute11=True,compute12=True,compute21=True,compute22=True):
    k11,k12,k21,k22 = K
    m11 = None
    m12 = None
    m21 = None
    m22 = None
    if compute11:
        m11 = np.einsum('wpj,wjq->wpq',k11,t)
        m11 += np.einsum('wpj,qj->wpq',k12,h)
    if compute12:
        m12 = np.einsum('wpj,wjq->wpq',k11,p)
    if compute21:
        m21 = np.einsum('wpj,wjq->wpq',k21,t)
        m21 += np.einsum('wpj,qj->wpq',k22,h)
    if compute22:
        m22 = np.einsum('wpj,wjq->wpq',k21,p)
    return m11,m12,m21,m22

def _Kp(K,p,compute11=True,compute21=True):
    k11,k21 = K
    m11 = None
    m21 = None
    if compute11:
        m11 = np.einsum('wpi,wiq->wpq',k11,p)
    if compute21:
        m21 = np.einsum('wpi,wiq->wpq',k21,p)
    return m11,None,m21,None 

def _pK(K,p,commpute11=True,compute12=True):
    k11,k12 = K
    m11 = None
    m12 = None
    if compute11:
        m11 = np.einsum('wip,wij->wpj',k11,p)
    if compute12:
        m11 = np.einsum('wip,wij->wpj',k12,p)
    return m11,m12,None,None

def _rdm_intermediates2(Kmap,p,t1,t2,h1,h2):
    kmap = dict()
    K = _bK(Kmap['Kxy'],t1,p,h1)
    kmap['Kxy11'] = _Kb(K,t1,p,h1)

    K = _bK(Kmap['Kx'],t2,p,h2,n1,n1)
    kmap['Kx21'] = _Kb(K,t1,p,h1)
    kmap['Kx22'] = _Kb(K,t2,p,h2,compute21=False)

    kmap['K12'] = _Kb(Kmap['K1'],t2,p,h2)

    K = _bK(Kmap['Kx'],t1,p,h1)
    kmap['Kx11'] = _Kb(K,t1,p,h1,compute21=False)

    K = _bK(Kmap['Ky'],t1,p,h1,n1,n1)
    kmap['Ky11'] = _Kb(K,t1,p,h1,compute21=False)
    kmap['Ky1p'] = _Kp((K[0],K[2]),p) 

    K = Kmap['K1']
    kmap['K1p'] = _Kp((K[0],K[2]),p) 

    K = _bK(Kmap['K'],t2,p,h2,compute12=False,compute21=False,compute22=False)
    kmap['K2p'] = _Kp((K[0],K[2]),p,compute21=False) 

    K = Kmap['Ky']
    K = _Kp((K[0],None),p,compute21=False)
    kmap['Kypp'] = _pK((K[0],K[1]),p,compute12=False) 
    return kmap 

def _select(K,ixs,term,axes=(1,2)):
    ls = [None] * 4
    for ix in ixs:
        ls[ix] = term.selec(K[ix],axes)
    return ls

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
        n1 = walkers.nup + walkers.ndown
        n2 = B.shape[1]
        nb = walker.nbasis
        nw = walkers.nwalkers

        CZ,K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        K = np.block([[K11,K12],[-K12.transpose(0,2,1),np.zeros((nw,n2,n2))]])
        K = np.linalg.inv(K)
        K = _parse(K,n1)
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False
        R0 = None
        if compute_R0:
            D0 = _D0(walkers,CZ,B,K[0],K[1])
            tr0 = np.einsum('wij->w',D0**2)
            R0 = 1./np.sqrt(1.+self.eps_sq*tr0)
        R = None
        if compute_R:
            Kx,Ky,Kxy = _rdm_intermediates1(walkers,CZ,B,K)
            R = np.ones((self.nkeys,nw))
            Kmap = {'K':K,'Kx':Kx,'Ky':Ky,'Kxy':Kxy}

        ovlp = np.zeros((self.nkeys,nw)) 
        for d,terms in enumerate(self.terms):
            U = self.chol_basis[d]
            p = _p0(walkers,U)
            t1 =  _conjugate_walkers_left(walkers,self.Z1U[d])
            h1 = -self.h1[d]
            s1 = self.s1[d]
            K1 = _bK(K,t1,p,h1,n1,n1)
            K11 = _Kb(K1,t1,p,h1,nb*2,n1,compute21=False)

            if compute_R:
                t2 =  _conjugate_walkers_left(walkers,self.Z2U[d])
                h2 = self.h2[d]
                s2 = self.s2[d]
                Kmap['K1'] = K1
                kmap = _rdm_intermediates2(Kmap,p,t1,t2,h1,h2)

            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                ns = term.ns
                s = term.select(s,(0,1))*np.outer(term.ds,term.ds)
                ds = np.diag(term.ds)

                c1 = np.block([[np.zeros((ns,ns)),ds],[-ds,s1]]) 
                c1 = np.linalg.inv(c1)
                pf_bot = pf.pfaffian(c1)

                m = _select(K11,(0,1,3),term)
                m = np.block([[m[0],m[1]],[-m[1].transpose(0,2,1),m[3]]])
                m = m + c1.reshape(1,ns*2,ns*2)
                m = 0.5 * (top-top.transpose(0,2,1))
                for w in range(nw):
                    ovlp[kix,w] = pf.pfaffian(m[w])/pf_bot

                if not compute_R:
                    continue
                tr = tr0.copy()
                m = np.linalg.inv(m)
                c2 = np.block([[np.zeros((ns,ns)),ds],[ds,s2]]) 
                d2 = np.sqrt(terms.ds*2.+term.ds**2)

                # (0,0,0,1),(0,1,0,0)
                k = _select(kmap['Kxy11'],(0,1,2,3),term)
                k = np.block([[k[0],k[1]],[k[2],k[3]]])
                tr -= 2.*np.einsum('wij,wji->w',k,m)

                # (0,0,1,0)
                k = _select(kmap['Kx22'],(0,1,3),term)
                k = np.block([[k[0],k[1]],[k[1].transpose(0,2,1),k[3]]])
                tr += np.einsum('wij,ji->w',k,c2)

                # (0,0,1,1),(0,1,1,0)
                k = _select(kmap['Kx21'],(0,1,2,3),term)
                k = np.block([[k[0],k[1]],[k[2],k[3]]])
                tmp = np.einsum('wji,jk->wik',k,c2)
                k = _select(kmap['K12'],(0,1,2,3),term)
                k = np.block([[k[0],k[1]],[k[2],k[3]]])
                mK12 = np.einsum('wij,wjk->wik',m,k)
                tr -= 2.*np.einsum('wij,wij->w',tmp,mK12)

                # (0,1,0,1)
                k = _select(kmap['Kx11'],(0,1,3),term)
                Kx11 = np.block([[k[0],k[1]],[k[1].transpose(0,2,1),k[3]]])
                tmp = np.einsum('wij,wkj->wik',Kx11,m)
                k = _select(kmap['Ky11'],(0,1,3),term)
                Ky11 = np.block([[k[0],k[1]],[k[1].transpose(0,2,1),k[3]]])
                tmp = np.einsum('wij,wjk->wik',tmp,Ky11)
                tr += np.einsum('wij,wji->w',tmp,m)

                # (0,1,1,1)
                tmp = np.einsum('wji,wjk->wik',mK12,Kx11)
                tmp = np.einsum('wij,wjk->w',tmp,mK12)
                tr += np.einsum('wij,ji->w',tmp,c2)

                # (1,0,0,0)
                k = term.select(kmap['Kypp'][0],(1,2))
                tr += np.einsum('wii,i->w',k,d2**2)

                # (1,0,0,1)
                k = _select(kmap['Ky1p'],(0,2),term)
                tmp = np.stack([k[0],k[2]],axis=1) * d2.reshape(1,1,ns)
                k = _select(kmap['K1p'],(0,2),term)
                k = np.stack([k[0],k[2]],axis=1)
                mK1pd = np.einsum('wij,wjk,k->wik',m,k,d)
                tr -= 2*np.einsum('wji,wji->w',tmp,mK1p)
                
                # (1,0,1,0)
                k = _select(kmap['K2p'],(0,2),term)
                K2pd = np.stack([k[0],k[2]],axis=1) * d2.reshape(1,1,ns)
                tr += np.einsum('wji,wji->w',K2pd,K2pd)

                # (1,0,1,1),(1,1,1,0)
                k = _select(kmap['K12'],(0,1,2,3),term)
                K12 = np.block([[k[0],k[1]],[k[2],k[3]]])
                K12c = np.einsum('wij,jk->wik',K12,c2)
                tmp = np.einsum('wji,wkj->wik',K2pd,K12c)
                tr -= 2.*np.einsum('wij,wji->w',tmp,mK1pd)

                # (1,1,0,1)
                tmp = np.einsum('wji,wjk->wik',mK1pd,Ky11)
                tr += np.einsum('wij-wji->w',tmp,mK1pd)

                # (1,1,1,1)
                tmp = np.einsum('wji,wjk->wik',mK1pd,K12c)
                tmp = np.einsum('wij,wjk->wik',tmp,K12)
                tr += np.einsum('wij,wji->w',tmp,mK1pd)

                R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

    def _get_trial_ovlp_ratio(self,walkers,trial):
        B,Z = trial.psi0
        n1 = walkers.nup + walkers.ndown
        n2 = B.shape[1]

        K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        K = np.block([[K11,K12],[-K12.transpose(0,2,1),np.zeros((nw,n2,n2))]])
        ovlp0 = np.array([pf.pfaffian(Ki) for Ki in K])

        phia = walkers.phia.copy()
        phib = walkers.phib.copy()
        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        R = np.ones((self.nkeys,nw))
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
