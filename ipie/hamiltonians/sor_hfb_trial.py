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
    return np.einsum('wxi,dxp->dwip',walkers.phi.real,ZU)

@plum.dispatch
def _conjugate_walkers_left(walkers:UHFWalkers,ZU):
    nw,nb,_ = walkers.phia.shape
    nu,nd = walkers.nup,walkers.ndown
    nchol,_,sh2 = ZU.shape
    t = np.zeros((nchol,nw,nu+nd,sh2))
    t[:,:,:nu] = np.einsum('wxi,dxq->dwiq',walkers.phia.real,ZU[:,:nb])
    t[:,:,nu:] = np.einsum('wxi,dxq->dwiq',walkers.phib.real,ZU[:,nb:])
    return t

def _t0(walkers,ZU):
    return _conjugate_walkers_left(walkers,ZU)

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
    return K

@plum.dispatch
def _K11(walkers:GHFWalkers,Z):
    CZ = _K12(walkers,Z)
    return np.einsum('wix,wxj->wij',CZ,walkers.phi.real)

@plum.dispatch
def _p0(walkers:UHFWalkers,U):
    C = walkers2uhf(walkers)
    return conjugate_chol_right(U,C.transpose(0,1,3,2),full=True)

@plum.dispatch
def _p0(walkers:GHFWalkers,U):
    C = walkers2ghf(walkers)
    return conjugate_chol_right(U,C.transpose(0,2,1))

def assemble2(K11,K12,K21=None,K22=None):
    nw,n1,n2 = K12.shape
    n = n1+n2
    K = np.zeros((nw,n,n))
    K[:,:n1,:n1] = K11
    K[:,:n1,n1:] = K12
    if K21 is None:
        K21 = -K12.transpose(0,2,1)
    K[:,n1:,:n1] = K21 
    if K22 is not None:
        K[:,n1:,n1:] = K22
    return K

def block_inv(K11,K12):
    nw,nocc,no = K12.shape
    K = assemble2(K11,K12)
    #for Ki in K:
    #    _,s,_ = np.linalg.svd(Ki)
    #    print(s)
    Kinv = np.linalg.inv(K)
    return Kinv[:,:nocc,:nocc],Kinv[:,:nocc,nocc:],Kinv[:,nocc:,nocc:]

class SORHFBTrial(SORHFTrial):

    def trial_precompute(self,trial):
        nb = trial.nbasis
        no = trial.nocc
        B,Z = trial.psi0

        Z = Z.reshape(1,nb*2,nb*2)
        ZU = conjugate_chol_right(self.chol_basis,Z)
        self.s = conjugate_chol_right_left(self.chol_basis,ZU)[:,0]
        self.ZU = ZU[:,0]

        B = B.reshape(1,nb*2,no)
        self.h = conjugate_chol_left(self.chol_basis,B)[:,0]

    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        B,Z = trial.psi0
        t = _t0(walkers,self.ZU)
        p = _p0(walkers,self.chol_basis)
        K11 = _K11(walkers,Z)
        K12 = _K12(walkers,B)
        #K = assemble(K11,K12)
        #for w,Ki in enumerate(K12):
        #    print(pf.pfaffian(K[w]))
        #    #u,s,v = np.linalg.svd(Ki.T,full_matrices=True)
        #    #v = v[len(s):].T
        #    #print('Q shape=',v.shape)
        #    #print('K12 kernel=',np.linalg.norm(np.dot(Ki.T,v)))
        #    #z = np.dot(K11[w],v)
        #    #print('K11v=',np.linalg.norm(z))
        #    #_,s,_ = np.linalg.svd(z)
        #    #print('z singular values=',s)
        k11,k12,k22 = block_inv(K11,K12) 
        #exit()
        print(np.linalg.norm(k11+k11.transpose(0,2,1)))
        print(np.linalg.norm(k22+k22.transpose(0,2,1)))
        #exit()

        tk11 = np.einsum('dwip,wij->dwpj',t,k11)
        pk11 = np.einsum('dwip,wij->dwpj',p,k11)
        tk11t = np.einsum('dwpj,dwjq->dwpq',tk11,t)
        tk11p = np.einsum('dwpj,dwjq->dwpq',tk11,p)
        pk11p = np.einsum('dwpj,dwjq->dwpq',pk11,p)
        tk11 = pk11 = None

        k12h = np.einsum('wij,dpj->dwip',k12,self.h)
        tk12h = np.einsum('dwip,dwiq->dwpq',t,k12h)
        pk12h = np.einsum('dwip,dwiq->dwpq',p,k12h)
        k12h = None

        hk22 = np.einsum('dpi,wij->dwpj',self.h,k22)
        hk22h = np.einsum('dwpj,dqj->dwpq',hk22,self.h)
        hk22 = None

        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        for d,terms in enumerate(self.terms):
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

                tkt = term.select(tk11t[d],(1,2))
                tkh = term.select(tk12h[d],(1,2))
                hkh = term.select(hk22h[d],(1,2))
                M11 = tkt - tkh + tkh.transpose(0,2,1) + hkh

                tkp = term.select(tk11p[d],(1,2))
                pkh = term.select(pk12h[d],(1,2))
                M12 = tkp + pkh.transpose(0,2,1)

                M22 = term.select(pk11p[d],(1,2))
                top = assemble2(M11,M12,K22=M22) 
                top += cinv.reshape(1,ns*2,ns*2) 
                top = 0.5 * (top-top.transpose(0,2,1))
                for w in range(nw):
                    ovlp[kix,w] = pf.pfaffian(top[w])/pf_bot
                print(d,i,ovlp[kix])
        return ovlp,None,None

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
                U = term.get_rotation_matrix(self.chol_basis[d])
                if U[0] is not None:
                    walkers.phia = np.einsum('xy,wyi->wxi',U[0],phia)
                if U[1] is not None:
                    walkers.phib = np.einsum('xy,wyi->wxi',U[1],phib)
                K11 = _K11(walkers,Z)
                K12 = _K12(walkers,B)
                K = assemble2(K11,K12) 
                ovlp[kix] = np.array([pf.pfaffian(Ki) for Ki in K])/ovlp0
                print(d,i,ovlp[kix])
        return ovlp,None,None
