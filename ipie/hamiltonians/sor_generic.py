import numpy as np
import scipy,itertools,plum
from ipie.hamiltonians.generic_base import GenericBase
from ipie.hamiltonians.sor_base import * 
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

@plum.dispatch
def _walkers2uhf(walkers:UHFWalkers):
    return np.array([walkers.phia.real,walkers.phib.real])

@plum.dispatch
def _walkers2ghf(walkers:UHFWalkers):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    C = np.zeros((nw,nb*2,nu+nd))
    C[:,:nb,:nu] = walkers.phia.real
    C[:,nb:,nu:] = walkers.phib.real
    return C 

@plum.dispatch
def _walkers2ghf(walkers:GHFWalkers):
    return walkers.phi.real

@plum.dispatch
def _trial2uhf(trial:SingleDet):
    return np.array([trial.psi0a.real,trial.psi0b.real])

@plum.dispatch
def _trial2ghf(trial:SingleDet):
    nb = trial.nbasis
    nu,nd = trial.nelec 
    C = np.zeros((nb*2,nu+nd))
    C[:nb,:nu] = trial.psi0a.real
    C[nb:,nu:] = trial.psi0b.real
    return C 

@plum.dispatch
def _trial2ghf(trial:SingleDetGHF):
    return trial.psi0.real

def _make_full(Daa,Dbb,Dab=None,Dba=None):
    nchol,nw,n1,n2 = Daa.shape 
    D = np.zeros((nchol,nw,n1*2,n2*2))
    D[:,:,:n1,:n2] = Daa
    D[:,:,n1:,n2:] = Dbb
    if Dab is not None:
        D[:,:,:n1,n2:] = Dab
    if Dba is not None:
        D[:,:,n1:,:n2] = Dba
    return D

@plum.dispatch
def _ovlp(walkers:UHFWalkers,inv=True):
    C = _walkers2uhf(walkers)
    CdC = np.einsum('swxi,swxj->swij',C,C)
    if inv:
        CdC = np.linalg.inv(CdC)
    return C,CdC

@plum.dispatch
def _ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True):
    C = _walkers2uhf(walkers)
    B = _trial2uhf(trial)
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

def _conjugate_1rdm_left(U,D,full=False):
    if len(D.shape)==4:
        UD = np.einsum('dxp,swxy->sdwpy',U,D)
        if full:
            UD = _make_full(UD[0],UD[1])
        return UD
    nw = D.shape[0]
    nchol,nb,_ = U.shape
    UD = np.zeros((nchol,nw,nb*2,nb*2))
    UD[:,:,:nb] = np.einsum('dxp,wxy->dwpy',U,D[:,:nb])
    UD[:,:,nb:] = np.einsum('dxp,wxy->dwpy',U,D[:,nb:])
    return UD

def _conjugate_1rdm_right(U,D,full=False):
    if len(D.shape)==4:
        DU = np.einsum('swxy,dyq->sdwxq',D,U)
        if full:
            DU = _make_full(DU[0],DU[1])
        return DU
    nw = D.shape[0]
    nchol,nb,_ = U.shape
    DU = np.zeros((nchol,nw,nb*2,nb*2))
    DU[:,:,:,:nb] = np.einsum('wxy,dyq->dwxq',D[:,:,:nb],U)
    DU[:,:,:,nb:] = np.einsum('wxy,dyq->dwxq',D[:,:,nb:],U)
    return DU

def _conjugate_1rdm_both(U,D=None,UD=None,full=False):
    if UD is None:
        UD = _conjugate_1rdm_left(U,D)
    if len(UD.shape)==5:
        UDU = np.einsum('sdwxy,dyq->sdwxq',UD,U)
        if full:
            UDU = _make_full(UDU[0],UDU[1])
        return UDU
    nw = UD.shape[1]
    nchol,nb,_ = U.shape
    UDU = np.zeros((nchol,nw,nb*2,nb*2))
    UDU[:,:,:,:nb] = np.einsum('dwpy,dyq->dwpq',UD[:,:,:,:nb],U)
    UDU[:,:,:,nb:] = np.einsum('dwpy,dyq->dwpq',UD[:,:,:,nb:],U)
    return UDU

def _trace_regularize(D0):
    if len(D0.shape)==3:
        return np.einsum('wxy->w',D0**2)
    return np.einsum('swxy->w',D0**2)

def _conjugate_regularize(U,D,UD,full=True):
    DU = _conjugate_1rdm_right(U,D)
    if len(D.shape)==3:
        UDDtU = np.einsum('dwpx,dwqx->dwpq',UD,UD)
        UDtDU = np.einsum('dwxp,dwxq->dwpq',DU,DU)
        UDDt = np.einsum('dwpx,wyx->dwpy',UD,D)
        UDDDU = np.einsum('dwpx,dwxq->dwpq',UDDt,DU)
        return UDDtU,UDtDU,UDDDU
    UDDtU = np.einsum('sdwpx,sdwqx->sdwpq',UD,UD)
    UDtDU = np.einsum('sdwxp,sdwxq->sdwpq',DU,DU)
    UDDt = np.einsum('sdwpx,swyx->sdwpy',UD,D)
    UDDDU = np.einsum('sdwpx,sdwxq->sdwpq',UDDt,DU)
    if full:
        UDDtU = _make_full(UDDtU[0],UDDtU[1])
        UDtDU = _make_full(UDtDU[0],UDtDU[1])
        UDDDU = _make_full(UDDDU[0],UDDDU[1])
    return UDDtU,UDtDU,UDDDU

class SumOfRotationBase(GenericBase):

    def parse_decomposition(self):
        self.chol_basis = np.array(self.chol_basis)
        self.keys = [(d,i) for d,terms in enumerate(self.terms) for i in range(len(terms))]
        self.nkeys = len(self.keys)
        self.key_map = {key:kix for kix,key in enumerate(self.keys)} 

        Lambda = self.Lambda1 + self.Lambda2
        self.bare_gf = np.zeros(self.nkeys) 
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                kix = self.key_map[d,i]
                self.bare_gf[kix] = term.ai/Lambda

    def calc_trial_ovlp_ratio(self,walkers,trial,compute_R0=True,compute_R=True):
        D0 = _D0(walkers,trial)
        if self.eps_sq is None:
            compute_R0 = False
            compute_R = False

        R0 = None 
        if compute_R0:
            tr0 = _trace_regularize(D0)
            R0 = 1./np.sqrt(1.+self.eps_sq*tr0)

        UD = _conjugate_1rdm_left(self.chol_basis,D0) 
        UDU = _conjugate_1rdm_both(self.chol_basis,UD=UD,full=True) 

        nw = walkers.nwalkers
        ovlp = np.zeros((self.nkeys,nw)) 
        if not compute_R:
            for d,terms in enumerate(self.terms):
                for i,term in enumerate(terms):
                    kix = self.key_map[d,i]
                    ovlp[kix] = term.compute_trial_ovlp(UDU[d])
            return ovlp,R0,None

        M = dict()
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                M1,M2 = term.compute_M(UDU[d])
                M[d,i] = M1,M2
                kix = self.key_map[d,i]
                ovlp[kix] = term.ds.prod()/np.linalg.det(M1)

        P1,P2,P3 = _conjugate_regularize(self.chol_basis,D0,UD)
        R = np.ones((self.nkeys,nw))
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                tr = tr0.copy()

                P1i = term.select(P1[d],(1,2))
                P2i = term.select(P2[d],(1,2))
                UDUi = term.select(UDU[d],(1,2))
                M1,M2 = M[d,i]

                t = np.einsum('wij,wkj->wik',P1i,M1)
                t -= 2*np.einsum('wij,wkj->wik',UDUi,M2)
                t = np.einsum('wij,wjk->wik',t,P2i)
                t = np.einsum('wij,wji->w',t,M1)
                tr += t 

                m = np.einsum('wij,wkj->wik',M2,M2)
                m += 2*M2
                tr += np.einsum('wij,wji->w',P2i,m)

                P3i = term.select(P3[d],(1,2))
                tr -= 2*np.einsum('wij,wji->w',P3i,M1)

                kix = self.key_map[d,i]
                R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

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
        if nminus>0:
            print('number of minus=',nminus)
            #exit()

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
    
    @plum.dispatch
    def update_workers(self,keys,walkers:UHFWalkers):
        for w,(d,i) in enumerate(keys):
            term = self.terms[d][i] 
            phi = [walkers.phia[w],walkers.phib[w]]
            phi = term.apply_rotation(phi,self.chol_basis[d])
            walkers.phia[w] = phi[0]
            walkers.phib[w] = phi[1]

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
        return E1+E2,E1,E2,R0

    def _get_trial_ovlp_ratio(self,walkers,trial):
        C = _walkers2ghf(walkers)
        B = _trial2ghf(trial)
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
            for i,term in enumerate(terms):
                U = term.get_rotation_matrix(self.chol_basis[d])
                Ufull = np.eye(nb*2)
                if U[0] is not None:
                    Ufull[:nb,:nb] = U[0]
                if U[1] is not None:
                    Ufull[nb:,nb:] = U[1]
                C_ = np.einsum('xy,wyi->wxi',Ufull,C) 
                BdC = np.einsum('xi,wxj->wij',B,C_)
                kix = self.key_map[d,i]
                ovlp[kix] = np.linalg.det(BdC)/detBdC 

                BdCinv = np.linalg.inv(BdC)
                D = np.einsum('wxi,wij->wxj',C_,BdCinv)
                D = np.einsum('wxi,yi->wxy',D,B)
                tr = np.einsum('wxy->w',D**2)
                R[kix] = 1./np.sqrt(1.+self.eps_sq*tr)
        return ovlp,R0,R

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

class SumOfRotationGeneric(SumOfRotationBase):

    def decompose_h2(self,eri,thresh=1e-6,cmax=None):
        return
    #    if cmax is None:
    #        thresh = 1e-6 
    #        s,chol = np.linalg.eigh(eri.reshape((self.nbasis**2,)*2)) 
    #        s = s[s>thresh]
    #        chol = chol[:,-len(s):].T
    #    else:
    #        s = None
    #    nchol = chol.shape[0]
    #    chol = chol.reshape(nchol,self.nbasis,self.nbasis)
    #    self.chol_eigvecs = []

    #    chols = [h1e] + list(chol)
    #    for Li,ci in zip(chols,coeffs):
    #        if np.fabs(ci)<thresh:
    #            continue
    #        assert np.linalg.norm(Li.imag)<thresh
    #        assert np.linalg.norm(Li-Li.T)<thresh
    #        ei,vi = np.linalg.eigh(Li) 
    #        print(ei)
    #        self.chol_eigvecs.append(vi)
    #        self.chol_eigvals.append(ei)
    #        self.chol_coeffs.append(ci)

class SumOfRotationOnsite(SumOfRotationBase):
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

        self.chol_basis.append(np.eye(self.nbasis))
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

    def _get_MB_hamiltonian(self,nelecs,thresh=1e-6):
        basis = get_all_configs_u11((self.nbasis,self.nbasis),nelecs)
        basis_map = {cf:i for i,cf in enumerate(basis)}

        H = np.zeros((len(basis),)*2)
        for ix1,cf1 in enumerate(basis):
            for i,j in itertools.product(range(self.nbasis),repeat=2):
                if np.fabs(self.H1[i,j])<thresh:
                    continue
                for s in (0,1):
                    ops = (2*i+s,'cre'),(2*j+s,'des') 
                    cf2,sign = string_act(cf1,ops)
                    if cf2 is not None:
                        ix2 = basis_map[cf2]
                        H[ix2,ix1] += self.H1[i,j]*sign
            H[ix1,ix1] += self.U*count_double_occupancy(cf1,self.nbasis)
        return H,basis

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
