import numpy as np
import scipy,itertools,plum
from ipie.hamiltonians.generic_base import GenericBase
from ipie.hamiltonians.sor_base import Udiag  
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

def _conjugate_h1_uhf(UD,DU,U,A,full=False):
    ADU = np.einsum('xy,sdwyq->sdwxq',A,DU) 
    A1 = np.einsum('sdwpx,sdwxq->sdwpq',UD,ADU)
    A2 = np.einsum('dxp,sdwxq->sdwpq',U,ADU)
    if full:
        A1 = _make_full(A1[0],A1[1])
        A2 = _make_full(A2[0],A2[1])
    return A1,A2

def _trace_h1(D,A):
    if len(D.shape)==4:
        return np.einsum('swxy,yx->w',D,A)
    nb = A.shape[0]
    tr = np.einsum('wxy,yx->w',D[:,:nb,:nb],A)
    tr += np.einsum('wxy,yx->w',D[:,nb:,nb:],A)
    return tr

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

    def _update_gf1_uhf(self,gf,D0,UD,DU,M,A,sq=False,fac=1):
        tr0 = _trace_h1(D0,A)
        A1,A2 = _conjugate_h1_uhf(UD,DU,self.chol_basis,A,full=True)
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                Ai1 = term.select(A1[d],(1,2))
                Ai2 = term.select(A2[d],(1,2))
                M1,M2 = M[d,i]
                tr = tr0 - np.einsum('wij,wji->w',Ai1,M1)
                tr += np.einsum('wij,wji->w',Ai2,M2)
                if sq:
                    tr = tr**2
                if (d,i) not in gf:
                    gf[d,i] = 0
                gf[d,i] += tr*fac 
        return gf,A1,A2

    @plum.dispatch
    def calc_bare_gf(self,walkers:UHFWalkers):
        # basis indexing: 
        # \delta-basis: p,q,...
        # ham basis: x,y,...
        # walker basis: i,j,...

        # compute (U^\delta)^\dagger C for all \delta
        D0,D0_ovlp = _D0(walkers,ovlp=True)
        DU = _conjugate_1rdm_right(self.chol_basis,D0) 
        UD = DU.transpose(0,1,2,4,3)
        UDU = _conjugate_1rdm_both(self.chol_basis,UD=UD,full=True) 

        D0_ovlp = D0_ovlp.prod(axis=0)
        M = dict()
        ovlp = dict()
        # compute M,detX
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                M1,M2 = term.compute_M(UDU[d])
                M[d,i] = M1,M2
                ovlp[d,i] = term.ds.prod()/np.linalg.det(M1)

        # compute H-expectation value 
        gf = dict()
        gf = self._update_gf1_uhf(gf,D0,UD,DU,M,self.H1)[0]
        gf = self._update_gf_2body_uhf(gf,D0,UD,DU,M)
        # compute det(X)(1.-H/\Lambda)/scale
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                #gf[d,i] = ovlp[d,i]*D0_ovlp*(self.Lambda - gf[d,i])
                gf[d,i] = ovlp[d,i]*D0_ovlp*(1.-gf[d,i]/self.Lambda)/self.scale
        return gf

    def calc_trial_ovlp(self,walkers,trial):
        D0 = _D0(walkers,trial)
        UD = _conjugate_1rdm_left(self.chol_basis,D0) 
        UDU = _conjugate_1rdm_both(self.chol_basis,UD=UD,full=True) 

        R0 = 1.
        ovlp = dict()
        if self.eps_sq is None:
            for d,terms in enumerate(self.terms):
                for i,term in enumerate(terms):
                    ovlp[d,i] = term.compute_trial_ovlp(UDU[d])
            return ovlp,R0

        M = dict()
        # compute M,detX
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                M1,M2 = term.compute_M(UDU[d])
                M[d,i] = M1,M2
                ovlp[d,i] = term.ds.prod()/np.linalg.det(M1)

        tr0 = _trace_regularize(D0)
        R0 = 1./np.sqrt(1.+self.eps_sq*tr0)
        #print(f'RANK={RANK},Rmean={np.mean(self.R0)}')

        P1,P2,P3 = _conjugate_regularize(self.chol_basis,D0,UD)
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

                R = 1./np.sqrt(1.+self.eps_sq*tr)
                ovlp[d,i] *= R0/R
        return ovlp,R0

    def sample_from_gf(self,gf,ovlp):
        p = np.array([gf[key]*ovlp[key] for key in self.keys]) 
        sign = np.sign(p)
        s = sign.flatten()
        nminus = len(s[s<-0.5])
        if nminus>0:
            print('number of minus=',nminus)
            #exit()

        p = np.fabs(p)
        b = p.sum(axis=0)
        nwalker = b.size
        p /= b.reshape(1,nwalker)
        keys = [None] * nwalker
        for w in range(nwalker):
            kix = np.random.choice(self.nkeys,p=p[:,w])
            keys[w] = self.keys[kix]
            b[w] *= sign[kix,w] 
        return keys,b
    
    @plum.dispatch
    def update_workers(self,keys,walkers:UHFWalkers):
        for w,(d,i) in enumerate(keys):
            term = self.terms[d][i] 
            U = term.get_rotation_matrix(self.chol_basis[d])
            if U[0] is not None:
                walkers.phia[w] = np.dot(U[0],walkers.phia[w])
            if U[1] is not None:
                walkers.phib[w] = np.dot(U[1],walkers.phib[w])

    # NOT USED
    #def compute_eloc_from_sor(self,walkers,trial,thresh=1e-10):
    #    C = self._pack_walkers(walkers,thresh=thresh)
    #    UdC = np.einsum('dxp,swxi->dswpi',self.chol_basis,C)
    #    ovlp = self.calc_trial_ovlp(trial,C,UdC,thresh=thresh)
    #    eloc = 0
    #    for d,terms in enumerate(self.terms):
    #        for i,term in enumerate(terms):
    #            eloc += term.ai*ovlp[d,i]
    #    return self.Lambda-eloc,np.zeros(1),np.zeros(1)

    def local_energy(self,walkers,trial):
        D = _D0(walkers,trial)
        tr0 = _trace_regularize(D)
        R0 = 1.
        if self.eps_sq is not None:
            R0 = 1./np.sqrt(1.+self.eps_sq*tr0)
        if len(D.shape)==3:
            nb = self.nbasis
            Daa,Dbb,Dab,Dba = D[:,:nb,:nb],D[:,nb:,nb:],D[:,:nb,nb:],D[:,nb:,:nb]
        else:
            Daa,Dbb = D[0],D[1]
            Dab = Dba = None
        E1 = self._compute_eloc1_from_1rdm(Daa,Dbb)
        E2 = self._compute_eloc2_from_1rdm(Daa,Dbb,Dab,Dba) 
        return E1+E2,E1,E2,R0

    def _compute_eloc1_from_1rdm(self,Daa,Dbb):
        E1 = np.einsum('wxy,xy->w',Daa,self.H1)
        E1 += np.einsum('wxy,xy->w',Dbb,self.H1)
        return E1

    @plum.dispatch
    def _get_walker_1rdms(self,walkers:UHFWalkers):
        C = _walkers2uhf(walkers)
        rdm1 = dict()
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                U = term.get_rotation_matrix(self.chol_basis[d])
                D = [None] * 2 
                for s,u in enumerate(U):
                    if u is None:
                        C_ = C[s].copy()
                    else:
                        C_ = np.einsum('xy,wyi->wxi',u,C[s]) 
                    X = np.einsum('wxi,wxj->wij',C_,C[s]) 
                    Xinv = np.linalg.inv(X)
                    D[s] = np.einsum('wxi,wij->wxj',C[s],Xinv)
                    D[s] = np.einsum('wxi,wyi->wxy',D[s],C_)
                rdm1[d,i] = np.array(D)
        return rdm1

    @plum.dispatch
    def _get_walker_ovlp(self,walkers:UHFWalkers):
        C = _walkers2uhf(walkers)
        CdC = np.einsum('swxi,swxj->swij',C,C)
        #detCdC = np.linalg.det(CdC)
        ovlp = dict() 
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                U = term.get_rotation_matrix(self.chol_basis[d])
                det = 1.
                for s,u in enumerate(U):
                    if u is None:
                        continue
                    C_ = np.einsum('xy,wyi->wxi',u,C[s]) 
                    CdC = np.einsum('wxi,wxj->wij',C[s],C_)
                    det *= np.linalg.det(CdC)#/detCdC[s] 
                ovlp[d,i] = det
        return ovlp

    def _get_trial_ovlp(self,walkers,trial):
        C = _walkers2ghf(walkers)
        B = _trial2ghf(trial)
        BdC = np.einsum('xi,wxj->wij',B,C)
        
        BdCinv = np.linalg.inv(BdC)
        D = np.einsum('wxi,wij->wxj',C,BdCinv)
        D = np.einsum('wxi,yi->wxy',D,B)
        tr = np.einsum('wxy->w',D**2)
        R0 = 1./np.sqrt(1.+self.eps_sq*tr)

        detBdC = np.linalg.det(BdC)
        ovlp = dict() 
        nb = trial.nbasis 
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
                ovlp[d,i] = np.linalg.det(BdC)/detBdC 

                BdCinv = np.linalg.inv(BdC)
                D = np.einsum('wxi,wij->wxj',C_,BdCinv)
                D = np.einsum('wxi,yi->wxy',D,B)
                tr = np.einsum('wxy->w',D**2)
                R = 1./np.sqrt(1.+self.eps_sq*tr)
                ovlp[d,i] *= R0/R
        return ovlp,R0

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
    def get_decomposition(self,h1e,eri,thresh=1e-6,cmax=None):
        if cmax is None:
            thresh = 1e-6 
            s,chol = np.linalg.eigh(eri.reshape((self.nbasis**2,)*2)) 
            s = s[s>thresh]
            chol = chol[:,-len(s):].T
        else:
            s = None
        nchol = chol.shape[0]
        chol = chol.reshape(nchol,self.nbasis,self.nbasis)
        self.chol_eigvecs = []

        chols = [h1e] + list(chol)
        for Li,ci in zip(chols,coeffs):
            if np.fabs(ci)<thresh:
                continue
            assert np.linalg.norm(Li.imag)<thresh
            assert np.linalg.norm(Li-Li.T)<thresh
            ei,vi = np.linalg.eigh(Li) 
            print(ei)
            self.chol_eigvecs.append(vi)
            self.chol_eigvals.append(ei)
            self.chol_coeffs.append(ci)

    def _update_gf2(self,gf,Aconj1,typ1,Aconj2,typ2,fac=1):
        for d,terms in enumerate(self.terms):
            for i,term in enumerate(terms):
                tr = term._trace2(Aconj1[d],typ1,Aconj2[d],typ2)
                gf[d,i] += tr*fac
        return gf

    def _update_gf_2body_uhf(self,gf,C,UdC):
        # cholesky 
        for chol in self.chol:
            gf,AC,Aconj1,Aconj2 = self._update_gf1(gf,chol,C,UdC,sq=True)

            AD0A = np.einsum('swxi,swyi->swxy',AC,AC) 
            gf = self._update_gf1(gf,AD0A,C,UdC,fac=2)[0]

            AD0AC = np.einsum('swxy,swyi->swxi',AD0A,self.C) 
            tr = np.einsum('swxi,swxi->w',self.C,AD0A) 
            for (d,i) in gf:
                gf[d,i] -= tr
            AD0A = AD0AC = None
            
            gf = self._update_gf2(gf,Aconj1,0,Aconj1,0)
            gf = self._update_gf2(gf,Aconj2,1,Aconj2,1)
            gf = self._update_gf2(gf,Aconj2,0,Aconj1,1,fac=-2)
        return gf

    def _compute_eloc2_from_1rdm(self,D):
        raise NotImplementedError

class SumOfRotationOnsite(SumOfRotationBase):
    def __init__(self,h1e,U,eps_sq=None):
        self.eps_sq = eps_sq
        self.U = U
        self.scale = 1.
        super().__init__(h1e)

    def get_decomposition(self,at,gu,thresh=1e-6,iprint=0,nelec=None):
        self.chol_basis = []
        self.terms = []
        self.Lambda = 0.
        if RANK>0:
            iprint = 0

        # hopping
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
            self.Lambda += 2*ak
            if iprint>0:
                print(f'band={k},ek={ek},gk={gk}')
            terms.append(Udiag(ak,(k,),(gk,)))
            terms.append(Udiag(ak,(k+self.nbasis,),(gk,)))
        self.terms.append(terms)

        # onsite interaction
        self.chol_basis.append(np.eye(self.nbasis))
        terms = []
        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('a_U=',ai)
        self.Lambda += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda += self.U*sum(nelec)/2 
            if RANK==0:
                eks = np.sort(eks)
                Emax = eks[:nelec[0]].sum()+eks[:nelec[1]].sum()
                Emax += self.U * min(nelec)
                print('Emax upper bound=',Emax)
                Emax = max(eks)*sum(nelec)
                Emax += self.U * min(nelec)
                print('Emax upper bound=',Emax)
        for i in range(self.nbasis): 
            terms.append(Udiag(ai,(i,i+self.nbasis,),(gu,-gu,)))
            terms.append(Udiag(ai,(i,i+self.nbasis,),(-gu,gu,)))
        self.terms.append(terms)

        self.chol_basis = np.array(self.chol_basis)
        self.keys = [(d,i) for d,terms in enumerate(self.terms) for i in range(len(terms))]
        self.nkeys = len(self.keys)

    def _update_gf_2body_uhf(self,gf,D0,UD,DU,M):
        D0_diag = np.diagonal(D0,axis1=2,axis2=3)
        UD = _make_full(UD[0],UD[1])
        DU = _make_full(DU[0],DU[1])
        nb = self.nbasis
        for d,terms in enumerate(self.terms):
            U = np.zeros((nb*2,)*2)
            U[:nb,:nb] = self.chol_basis[d]
            U[nb:,nb:] = self.chol_basis[d]
            for i,term in enumerate(terms):
                UDi = term.select(UD[d],(1,))
                DUi = term.select(DU[d],(2,))
                M1,M2 = M[d,i]
                D1 = np.einsum('wxi,wij->wxj',DUi,M1)
                D1 = np.einsum('wxj,wjx->wx',D1,UDi)

                Ui = term.select(U,(1,))
                D2 = np.einsum('wxi,wij->wxj',DUi,M2)
                D2 = np.einsum('wxi,xi->wx',D2,Ui)
                Da = D0_diag[0]-D1[:,:nb]+D2[:,:nb]
                Db = D0_diag[1]-D1[:,nb:]+D2[:,nb:]
                gf[d,i] += self.U*(Da*Db).sum(axis=1)
        return gf

    def _compute_eloc2_from_1rdm(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = self.U*np.einsum('wii,wii->w',Daa,Dbb)
        if Dab is None:
            return E2
        if Dba is None:
            return E2
        E2 -= self.U*np.einsum('wii,wii->w',Dab,Dba)
        return E2

    def _get_MB_hamiltonian(self,nelecs,thresh=1e-6):
        from ipie.hamiltonians.bitstring_utils import get_all_configs_u11,string_act,count_double_occupancy
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

