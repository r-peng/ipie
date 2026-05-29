import numpy as np
import scipy,itertools,plum,h5py
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        string_act,
        count_double_occupancy, 
)
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

@plum.dispatch
def walkers2tensor(walkers:UHFWalkers):
    if walkers.ndown==0:
        return walkers.phia
    return xp.concatenate((walkers.phia.real,walkers.phib.real),axis=2)

@plum.dispatch
def walkers2tensor(walkers:GHFWalkers):
    return walkers.phi.real

@plum.dispatch
def tensor2walkers(walkers:UHFWalkers,phi):
    phi = xp.asarray(phi)
    walkers.phia = phi[:,:,:walkers.nup]
    walkers.phib = None
    if walkers.ndown>0:
        walkers.phib = phi[:,:,walkers.nup:]
    return walkers

@plum.dispatch
def tensor2walkers(walkers:GHFWalkers,phi):
    walkers.phi = xp.asarray(phi)
    return walkers 

def save_walkers(walkers,comm,dirname):
    RANK,SIZE = comm.rank,comm.size
    if RANK>0:
        obj = to_host(walkers2tensor(walkers)),to_host(walkers.weight),to_host(walkers.sgn_ovlp)
        comm.send(obj,0)
        return
    phi = [to_host(walkers2tensor(walkers))] + ([None] * (SIZE-1))
    weights = [to_host(walkers.weight)] + ([None] * (SIZE-1))
    sgn_ovlp = [to_host(walkers.sgn_ovlp)] + ([None] * (SIZE-1))
    for r in range(1,SIZE):
        phi[r],weights[r],sgn_ovlp[r] = comm.recv(source=r)
    with h5py.File(f'{dirname}/walkers.hdf5','w') as f:
        f.create_dataset('phi',data=np.concatenate(phi,axis=0))
        f.create_dataset('log_weights',data=np.concatenate(weights,axis=0))
        f.create_dataset('sgn_ovlp',data=np.concatenate(sgn_ovlp,axis=0))

def load_walkers(walkers,comm,dirname):
    with h5py.File(f'{dirname}/walkers.hdf5','r') as f:
        phi = f['phi'][:]
        log_weights = f['log_weights'][:]
        sgn_ovlp = f['sgn_ovlp'][:]
    print(phi.shape)
    print(log_weights.shape)
    print(sgn_ovlp.shape)
    exit()

    RANK,SIZE = comm.rank,comm.size

    nw = log_weights.size
    b,r = nw//SIZE,nw%SIZE
    counts = np.array([b]*SIZE)
    if r>0:
        counts[:r] += 1
    counts = np.cumsum(counts)
    start = 0 if RANK==0 else counts[RANK-1]
    stop = counts[RANK]
    print(f'RANK={RANK},start={start},stop={stop}')
    walkers = tensor2walkers(walkers,phi[start:stop])
    walkers.weight = xp.exp(xp.asarray(log_weights[start:stop]))
    walkers.sgn_ovlp = xp.asarray(sgn_ovlp[start:stop])
    return walkers

class Rotation:

    def __init__(self,chol_idx,pa=None,ga=None,pb=None,gb=None):
        self.chol_idx = chol_idx
        self.p = [None,None]
        self.g = [None,None]
        self.d = [None,None]
        self.r = [None,None]
        if pa is not None:
            self.p[0] = xp.asarray(pa,dtype=int)
            self.g[0] = xp.asarray(ga)
            self.d[0] = xp.exp(self.g[0])-1.
            self.r[0] = len(pa)
        if pb is not None:
            self.p[1] = xp.asarray(pb,dtype=int)
            self.g[1] = xp.asarray(gb)
            self.d[1] = xp.exp(self.g[1])-1.
            self.r[1] = len(pb)

    def get_MB_kappa(self,v,basis):
        kappa = [None] * 2
        for s in (0,1):
            p,g = self.p[s],self.g[s]
            if p is None:
                continue
            ks = np.einsum('xr,yr,r->xy',v[:,p],v[:,p],g)
            kappa[s] = quadratic2MB(ks,basis,s) 
        return kappa

    def get_rotation_matrix(self,v):
        U = [None] * 2
        for s in (0,1):
            p,d = self.p[s],self.d[s]
            if p is None:
                continue
            diag = xp.ones(v.shape[0])
            diag[p] += d
            U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        return U

def _update_walkers(C,d,u,w):
    if d is None:
        return C
    right = xp.einsum('wr,wxr,wxi->wri',d,u,C[w])
    C[w] += xp.einsum('wxr,wri->wxi',u,right)
    return C

def _ovlp(C,B=None):
    if C is None:
        return None
    if B is None:
        return xp.einsum('wxi,wxj->wij',C,C)
    return xp.einsum('wxi,xj->wij',C,B)

def _inv(ovlp):
    if ovlp is None:
        return None
    return xp.linalg.inv(ovlp)

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,inv=True):
    C = walkers.phia.real,walkers.phib.real
    CdC = [_ovlp(Ci) for Ci in C]
    if not inv:
        return CdC
    return [_inv(oi) for oi in CdC]

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True):
    C = walkers.phia.real,walkers.phib.real
    B = trial.psi0a.real,trial.psi0b.real
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,B)] 
    if not inv:
        return CdB
    return [_inv(oi) for oi in CdB]

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True):
    C = walkers.phia.real,walkers.phib.real
    B = trial.psi0a.real,trial.psi0b.real
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,B)] 
    CdB = xp.concatenate(CdB,axis=1)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

@plum.dispatch
def compute_ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True):
    CdB = _ovlp(walkers.phi.real,B=trial.psi0.real)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

def _rdm1(ovlp,C,B=None):
    if C is None:
        return None
    if B is None:
        D = xp.einsum('wxi,wij->wxj',C,ovlp) 
    else:
        D = xp.einsum('xi,wij->wxj',B,ovlp) 
    return xp.einsum('wxj,wyj->wxy',D,C) 

def _det(ovlp,inv=True):
    if ovlp is None:
        return None
    det = xp.linalg.det(ovlp)
    if inv:
        det = 1./det
    return det

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,ovlp=False):
    CdCinv = compute_ovlp(walkers)
    C = walkers.phia.real,walkers.phib.real
    D = [_rdm1(oi,Ci) for oi,Ci in zip(CdCinv,C)] 
    if not ovlp:
        return D
    ovlp = _det(CdCinv[0])
    if CdCinv[1] is not None:
        ovlp *= _det(CdCinv[1])
    return D,ovlp

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDet,ovlp=False):
    CdBinv = compute_ovlp(walkers,trial)
    C = walkers.phia.real,walkers.phib.real
    B = trial.psi0a.real,trial.psi0b.real
    D = [_rdm1(oi,Ci,B=Bi) for oi,Ci,Bi in zip(CdBinv,C,B)] 
    if not ovlp:
        return D
    ovlp = _det(CdBinv[0])
    if CdBinv[1] is not None:
        ovlp *= _det(CdBinv[1])
    return D,ovlp

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = compute_ovlp(walkers,trial)
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
def compute_rdm1(walkers:GHFWalkers,trial:SingleDetGHF,ovlp=False):
    CdBinv = compute_ovlp(walkers,trial)
    D = _rdm1(CdBinv,walkers.phi.real,B=trial.psi0.real)
    if not ovlp:
        return D
    return D,1./xp.linalg.det(CdBinv)

def _trace(D):
    if D is None:
        return 0
    return xp.einsum('wxy->w',D**2)

def compute_trace(D):
    if isinstance(D,list):
        return  _trace(D[0])+_trace(D[1]) 
    return _trace(D)

class SumOfRotationBase:

    def __init__(self,eps_sq=None):
        self.eps_sq = eps_sq
        self.chol_basis = []
        self.terms = [] 
        self.bare_gf = []
        self.Lambda = 0 

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        self.nbasis = self.h1e.shape[0]

        ek,vk = xp.linalg.eigh(self.h1e-self.v0) 
        self.chol_basis.append(vk)
        chol_idx = len(self.chol_basis)-1

        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        assert at>xp.amax(xp.fabs(ek))
        nonzero_bands = xp.nonzero(ek)[0]
        eta = xp.log(1.-ek/at)

        self.bare_gf += [at] * (2*nonzero_bands.size)
        self.Lambda += at*2*nonzero_bands.size
        for k in nonzero_bands:
            self.terms.append(Rotation(chol_idx,pa=[k],ga=[eta[k]]))
            self.terms.append(Rotation(chol_idx,pb=[k],gb=[eta[k]]))
        return ek

    def parse_decomposition(self,iprint=0):
        if iprint>0:
            print('Lambda=',self.Lambda)
        self.chol_basis = xp.asarray(self.chol_basis)
        self.bare_gf = xp.asarray(self.bare_gf)/self.Lambda
        assert self.bare_gf.size==len(self.terms)
        if iprint>0:
            print('normalization=',xp.fabs(self.bare_gf).sum())

    def compute_guiding_fxn(self,walkers,trial):
        if self.eps_sq is None:
            CdB = compute_ovlp(walkers,trial,inv=False)
            if isinstance(CdB,list):
                ovlp = _det(CdB[0],inv=False)
                if CdB[1] is not None:
                    ovlp *= _det(CdB[1],inv=False)
            else:
                ovlp = xp.linalg.det(CdB)
        else:
            D,ovlp = compute_rdm1(walkers,trial,ovlp=True)
            R = compute_trace(D)
            R = 1./xp.sqrt(1.+self.eps_sq*R)
            ovlp /= R
        walkers.ovlp = ovlp

    def _parse_sampled_rotations(self,kixs):
        dmap = {(s,r):[] for s in (0,1) for r in (1,2)}
        umap = {(s,r):[] for s in (0,1) for r in (1,2)}
        wmap = {(s,r):[] for s in (0,1) for r in (1,2)}
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            for s in (0,1):
                p = term.p[s]
                if p is None:
                    continue
                r = term.r[s]
                dmap[s,r].append(term.d[s])
                umap[s,r].append(self.chol_basis[term.chol_idx][:,p])
                wmap[s,r].append(w)
        for s,r in itertools.product((0,1),(1,2)):
            if len(dmap[s,r])==0:
                dmap[s,r] = None
                umap[s,r] = None
                wmap[s,r] = None
                continue
            dmap[s,r] = xp.asarray(dmap[s,r])
            umap[s,r] = xp.asarray(umap[s,r])
            wmap[s,r] = xp.asarray(wmap[s,r])
        return dmap,umap,wmap

    @plum.dispatch
    def update_walkers(self,kixs,walkers:UHFWalkers):
        dmap,umap,wmap = self._parse_sampled_rotations(kixs)
        for r in (1,2):
            walkers.phia = _update_walkers(walkers.phia.real,dmap[0,r],umap[0,r],wmap[0,r])
            if walkers.phib is not None:
                walkers.phib = _update_walkers(walkers.phib.real,dmap[1,r],umap[1,r],wmap[1,r])

    @plum.dispatch
    def update_walkers(self,kixs,walkers:GHFWalkers):
        dmap,umap,wmap = self._parse_sampled_rotations(kixs)
        nb = walkers.nbasis
        for r in (1,2):
            walkers.phi[:,:nb] = _update_walkers(walkers.phi[:,:nb].real,dmap[0,r],umap[0,r],wmap[0,r])
            walkers.phi[:,nb:] = _update_walkers(walkers.phi[:,nb:].real,dmap[1,r],umap[1,r],wmap[1,r])

    @plum.dispatch
    def _update_walkers_slow(self,kixs,walkers:UHFWalkers):
        phia = walkers.phia.real.copy()
        phib = None
        if walkers.phib is not None:
            phib = walkers.phib.real.copy()
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            U = term.get_rotation_matrix(self.chol_basis[term.chol_idx])
            if U[0] is not None:
                phia[w] = xp.dot(U[0],phia[w])
            if phib is not None:
                if U[1] is not None:
                    phib[w] = xp.dot(U[1],phib[w])
        return phia,phib

    def local_energy(self,walkers,trial):
        D = compute_rdm1(walkers,trial)
        R = None 
        if self.eps_sq is not None:
            R = compute_trace(D)
            R = 1./xp.sqrt(1.+self.eps_sq*R)

        if isinstance(D,list):
            Daa,Dbb = D
            Dab = Dba = None
        else:
            nb = self.nbasis
            Daa,Dab,Dba,Dbb = D[:,:nb,:nb],D[:,:nb,nb:],D[:,nb:,:nb],D[:,nb:,nb:]

        E1 = xp.einsum('ij,wij->w',self.h1e,Daa+Dbb)
        E2 = self.compute_E2(Daa,Dbb,Dab=Dab,Dba=Dba)
        return E1+E2,E1,E2,R

    def _get_MB_gf(self,basis):
        H = 0
        print('called')
        for ai,term in zip(self.bare_gf,self.terms): 
            kappa = term.get_MB_kappa(self.chol_basis[term.chol_idx],basis)
            U = None
            for spin,k in enumerate(kappa):
                if k is None:
                    continue
                Us = scipy.linalg.expm(k)
                if U is None:
                    U = Us
                else:
                    U = np.dot(U,Us)
            H += ai*U
        return H

class HubbardSOR(SumOfRotationBase):

    def __init__(self,h1e,U,eps_sq=None):
        super().__init__(eps_sq=eps_sq)
        self.h1e = xp.asarray(h1e)
        self.U = U
        self.v0 = 0.

    def decompose_h2(self,gu,iprint=0,nelec=None):
        self.chol_basis.append(xp.eye(self.nbasis))
        chol_idx = len(self.chol_basis)-1

        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print('ai=',ai)
        self.bare_gf += [ai] * (2*self.nbasis)
        self.Lambda += 2*ai*self.nbasis
        if nelec is not None:
            self.Lambda += self.U*sum(nelec)/2 
        for i in range(self.nbasis): 
            self.terms.append(Rotation(chol_idx,pa=[i],ga=[gu],pb=[i],gb=[-gu]))
            self.terms.append(Rotation(chol_idx,pa=[i],ga=[-gu],pb=[i],gb=[gu]))

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = xp.einsum('wii,wii->w',Daa,Dbb)
        if Dab is None:
            return E2*self.U
        E2 -= xp.einsum('wii,wii->w',Dab,Dba)
        return E2*self.U

class QCSOR(SumOfRotationBase):

    def __init__(self,h1e,chol,eps_sq=None):
        super().__init__(eps_sq=eps_sq)
        self.h1e = xp.asarray(h1e)
        self.chol = xp.asarray(chol)
        self.v0 = .5*xp.einsum('npr,nrs->ps',self.chol,self.chol) 

    def decompose_h2(self,ai,thresh=1e-6,iprint=0):
        aisq = ai**2/2
        if iprint>0:
            print('ai=',ai)
        for i,L in enumerate(self.chol):
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_idx = len(self.chol_basis)-1

            if iprint>0:
                print(f'chol={i},bands={ek}')
            assert ai>xp.amax(xp.fabs(ek))
            nonzero_bands = xp.nonzero(ek)[0]
            self.Lambda += (ai*2*nonzero_bands.size)**2/2.
            eta_plus = xp.log(1.+ek/ai) 
            eta_minus = xp.log(1.-ek/ai) 

            for p,q in itertools.product(nonzero_bands,repeat=2):
                self.bare_gf += [aisq] * 4
                eta_p = eta_minus[p]
                eta_q = eta_plus[q]
                if p==q:
                    self.terms.append(Rotation(chol_idx,pa=[p],ga=[eta_p+eta_q]))
                    self.terms.append(Rotation(chol_idx,pb=[p],gb=[eta_p+eta_q]))
                else:
                    self.terms.append(Rotation(chol_idx,pa=[p,q],ga=[eta_p,eta_q]))
                    self.terms.append(Rotation(chol_idx,pb=[p,q],gb=[eta_p,eta_q]))
                self.terms.append(Rotation(chol_idx,pa=[p],ga=[eta_p],pb=[q],gb=[eta_q]))
                self.terms.append(Rotation(chol_idx,pa=[q],ga=[eta_q],pb=[p],gb=[eta_p]))

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = 0
        for i,L in enumerate(self.chol):
            DaaL = xp.einsum('wpq,qr->wpr',Daa,L)
            DbbL = xp.einsum('wpq,qr->wpr',Dbb,L)
            E2 += xp.einsum('wpp->w',DaaL+DbbL)**2
            E2 -= xp.einsum('wpq,wqp->w',DaaL,DaaL)
            E2 -= xp.einsum('wpq,wqp->w',DbbL,DbbL)
            if Dab is not None:
                DabL = xp.einsum('wpq,qr->wpr',Dab,L)
                DbaL = xp.einsum('wpq,qr->wpr',Dba,L)
                E2 -= 2.*xp.einsum('wpq,wqp->w',DabL,DbaL)
        return 0.5*E2

##### MB helper fxns #####
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

def hubbard2MB(h1e,U,nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,1,thresh=thresh)
    for ix,cf in enumerate(basis):
        H[ix,ix] += U*count_double_occupancy(cf,nsite)
    return H,basis,basis_map

def chol2MB(h1e,chol,nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,1,thresh=thresh)
    for i,L in enumerate(chol):
        L_ = quadratic2MB(L,basis,0,thresh=thresh)
        L_ += quadratic2MB(L,basis,1,thresh=thresh)
        H += np.dot(L_,L_)/2.
    return H,basis,basis_map
