import numpy as np
import scipy,itertools
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        string_act,
        count_double_occupancy, 
)
from ipie.utils.backend import arraylib as xp

class Rotation:

    def __init__(self,chol_idx,typ,p,g,d,d2):
        self.chol_idx = chol_idx
        assert typ in ['h1a','h1b','h2a','h2b','h2ab']
        self.typ = typ

        self.p = p 
        self.g = g
        self.d = d
        self.d2 = d2
        
    def get_MB_kappa(self,v,basis):
        kappa = [None] * 2
        if self.typ=='h2ab':
            for s,p in enumerate(self.p):
                ks = np.outer(v[:,p],v[:,p]*self.g[s])
                kappa[s] = quadratic2MB(ks,basis,s) 
        else:
            s = 0 if self.typ[-1]=='a' else 1
            ks = np.einsum('xr,yr,r->xy',v[:,self.p],v[:,self.p],self.g)
            kappa[s] = quadratic2MB(ks,basis,s) 
        return kappa

    def get_rotation_matrix(self,v):
        U = [None] * 2
        if self.typ=='h2ab':
            for s,p in enumerate(self.p):
                diag = xp.ones(v.shape[0])
                diag[p] += self.d[s]
                U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        else:
            diag = xp.ones(v.shape[0])
            diag[self.p] += self.d
            s = 0 if self.typ[-1]=='a' else 1
            U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        return U

class SumOfRotationBase:

    def __init__(self,sample_method=0):
        self.chol_basis = []
        self.terms = [] 
        self.coeffs = []
        self.sample_method = sample_method

        # additional saved quantites 
        if self.sample_method==0:
            return
        elif self.sample_method==1:
            # sample with a_i*det(I_r+d_i)
            self.importance_factor = []
        else:
            raise NotImplementedError
        #if self.sample_method==2:
        #    # sample with a_i*det(I_r+d_i*D0[i])
        #    self.kix_map = {1:[],2:[]}
        #    self.dmap = {1:[],2:[]}
        #    self.pmap = {1:[],2:[]}

    def add_term(self,ai,chol_idx,typ,p,g,thresh=1e-6):
        p = xp.asarray(p,dtype=int)
        g = xp.asarray(g)
        d = xp.exp(g)-1.
        zeros = d[xp.fabs(d)<thresh]
        if zeros.size>0:
            return

        d2 = 2*d+d**2
        neg = d2+1.
        neg = neg[neg<thresh]
        if neg.size>0:
            print(f'incompatible d={d} for low rank lowdn.')
            raise ValueError

        self.terms.append(Rotation(chol_idx,typ,p,g,d,d2))
        self.coeffs.append(ai)

        if self.sample_method==1:
            self.importance_factor.append((d+1.).prod())
            #print(g,d,self.importance_factor[-1])
        #if self.sample_method==2:
        #    r = p.size
        #    self.kix_map

    def decompose_h1(self,at,thresh=1e-6,iprint=0):
        self.nbasis = self.h1e.shape[0]

        ek,vk = xp.linalg.eigh(self.h1e-self.v0) 
        self.chol_basis.append(vk)
        chol_idx = len(self.chol_basis)-1

        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        assert at>xp.amax(xp.fabs(ek))
        eta = xp.log(1.-ek/at)

        for k,eta_k in enumerate(eta):
            self.add_term(at,chol_idx,'h1a',[k],[eta_k])
            self.add_term(at,chol_idx,'h1b',[k],[eta_k])
        return ek

    def parse_decomposition(self,iprint=0):
        self.chol_basis = xp.asarray(self.chol_basis)
        self.coeffs = xp.asarray(self.coeffs)
        self.Lambda = self.coeffs.sum()
        self.coeffs /= self.Lambda
        self.nterms = len(self.terms) 
        assert self.coeffs.size==self.nterms
        if iprint>0:
            print('Lambda=',self.Lambda)
            print('normalization=',xp.fabs(self.coeffs).sum())
            print('number of terms=',self.nterms)

        if self.sample_method==1:
            self.importance_factor = xp.asarray(self.importance_factor)
            assert self.importance_factor.size==self.nterms

    def compute_prob(self,D=None):
        if self.sample_method==0:
            p = self.coeffs
        elif self.sample_method==1:
            p = self.coeffs * self.importance_factor 
        else:
            raise NotImplementedError

        p = xp.fabs(p)
        p /= p.sum()
        sign = self.coeffs / p
        return p,sign

    #def parse_sampled_rotations(self,kixs):
    #    rotations = dict()
    #    for w,kix in enumerate(kixs):
    #        term = self.terms[kix]
    #        typ = term.typ
    #        if typ not in rotations:
    #            rotations[typ] = {'d':[],'d2':[],'u':[],'w':[]}
    #        rotations[typ]['w'].append(w)
    #        rotations[typ]['d'].append(term.d)
    #        rotations[typ]['d2'].append(term.d2)
    #        rotations[typ]['u'].append(self.chol_basis[term.chol_idx][:,term.p])
    #    for typ in rotations:
    #        for key in ['w','d','d2','u']:
    #            rotations[typ][key] = xp.asarray(rotations[typ][key]) 
    #    return rotations

    def parse_sampled_rotations_slow(self,kixs):
        Us = [None] * len(kixs)
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            Us[w] = term.get_rotation_matrix(self.chol_basis[term.chol_idx])
        return Us

    def local_energy(self,walkers):
        D = walkers.D 
        if isinstance(D,list):
            Daa,Dbb = D
            Dab = Dba = None
        else:
            nb = self.nbasis
            Daa,Dab,Dba,Dbb = D[:,:nb,:nb],D[:,:nb,nb:],D[:,nb:,:nb],D[:,nb:,nb:]

        E1 = xp.einsum('ij,wij->w',self.h1e,Daa)
        if Dbb is not None:
            E1 += xp.einsum('ij,wij->w',self.h1e,Dbb)
        E2 = self.compute_E2(Daa,Dbb,Dab=Dab,Dba=Dba)
        return E1+E2,E1,E2

    def _get_MB_gf(self,basis):
        H = 0
        print('called')
        for ai,term in zip(self.coeffs,self.terms): 
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

    def __init__(self,h1e,U):
        super().__init__()
        self.h1e = xp.asarray(h1e)
        self.U = U
        self.v0 = -0.5*U*xp.eye(self.h1e.shape[0]) 

    def decompose_h2(self,gu,iprint=0,nelec=None):
        self.chol_basis.append(xp.eye(self.nbasis))
        chol_idx = len(self.chol_basis)-1

        ai = self.U/(np.cosh(gu)-1)/4
        if iprint>0:
            print(f'eta={gu},ai={ai}')
        for i in range(self.nbasis): 
            self.add_term(ai,chol_idx,'h2ab',[i,i],[gu,-gu])
            self.add_term(ai,chol_idx,'h2ab',[i,i],[-gu,gu])

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = xp.einsum('wii,wii->w',Daa,Dbb)
        if Dab is None:
            return E2*self.U
        E2 -= xp.einsum('wii,wii->w',Dab,Dba)
        return E2*self.U

class QCSOR(SumOfRotationBase):

    def __init__(self,h1e,chol):
        super().__init__()
        self.h1e = xp.asarray(h1e)
        self.chol = xp.asarray(chol)
        self.v0 = .5*xp.einsum('npr,nrs->ps',self.chol,self.chol) 

    def decompose_h2(self,ai,thresh=1e-6,iprint=0):
        aisq = ai**2/2
        if iprint>0:
            print('ai=',ai)
        for L in self.chol:
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_idx = len(self.chol_basis)-1

            if iprint>0:
                print(f'chol_idx={chol_idx},bands={ek}')
            assert ai>xp.amax(xp.fabs(ek))
            eta_plus = xp.log(1.+ek/ai) 
            eta_minus = xp.log(1.-ek/ai) 

            for p,q in itertools.product(np.arange(self.nbasis),repeat=2):
                eta_p = eta_minus[p]
                eta_q = eta_plus[q]
                if p==q:
                    self.add_term(aisq,chol_idx,'h1a',[p],[eta_p+eta_q])
                    self.add_term(aisq,chol_idx,'h1b',[p],[eta_p+eta_q])
                else:
                    self.add_term(aisq,chol_idx,'h2a',[p,q],[eta_p,eta_q])
                    self.add_term(aisq,chol_idx,'h2b',[p,q],[eta_p,eta_q])
                self.add_term(aisq,chol_idx,'h2ab',[p,q],[eta_p,eta_q])
                self.add_term(aisq,chol_idx,'h2ab',[q,p],[eta_q,eta_p])

    def compute_E2(self,Daa,Dbb,Dab=None,Dba=None):
        E2 = 0
        for i,L in enumerate(self.chol):
            DaaL = xp.einsum('wpq,qr->wpr',Daa,L)
            E2_1 = xp.einsum('wpp->w',DaaL)
            if Dbb is not None:
                DbbL = xp.einsum('wpq,qr->wpr',Dbb,L)
                E2_1 += xp.einsum('wpp->w',DbbL)
            E2 += E2_1**2

            E2 -= xp.einsum('wpq,wqp->w',DaaL,DaaL)
            if Dbb is not None:
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
