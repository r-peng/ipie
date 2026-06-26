import numpy as np
import scipy,itertools,plum
from ipie.hamiltonians.bitstring_utils import (
        get_all_configs_u11,
        get_all_configs_u1,
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

    def get_trial_expectation_1(self,UB):
        uB = UB[self.chol_idx][self.p[0]]
        n = xp.dot(uB,uB) 
        return 1.+self.d[0]*n

    def get_trial_expectation_2(self,UB1,UB2,cross=True):
        uB1 = UB1[self.chol_idx][self.p[0]]
        uB2 = UB2[self.chol_idx][self.p[1]]
        n1 = xp.dot(uB1,uB1) 
        n2 = xp.dot(uB2,uB2) 
        cross = xp.dot(uB1,uB2) if cross else 0.
        d1,d2 = self.d
        return 1.+d1*n1+d2*n2+d1*d2*(n1*n2-cross**2)

    def get_trial_expectation(self,UBa,UBb,cross):
        if self.typ=='h1a':
            return self.get_trial_expectation_1(UBa)
        if self.typ=='h1b':
            if UBb is None:
                return 1.
            else:
                return self.get_trial_expectation_1(UBb)
        if self.typ=='h2a':
            return self.get_trial_expectation_2(UBa,UBa)
        if self.typ=='h2b':
            if UBb is None:
                return 1.
            else:
                return self.get_trial_expectation_2(UBb,UBb)
        if self.typ=='h2ab':
            if UBb is None:
                return self.get_trial_expectation_1(UBa)
            else:
                return self.get_trial_expectation_2(UBa,UBb,cross=cross)

class Rotations:
    def __init__(self):
        self.data = dict()
        self.typs = 'h1a','h1b','h2a','h2b','h2ab'
        self.keys = 'w','d','d2','u','uBa','uBb'
        for typ in self.typs:
            self.data[typ] = {key:[] for key in self.keys}

    def add_term(self,term,w,U,UBa,UBb):
        typ = term.typ
        p = term.p
        self.data[typ]['w'].append(w)
        self.data[typ]['d'].append(term.d)
        self.data[typ]['d2'].append(term.d2)
        self.data[typ]['u'].append(U[:,p])
        if typ=='h2ab':
            self.data[typ]['uBa'].append(UBa[p[:1]])
            if UBb is None: 
                return
            self.data[typ]['uBb'].append(UBb[p[1:]])
            return
        if typ[-1]=='a':
            self.data[typ]['uBa'].append(UBa[p])
            return
        if UBb is None:
            return
        self.data[typ]['uBb'].append(UBb[p])

    def parse_terms(self):
        for typ in self.typs:
            for key in self.keys:
                dat = self.data[typ][key]
                if len(dat)==0:
                    dat = None
                else:
                    dat = xp.asarray(dat) 
                self.data[typ][key] = dat 

    def get_data(self,typ):
        w = self.data[typ]['w']
        d = self.data[typ]['d']
        d2 = self.data[typ]['d2']
        u = self.data[typ]['u']
        uB = [self.data[typ]['uBa'],self.data[typ]['uBb']]
        return w,d,d2,u,uB

def _UB(B,chol_basis):
    if B is None:
        return None
    return xp.einsum('dxp,xi->dpi',chol_basis,B)

def _h1B(B,h1e):
    if B is None:
        return None
    return xp.dot(h1e,B)

class SumOfRotationBase:

    def __init__(self,apply_spin_down=True,importance_sample=True):
        self.chol_basis = []
        self.terms = [] 
        self.coeffs = []
        self.apply_spin_down = apply_spin_down
        self.importance_sample = importance_sample

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
            if self.apply_spin_down:
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

    def _conjugate(self,psi):
        self.UB = [_UB(Bi,self.chol_basis) for Bi in psi]
        self.h1B = [_h1B(Bi,self.h1e) for Bi in psi]

    def compute_importance_factor(self,cross):
        if not self.importance_sample:
            return None
        UBa,UBb = self.UB 
        fac = xp.asarray([term.get_trial_expectation(UBa,UBb,cross) for term in self.terms])
        #print(fac)
        #exit()
        return fac

    def compute_probability(self,fac):
        self.prob = self.coeffs
        if fac is not None:
            self.prob *= fac

        self.prob = xp.fabs(self.prob)
        self.prob /= self.prob.sum()
        self.a_over_q = self.coeffs / self.prob

    def parse_sampled_rotations(self,kixs):
        rotations = Rotations() 
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            chol_idx = term.chol_idx
            U = self.chol_basis[chol_idx]
            UBa = self.UB[0][chol_idx]
            UBb = None if self.UB[1] is None else self.UB[1][chol_idx]
            rotations.add_term(self.terms[kix],w,U,UBa,UBb)
        rotations.parse_terms()
        return rotations

    def parse_sampled_rotations_slow(self,kixs):
        Us = [None] * len(kixs)
        for w,kix in enumerate(kixs):
            term = self.terms[kix]
            Us[w] = term.get_rotation_matrix(self.chol_basis[term.chol_idx])
        return Us

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

    def __init__(self,h1e,U,**kwargs):
        super().__init__(**kwargs)
        self.h1e = xp.asarray(h1e)
        self.hubbard_U = U
        self.v0 = -0.5*U*xp.eye(self.h1e.shape[0]) 

    def decompose_h2(self,gu,iprint=0,nelec=None):
        self.chol_basis.append(xp.eye(self.nbasis))
        chol_idx = len(self.chol_basis)-1

        ai = self.hubbard_U/(np.cosh(gu)-1)/4
        if iprint>0:
            print(f'eta={gu},ai={ai}')
        assert self.apply_spin_down
        for i in range(self.nbasis): 
            self.add_term(ai,chol_idx,'h2ab',[i,i],[gu,-gu])
            self.add_term(ai,chol_idx,'h2ab',[i,i],[-gu,gu])

class QCSOR(SumOfRotationBase):

    def __init__(self,h1e,chol,**kwargs):
        super().__init__(**kwargs)
        self.h1e = xp.asarray(h1e)
        self.chol = xp.asarray(chol)
        self.v0 = .5*xp.einsum('npr,nrs->ps',self.chol,self.chol) 

    def _conjugate(self,psi):
        super()._conjugate(psi)
        self.LB = [_UB(Bi,self.chol) for Bi in psi]

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
                    if self.apply_spin_down: 
                        self.add_term(aisq,chol_idx,'h1b',[p],[eta_p+eta_q])
                else:
                    self.add_term(aisq,chol_idx,'h2a',[p,q],[eta_p,eta_q])
                    if self.apply_spin_down: 
                        self.add_term(aisq,chol_idx,'h2b',[p,q],[eta_p,eta_q])
                if self.apply_spin_down:
                    self.add_term(aisq,chol_idx,'h2ab',[p,q],[eta_p,eta_q])
                    self.add_term(aisq,chol_idx,'h2ab',[q,p],[eta_q,eta_p])
                else:
                    self.add_term(aisq,chol_idx,'h1a',[p],[eta_p])
                    self.add_term(aisq,chol_idx,'h1a',[q],[eta_q])

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

def eri2MB(eri,basis,spin1,spin2,thresh=1e-6):
    basis_map = {cf:i for i,cf in enumerate(basis)}
    H = np.zeros((len(basis),)*2)
    for (p,r,q,s) in itertools.product(range(eri.shape[0]),repeat=4):
        if np.absolute(eri[p,r,q,s])<thresh:
            continue
        for ix1,cf1 in enumerate(basis):
            ops = (2*p+spin1,'cre'),(2*q+spin2,'cre'),(2*s+spin2,'des'),(2*r+spin1,'des')
            cf2,sign = string_act(cf1,ops)
            if cf2 is None:
                continue
            ix2 = basis_map[cf2]
            H[ix2,ix1] += eri[p,r,q,s]*sign
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

def hubbard2MB(h1e,U,symmetry='u11',nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        if symmetry=='u11':
            basis = get_all_configs_u11((nsite,nsite),nelecs)
        elif symmetry=='u1':
            basis = get_all_configs_u1(2*nsite,sum(nelecs))
        else:
            raise NotImplementedError
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    H = quadratic2MB(h1e,basis,0,thresh=thresh)
    H += quadratic2MB(h1e,basis,1,thresh=thresh)
    for ix,cf in enumerate(basis):
        H[ix,ix] += U*count_double_occupancy(cf,nsite)
    return H,basis,basis_map

def chol2MB(h1e,chol=None,eri=None,nelecs=None,basis=None,basis_map=None,thresh=1e-6):
    nsite = h1e.shape[0]
    if basis is None:
        basis = get_all_configs_u11((nsite,nsite),nelecs)
    if basis_map is None:
        basis_map = {cf:i for i,cf in enumerate(basis)}

    v0 = 0.
    if chol is not None:
        v0 = .5*xp.einsum('npr,nrs->ps',chol,chol) 

    H = quadratic2MB(h1e-v0,basis,0,thresh=thresh)
    H += quadratic2MB(h1e-v0,basis,1,thresh=thresh)
    if chol is None:
        for s1,s2 in itertools.product((0,1),repeat=2):
            H += .5 * eri2MB(eri,basis,s1,s2,thresh=thresh)
        return H,basis,basis_map

    if eri is None:
        for i,L in enumerate(chol):
            L_ = quadratic2MB(L,basis,0,thresh=thresh)
            L_ += quadratic2MB(L,basis,1,thresh=thresh)
            H += .5 * np.dot(L_,L_)
        return H,basis,basis_map
    
