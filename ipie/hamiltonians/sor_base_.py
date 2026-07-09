import numpy as np
import plum
from ipie.utils.backend import arraylib as xp
from ipie.walkers.lafqmc_uhf_walkers import UHFWalkers
from ipie.walkers.lafqmc_ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet 
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF

class Rotation:

    def __init__(self,a,chol_idx,p,d,typ):
        self.a = a 
        self.chol_idx = chol_idx
        self.p = xp.asarray(p) 
        self.d = xp.asarray(d)
        assert typ in ['h1a','h1b','h2a','h2b','h2ab']
        self.typ = typ
        self.d2 = self.d*2.+self.d**2

    def add_term(self,a,d):
        if self.d.size!=1:
            raise ValueError
        self.d = self.a * self.d + a * d[0]
        self.a += a
        self.d /= self.a
        self.d2 = self.d*2.+self.d**2
        
    def get_rotation_matrix(self,chol_basis):
        v = chol_basis[self.chol_idx]
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

#    def get_trial_expectation_1(self,UB):
#        uB = UB[self.chol_idx][self.p[0]]
#        n = xp.dot(uB,uB) 
#        return 1.+self.d[0]*n
#
#    def get_trial_expectation_2(self,UB1,UB2,cross=True):
#        uB1 = UB1[self.chol_idx][self.p[0]]
#        uB2 = UB2[self.chol_idx][self.p[1]]
#        n1 = xp.dot(uB1,uB1) 
#        n2 = xp.dot(uB2,uB2) 
#        cross = xp.dot(uB1,uB2) if cross else 0.
#        d1,d2 = self.d
#        return 1.+d1*n1+d2*n2+d1*d2*(n1*n2-cross**2)
#
#    def get_trial_expectation(self,UBa,UBb,cross):
#        if self.typ=='h1a':
#            return self.get_trial_expectation_1(UBa)
#        if self.typ=='h1b':
#            if UBb is None:
#                return 1.
#            else:
#                return self.get_trial_expectation_1(UBb)
#        if self.typ=='h2a':
#            return self.get_trial_expectation_2(UBa,UBa)
#        if self.typ=='h2b':
#            if UBb is None:
#                return 1.
#            else:
#                return self.get_trial_expectation_2(UBb,UBb)
#        if self.typ=='h2ab':
#            if UBb is None:
#                return self.get_trial_expectation_1(UBa)
#            else:
#                return self.get_trial_expectation_2(UBa,UBb,cross=cross)

#class Rotations:
#    def __init__(self):
#        self.data = dict()
#        self.typs = 'h1a','h1b','h2a','h2b','h2ab'
#        self.keys = 'w','d','d2','u','uBa','uBb'
#        for typ in self.typs:
#            self.data[typ] = {key:[] for key in self.keys}
#
#    def add_hamiltonian_term(self,term,w,U,UBa,UBb):
#        typ = term.typ
#        p = term.p
#        self.data[typ]['w'].append(w)
#        self.data[typ]['d'].append(term.d)
#        self.data[typ]['d2'].append(term.d2)
#        self.data[typ]['u'].append(U[:,p])
#        if typ=='h2ab':
#            self.data[typ]['uBa'].append(UBa[p[:1]])
#            if UBb is None: 
#                return
#            self.data[typ]['uBb'].append(UBb[p[1:]])
#            return
#        if typ[-1]=='a':
#            self.data[typ]['uBa'].append(UBa[p])
#            return
#        if UBb is None:
#            return
#        self.data[typ]['uBb'].append(UBb[p])
#
#    def parse_terms(self):
#        for typ in self.typs:
#            for key in self.keys:
#                dat = self.data[typ][key]
#                if len(dat)==0:
#                    dat = None
#                else:
#                    dat = xp.asarray(dat) 
#                self.data[typ][key] = dat 
#
#    def get_data(self,typ):
#        w = self.data[typ]['w']
#        d = self.data[typ]['d']
#        return w,d,d2,u,uB
#
#    def add_itm(self,typ,key,itm):
#        self.data[typ][key] = itm 
#
#    def get_itm(self,typ,key):
#        if key=='uB':
#            return [self.data[typ]['uBa'],self.data[typ]['uBb']]
#        itm = self.data[typ][key]
#        if typ=='h2ab' and key=='u':
#            return [itm[:,:,:1],itm[:,:,1:]]
#        return itm

class SumOfRotationBase:

    def __init__(self,apply_spin_down=True,importance_sample=False,thresh=1e-6):
        self.chol = None
        self.chol_basis = []
        self.term_dict = dict() 
        self.apply_spin_down = apply_spin_down
        self.importance_sample = importance_sample
        self.thresh = thresh

    def add_term(self,ai,chol_idx,p,d,s):
        typ = None
        r = len(p)
        if r==1:
            if xp.fabs(d[0])<self.thresh:
                return
        elif r==2:
            if xp.fabs(d[0])<self.thresh and xp.fabs(d[1])<self.thresh:
                return
            elif xp.fabs(d[0])<self.thresh:
                p = [p[1]]
                d = [d[1]]
                s = [s[1]]
            elif xp.fabs(d[1])<self.thresh:
                p = [p[0]]
                d = [d[0]]
                s = [s[0]]
            else:
                typ = {(0,0):'h2a',(1,1):'h2b',(0,1):'h2ab'}[s[0],s[1]] 
                #if p[0]>p[1]:
                #    p = [p[1],p[0]]
                #    d = [d[1],d[0]]
                #    s = [s[1],s[0]]
        else:
            raise ValueError

        r = len(p)
        if r==1: 
            typ = ['h1a','h1b'][s[0]]

        key = chol_idx,tuple(p),tuple(s)
        if r==1:
            if key in self.term_dict:
                self.term_dict[key].add_term(ai,d)
            else:
                self.term_dict[key] = Rotation(ai,chol_idx,p,d,typ)
        else:
            if key not in self.term_dict:
                self.term_dict[key] = []
            self.term_dict[key].append(Rotation(ai,chol_idx,p,d,typ))

    def decompose_h1(self,at,iprint=0):
        self.nbasis = self.h1e.shape[0]

        ek,vk = xp.linalg.eigh(self.h1e-self.v0) 
        self.chol_basis.append(vk)
        chol_idx = len(self.chol_basis)-1

        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        assert at>xp.amax(xp.fabs(ek))

        for k,e_k in enumerate(ek):
            self.add_term(at,chol_idx,[k],[-e_k/at],[0])
            if self.apply_spin_down:
                self.add_term(at,chol_idx,[k],[-e_k/at],[1])
        return ek

    def parse_decomposition(self,iprint=0):
        self.chol_basis = xp.asarray(self.chol_basis)
        self.nchol = self.chol_basis.shape[0]
        self.chol_basis2 = dict()
        for i,Ui in enumerate(self.chol_basis):
            for j in range(i+1,self.nchol):
                Uj = self.chol_basis[j]
                self.chol_basis2[i,j] = xp.dot(Ui.T,Uj)

        self.terms = []
        self.coeffs = []
        self.kix2key = []
        self.p_dict = dict() 
        self.d_dict = dict() 
        self.kix_dict = dict()
        for term_key in self.term_dict:
            chol_idx,p,_ = term_key
            if len(p)==1:
                rot = self.term_dict[term_key] 
                if xp.fabs(rot.d[0])<self.thresh:
                    #print('not included',key,rot.d,rot.d2,rot.a)
                    continue
                terms = [rot]
            else:
                terms = self.term_dict[term_key] 
            for i,rot in enumerate(terms):
                self.terms.append(rot)
                self.coeffs.append(rot.a)
                kix = len(self.terms)-1

                key = chol_idx,rot.typ 
                print(kix,key,rot.d,rot.d2,rot.a)
                self.kix2key.append((key,i))
                if key not in self.p_dict:
                    self.p_dict[key] = []
                if key not in self.d_dict:
                    self.d_dict[key] = []
                if key not in self.kix_dict:
                    self.kix_dict[key] = []
                self.p_dict[key].append(rot.p)
                self.d_dict[key].append(rot.d)
                self.kix_dict[key].append(kix)

        self.term_dict = None
        for key in self.p_dict:
            self.p_dict[key] = xp.asarray(self.p_dict[key])
        for key in self.d_dict:
            self.d_dict[key] = xp.asarray(self.d_dict[key])
        for key in self.kix_dict:
            self.kix_dict[key] = xp.asarray(self.kix_dict[key])

        self.coeffs = xp.asarray(self.coeffs)
        self.Lambda = self.coeffs.sum()
        self.coeffs /= self.Lambda
        self.nterms = len(self.terms) 
        assert self.coeffs.size==self.nterms
        if iprint>0:
            print('Lambda=',self.Lambda)
            print('normalization=',xp.fabs(self.coeffs).sum())
            print('number of terms=',self.nterms)

    def get_ud(self,key,i):
        chol_idx,typ = key
        p = self.p_dict[key][i]
        d = self.d_dict[key][i]
        u = self.chol_basis[chol_idx]
        u = xp.asarray([u[:,pi] for pi in p])

        nw,r = p.shape
        u2 = xp.zeros((nw,self.nchol,self.nbasis,r))
        for i in range(self.nchol):
            if i<chol_idx:
                U2 = self.chol_basis2[i,chol_idx]
            elif i>chol_idx:
                U2 = self.chol_basis2[i,chol_idx].T
            else:
                U2 = xp.eye(self.nbasis)
            u2[i] = [U2[:,pi] for pi in p]
        u2 = xp.asarray(u2)
        return u,d,u2

    #def parse_sampled_rotations(self,kixs):
    #    rotations = Rotations() 
    #    for w,kix in enumerate(kixs):
    #        term = self.terms[kix]
    #        chol_idx = term.chol_idx
    #        U = self.chol_basis[chol_idx]
    #        UBa = self.UB[0][chol_idx]
    #        UBb = None if self.UB[1] is None else self.UB[1][chol_idx]
    #        rotations.add_hamiltonian_term(term,w,U,UBa,UBb)
    #    rotations.parse_terms()
    #    return rotations

    #def parse_sampled_rotations_slow(self,kixs):
    #    Us = [None] * len(kixs)
    #    for w,kix in enumerate(kixs):
    #        term = self.terms[kix]
    #        Us[w] = term.get_rotation_matrix(self.chol_basis[term.chol_idx])
    #    return Us

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
        dp = xp.exp(gu)-1.
        dm = xp.exp(-gu)-1.
        if iprint>0:
            print(f'eta={gu},ai={ai}')
        assert self.apply_spin_down
        for i in range(self.nbasis): 
            self.add_term(ai,chol_idx,[i,i],[dp,dm],[0,1])
            self.add_term(ai,chol_idx,[i,i],[dm,dp],[0,1])

    def local_energy(self,walkers,trial):
        E1,SC = trial.compute_E1(walkers)
    
        Daa,Dbb,Dab,Dba = trial.compute_1rdm_diag(SC)
        if Dbb is None:
            E2 = xp.zeros(Daa.shape[0])
            return E1+E2,E1,E2 
        E2 = (Daa*Dbb).sum(axis=1)
        if Dab is None:
            E2 *= self.hubbard_U
            return E1+E2,E1,E2
        E2 -= (Dab*Dba).sum(axis=1)
        E2 *= self.hubbard_U
        return E1+E2,E1,E2

class QCSOR(SumOfRotationBase):

    def __init__(self,h1e,chol,**kwargs):
        super().__init__(**kwargs)
        self.h1e = xp.asarray(h1e)
        self.chol = xp.asarray(chol)
        self.v0 = 0.

    def _conjugate(self,psi):
        super()._conjugate(psi)
        self.LB = [_UB(Bi,self.chol) for Bi in psi]

    def decompose_h2(self,ai,iprint=0):
        aisq = ai**2/2
        if iprint>0:
            print('ai,aisq=',ai,aisq)
        for L in self.chol:
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_idx = len(self.chol_basis)-1

            if iprint>0:
                print('nchol idx=',chol_idx)
                print('bands=',ek)
            assert ai>xp.amax(xp.fabs(ek))

            #for p,q in itertools.product(np.arange(self.nbasis),repeat=2):
            #    dpm = -ek[p]/ai
            #    dqp = ek[q]/ai
            #    if p!=q:
            #        self.add_term(aisq,chol_idx,[p,q],[dpm,dqp],[0,0])
            #        if self.apply_spin_down: 
            #            self.add_term(aisq,chol_idx,[p,q],[dpm,dqp],[1,1])
            #    if self.apply_spin_down:
            #        self.add_term(aisq,chol_idx,[p,q],[dpm,dqp],[0,1])
            #        self.add_term(aisq,chol_idx,[q,p],[dqp,dpm],[0,1])
            #    else:
            #        self.add_term(aisq,chol_idx,[p],[dpm],[0])
            #        self.add_term(aisq,chol_idx,[q],[dqp],[0])
            for p in range(self.nbasis):
                dp = ek[p]/ai
                if self.apply_spin_down:
                    self.add_term(aisq,chol_idx,[p,p],[-dp,dp],[0,1])
                    self.add_term(aisq,chol_idx,[p,p],[dp,-dp],[0,1])
                for q in range(p+1,self.nbasis):
                    dq = ek[q]/ai
                    self.add_term(aisq,chol_idx,[p,q],[-dp,dq],[0,0])
                    self.add_term(aisq,chol_idx,[p,q],[dp,-dq],[0,0])
                    if self.apply_spin_down: 
                        self.add_term(aisq,chol_idx,[p,q],[-dp,dq],[1,1])
                        self.add_term(aisq,chol_idx,[p,q],[dp,-dq],[1,1])
                        self.add_term(aisq,chol_idx,[p,q],[-dp,dq],[0,1])
                        self.add_term(aisq,chol_idx,[p,q],[dp,-dq],[0,1])
                        self.add_term(aisq,chol_idx,[q,p],[dq,-dp],[0,1])
                        self.add_term(aisq,chol_idx,[q,p],[-dq,dp],[0,1])

    @plum.dispatch
    def local_energy(self,walkers:UHFWalkers,trial:SingleDet):
        E1,SC = trial.compute_E1(walkers)
        E2 = trial.compute_chol_E2(SC,False)
        return E1+E2,E1,E2
    
    @plum.dispatch
    def local_energy(self,walkers,trial:SingleDetGHF):
        E1,SC = trial.compute_E1(walkers)
        E2 = trial.compute_chol_E2(SC,True)
        return E1+E2,E1,E2

