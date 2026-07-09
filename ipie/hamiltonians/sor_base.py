import numpy as np
from ipie.utils.backend import arraylib as xp
#from ipie.walkers.lafqmc_uhf_walkers import UHFWalkers
#from ipie.walkers.lafqmc_ghf_walkers import GHFWalkers
#from ipie.trial_wavefunction.lafqmc_single_det import SingleDet 
#from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF

class SumOfRotationBase:

    def __init__(self,apply_spin_down=True,importance_sample=False,thresh=1e-6):
        self.chol_basis = []
        self.chol_bands = []
        self.chol_ix = dict()
        self.term_dict = dict() 
        self.apply_spin_down = apply_spin_down
        self.thresh = thresh

    def add_h1_term(self,key,new_term):
        if key not in self.term_dict:
            self.term_dict[key] = new_term
            return
        term = self.term_dict[key]
        a1,d1 = term['a'],term['d'][0]
        a2,d2 = new_term['a'],new_term['d'][0]

        d1 = a1 * d1 + a2 * d2
        a1 += a2
        d1 /= a1

        term['a'] = a1
        term['d'] = [d1]
        self.term_dict[key] = term
        
    def add_term(self,a,chol_ix,p,d,s):
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
                pass
        else:
            raise ValueError

        key = chol_ix,tuple(p),tuple(s)
        term = {'a':a,'d':d}
        r = len(p)
        if r==1:
            self.add_h1_term(key,term)
        else:
            if key not in self.term_dict:
                self.term_dict[key] = []
            self.term_dict[key].append(term)

    def _decompose_h1(self,ek,at,iprint=0):
        self.nbasis = ek.size
        chol_ix = len(self.chol_basis)-1
        self.chol_ix['h1'] = [chol_ix]

        if iprint>0:
            print('at=',at)
            print('bands=',ek)
        assert at>xp.amax(xp.fabs(ek))

        for k,e_k in enumerate(ek):
            self.add_term(at,chol_ix,[k],[-e_k/at],[0])
            if self.apply_spin_down:
                self.add_term(at,chol_ix,[k],[-e_k/at],[1])
        return ek

    def parse_decomposition(self,iprint=0):
        self.chol_basis = xp.asarray(self.chol_basis)
        self.chol_bands = xp.asarray(self.chol_bands)
        for key in self.chol_ix:
            self.chol_ix[key] = xp.asarray(self.chol_ix[key])
        self.nchol = self.chol_basis.shape[0]
        self.chol_basis2 = {'eye':xp.eye(self.nbasis)}
        for i,Ui in enumerate(self.chol_basis):
            for j in range(i+1,self.nchol):
                Uj = self.chol_basis[j]
                self.chol_basis2[i,j] = xp.dot(Ui.T,Uj)

        self.a = []
        self.ix2key = []
        p_dict = dict() 
        d_dict = dict() 
        ix_dict = dict()
        for term_key in self.term_dict:
            chol_ix,p,spin = term_key
            assert spin in [(0,),(1,),(0,0),(1,1),(0,1)]
            if len(p)==1:
                term = self.term_dict[term_key] 
                if xp.fabs(term['d'][0])<self.thresh:
                    #print('not included',key,rot.d,rot.d2,rot.a)
                    continue
                terms = [term]
            else:
                terms = self.term_dict[term_key] 

            for term in terms:
                key = chol_ix,spin 
                a,d = term['a'],term['d']
                self.a.append(a)

                ix = len(self.a)-1
                if key not in p_dict:
                    p_dict[key] = []
                if key not in d_dict:
                    d_dict[key] = []
                if key not in ix_dict:
                    ix_dict[key] = []
                p_dict[key].append(p)
                d_dict[key].append(d)
                ix_dict[key].append(ix)

                i = len(p_dict[key])-1
                self.ix2key.append((key,i))
                if iprint>0:
                    print(ix,key,i,p,d,a)

        self.term_dict = dict() 
        for key in p_dict:
            p = xp.asarray(p_dict[key])
            d = xp.asarray(d_dict[key])
            ix = xp.asarray(ix_dict[key])
            self.term_dict[key] = {'p':p,'d':d,'ix':ix}

        self.a = xp.asarray(self.a)
        self.Lambda = self.a.sum()
        self.a /= self.Lambda
        self.nterms = self.a.size
        if iprint>0:
            print('Lambda=',self.Lambda)
            print('normalization=',xp.fabs(self.a).sum())
            print('number of terms=',self.nterms)

    def parse_samples(self,ixs):
        w_dict = dict()
        i_dict = dict()
        for w,ix in enumerate(ixs):
            key,i = self.ix2key[ix]
            if key not in w_dict:
                w_dict[key] = []
            w_dict[key].append(w)
            if key not in i_dict:
                i_dict[key] = []
            i_dict[key].append(i)

        samples = dict()
        for key in w_dict:
            w = xp.asarray(w_dict[key])
            i = xp.asarray(i_dict[key])
            u,d,u2 = self.get_batch_ud(key,i)
            samples[key] = {'w':w,'i':i,'u':u,'d':d,'u2':u2}
        return samples

    def get_batch_ud(self,key,i):
        chol_ix,spin = key
        dat = self.term_dict[key]
        p,d = dat['p'][i],dat['d'][i]
        u = self.chol_basis[chol_ix]
        u = xp.asarray([u[:,pi] for pi in p])
        
        nw,r = p.shape
        u2 = [None] * self.nchol 
        for i in range(self.nchol):
            if i<chol_ix:
                U2 = self.chol_basis2[i,chol_ix]
            elif i>chol_ix:
                U2 = self.chol_basis2[chol_ix,i].T
            else:
                U2 = self.chol_basis2['eye'] 
            u2[i] = [U2[:,pi] for pi in p]
        u2 = xp.asarray(u2)
        return u,d,u2

    def get_single_ud(self,ix):
        key,i = self.ix2key[ix]
        chol_ix,spin = key
        dat = self.term_dict[key]
        p,d = dat['p'][i],dat['d'][i]
        u = self.chol_basis[chol_ix]
        return u,d,p,spin 

    def get_rotation_matrix(self,ix):
        v,ds,ps,spin = self.get_single_ud(ix)
        U = [None] * 2
        if spin==(0,1):
            for s,p in enumerate(ps):
                diag = xp.ones(v.shape[0])
                diag[p] += ds[s]
                U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        else:
            s = spin[0]
            diag = xp.ones(v.shape[0])
            diag[ps] += ds
            U[s] = xp.einsum('xp,yp,p->xy',v,v,diag)
        return U

class HubbardSOR(SumOfRotationBase):

    def __init__(self,U,**kwargs):
        super().__init__(**kwargs)
        self.hubbard_U = U

    def decompose_h1(self,h1e,at,iprint=0):
        h1e = xp.asarray(h1e)
        ek,vk = xp.linalg.eigh(h1e) 
        self.chol_basis.append(vk)
        self.chol_bands.append(ek)

        self._decompose_h1(ek+0.5*self.hubbard_U,at,iprint=iprint)

    def decompose_h2(self,gu,iprint=0,nelec=None):
        self.chol_basis.append(xp.eye(self.nbasis))
        self.chol_bands.append(xp.ones(self.nbasis))
        chol_ix = len(self.chol_basis)-1
        self.chol_ix['U'] = [chol_ix]

        ai = self.hubbard_U/(np.cosh(gu)-1)/4
        dp = xp.exp(gu)-1.
        dm = xp.exp(-gu)-1.
        if iprint>0:
            print(f'eta={gu},ai={ai}')
        assert self.apply_spin_down
        for i in range(self.nbasis): 
            self.add_term(ai,chol_ix,[i,i],[dp,dm],[0,1])
            self.add_term(ai,chol_ix,[i,i],[dm,dp],[0,1])

    def local_energy(self,walkers):
        tr = walkers.compute_local_energy_intermediates(self.chol_bands,tr_ixs=self.chol_ix['h1'])
        E1 = tr[0][:,0]+tr[1][:,0]
    
        D = walkers.compute_1rdm_diag(self.chol_ix['U'][0])
        E2 = (D[0,0]*D[1,1]).sum(axis=1)
        if (0,1) not in D:
            E2 *= self.hubbard_U
            return E1+E2,E1,E2
        E2 -= (D[0,1]*D[1,0]).sum(axis=1)
        E2 *= self.hubbard_U
        return E1+E2,E1,E2

class QCSOR(SumOfRotationBase):

    def decompose_h1(self,h1e,at,iprint=0):
        h1e = xp.asarray(h1e)
        ek,vk = xp.linalg.eigh(h1e) 
        self.chol_basis.append(vk)
        self.chol_bands.append(ek)

        self._decompose_h1(ek,at,iprint=iprint)

    def decompose_h2(self,chol,ai,iprint=0):
        aisq = ai**2/2
        if iprint>0:
            print('ai,aisq=',ai,aisq)
        self.chol_ix['chol'] = []
        chol = xp.asarray(chol)
        for L in chol:
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            self.chol_bands.append(ek)
            chol_ix = len(self.chol_basis)-1
            self.chol_ix['chol'].append(chol_ix)

            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)
            assert ai>xp.amax(xp.fabs(ek))

            for p in range(self.nbasis):
                dp = ek[p]/ai
                if self.apply_spin_down:
                    self.add_term(aisq,chol_ix,[p,p],[-dp,dp],[0,1])
                    self.add_term(aisq,chol_ix,[p,p],[dp,-dp],[0,1])
                for q in range(p+1,self.nbasis):
                    dq = ek[q]/ai
                    self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[0,0])
                    self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[0,0])
                    if self.apply_spin_down: 
                        self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[1,1])
                        self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[1,1])
                        self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[0,1])
                        self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[0,1])
                        self.add_term(aisq,chol_ix,[q,p],[dq,-dp],[0,1])
                        self.add_term(aisq,chol_ix,[q,p],[-dq,dp],[0,1])

    def local_energy(self,walkers):
        h1_ix = self.chol_ix['h1'][0]
        chol_ix = self.chol_ix['chol']
        tr,mat = walkers.compute_local_energy_intermediates(self.chol_bands,mat_ixs=chol_ix)
        E1 = tr[0][:,h1_ix]+tr[1][:,h1_ix]

        tr = [tri[:,chol_ix] for tri in tr] 
        E2 = ((tr[0]+tr[1])**2).sum(axis=1) 
        E2 -= xp.einsum('wdij,wdji->w',mat[0],mat[0])
        E2 -= xp.einsum('wdij,wdji->w',mat[1],mat[1])
        if walkers.UBS.shape[2]==walkers.nbasis*2:
            E2 -= 2.*xp.einsum('wdij,wdji->w',mat[0],mat[1])
        E2 *= .5
        return E1+E2,E1,E2
    
