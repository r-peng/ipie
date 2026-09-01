import numpy as np
from ipie.utils.backend import arraylib as xp

def parse_density(density,ixs,a,thresh):
    if density is None:
        return a,None
    dmin = min([density[s][p] for (s,p) in ixs])
    if dmin<thresh: 
        return a,None
    return a/dmin,density

def get_coeffs_same_sign(ep,eq,p,q,sp,sq,sign_p,sign_q,a,uniform,density,thresh):
    np_,nq_ = None,None
    if density is not None:
        np_,nq_ = density[sp][p],density[sq][q]
        if np_<thresh or nq_<thresh:
            np_,nq_ = None,None

    if np_ is None or nq_ is None:
        if uniform=='coefficient':
            dp = sign_p*ep/a
            dq = sign_q*eq/a
        else:
            dp = sign_p*np.sign(ep)/a
            dq = sign_q*np.sign(eq)/a
    else:
        if np_<nq_:
            dp = sign_p/a
            dq = (sign_q/a)*(np_/nq_)
        else:
            dp = (sign_p/a)*(nq_/np_)
            dq = sign_q/a
    apq = -ep*eq/(dp*dq)
    #print(ep,eq,np_,nq_,dp,dq,apq)
    return dp,dq,apq


class SumOfRotationBase:

    def __init__(self,nbasis,thresh=1e-6,
            has_aa=True,has_bb=True,has_ab=True,
            compute_local_magnetization=None):
        self.nbasis = nbasis
        self.thresh = thresh
        self.has_aa = has_aa
        self.has_ab = has_ab
        self.has_bb = has_bb

        self.chol_basis = []
        self.term_dict = dict() 
        self.chol = None
        self.compute_local_magnetization = compute_local_magnetization

        self.run_2body_first = False
        self.v0 = 0. 

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
        assert a>0.
        if a<self.thresh:
            return

        typ = None
        r = len(p)
        if r==1:
            if np.fabs(d[0])<self.thresh:
                return
        elif r==2:
            if np.fabs(d[0])<self.thresh and np.fabs(d[1])<self.thresh:
                return
            elif np.fabs(d[0])<self.thresh:
                p = [p[1]]
                d = [d[1]]
                s = [s[1]]
            elif np.fabs(d[1])<self.thresh:
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

    def decompose_h1(self,h1e,dt,uniform='coefficient',iprint=0):
        if not self.run_2body_first:
            raise ValueError('Run 2-body decomposition first!')
        assert uniform in ['coefficient','rotation']

        self.h1e = h1e
        ek,vk = np.linalg.eigh(self.h1e+self.v0) 
        self.chol_basis.append(vk)
        chol_ix = len(self.chol_basis)-1

        if iprint>0:
            print('1-body decomposition: ')
            print('bands=',ek)
            print('ai=',1./dt)

        for p in range(ek.size):
            if np.fabs(ek[p])<self.thresh:
                continue

            if uniform=='coefficient':
                ap = 1./dt
            else:
                ap = np.fabs(ek[p]/dt)
            dp = -ek[p]/ap
            assert np.fabs(dp)<1.

            self.add_term(ap,chol_ix,[p],[dp],[0])
            if self.has_ab or self.has_bb:
                self.add_term(ap,chol_ix,[p],[dp],[1])
        self.h1e = xp.asarray(self.h1e)
        return ek

    def parse_decomposition(self,iprint=0):
        self.chol_basis = xp.asarray(self.chol_basis)
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
        E1_ixs = []
        E2_ixs = []
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
                if len(d)==1:
                    E1_ixs.append(ix)
                else:
                    E2_ixs.append(ix)

                if iprint>1:
                    print(ix,key,i,p,d,a)

        self.term_dict = dict() 
        for key in p_dict:
            p = xp.asarray(p_dict[key])
            d = xp.asarray(d_dict[key])
            ix = xp.asarray(ix_dict[key])
            self.term_dict[key] = {'p':p,'d':d,'ix':ix}
        self.E1_ixs = xp.asarray(E1_ixs)
        self.E2_ixs = xp.asarray(E2_ixs)

        self.a = xp.asarray(self.a)
        Lambda1 = self.a[self.E1_ixs].sum()
        Lambda2 = self.a[self.E2_ixs].sum()
        self.Lambda = xp.asarray([Lambda1,Lambda2,Lambda1+Lambda2])
        self.a /= self.Lambda[-1]
        self.nterms = self.a.size
        if iprint>0:
            print('Lambda=',self.Lambda)
            print('normalization=',xp.fabs(self.a).sum())
            print('number of terms=',self.nterms)
            #print('a=',self.a)

    def parse_samples(self,ixs,active=None):
        w_dict = dict()
        i_dict = dict()
        if active is None:
            active = list(range(ixs.size))
        for w,ix in zip(active,ixs):
            key,i = self.ix2key[ix]
            if key not in w_dict:
                w_dict[key] = []
            w_dict[key].append(w)
            if key not in i_dict:
                i_dict[key] = []
            i_dict[key].append(i)

        self.samples = dict()
        for key in w_dict:
            w = xp.asarray(w_dict[key])
            i = xp.asarray(i_dict[key])
            self.samples[key] = {'w':w,'i':i}

    def get_batch_ud(self,key,i):
        chol_ix,spin = key
        dat = self.term_dict[key]
        p,d = dat['p'][i],dat['d'][i]
        u = self.chol_basis[chol_ix][:,p]
        
        nw,r = p.shape
        u2 = [None] * self.nchol 
        for i in range(self.nchol):
            if i<chol_ix:
                U2 = self.chol_basis2[i,chol_ix]
            elif i>chol_ix:
                U2 = self.chol_basis2[chol_ix,i].T
            else:
                U2 = self.chol_basis2['eye'] 
            u2[i] = U2[:,p]
        u2 = xp.asarray(u2)
        return p,d,u,u2

    def get_term(self,ix):
        key,i = self.ix2key[ix]
        chol_ix,spin = key
        dat = self.term_dict[key]
        p,d = dat['p'][i],dat['d'][i]
        return chol_ix,spin,p,d

    def get_rotation_matrix(self,ix):
        chol_ix,spin,ps,ds = self.get_term(ix)
        v = self.chol_basis[chol_ix]
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

    def decompose_h2(self,U,dt,iprint=0,param='eta'):
        assert param in ['eta','lambda']
        self.hubbard_U = U
        self.chol_basis.append(np.eye(self.nbasis))
        chol_ix = len(self.chol_basis)-1

        d = np.sqrt(self.hubbard_U*dt)
        if param=='eta':
            ai = self.hubbard_U/(np.cosh(d)-1.)/4.
            dp = xp.exp(d)-1.
            dm = xp.exp(-d)-1.
            self.v0 = np.eye(self.nbasis) * self.hubbard_U/2.
        else:
            dp,dm = d,-d
            ai = self.hubbard_U/d**2/2. 
        assert np.fabs(dp)<1.
        assert np.fabs(dm)<1.

        if iprint>0:
            print('Hubbard 2-body decomposition: ')
            print('coefficient =',ai)
            print('rotation =',d)

        for i in range(self.nbasis): 
            self.add_term(ai,chol_ix,[i,i],[dp,dm],[0,1])
            self.add_term(ai,chol_ix,[i,i],[dm,dp],[0,1])
       
        self.run_2body_first = True

class QCSOR(SumOfRotationBase):

    def decompose_h2_old(self,chol,ai,iprint=0):
        self.chol_ix['chol'] = []
        self.chol = xp.asarray(chol)
        aisq = ai**2/2.
        if iprint>0:
            print('ai=',ai,aisq)
        for i,L in enumerate(self.chol):
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            #self.chol_bands.append(ek)
            chol_ix = len(self.chol_basis)-1
            #self.chol_ix['chol'].append(chol_ix)

            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)

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
        self.run_2body_first = True

    def decompose_h2(self,chol,dt,iprint=0,uniform='coefficient',trial=None):
        assert uniform in ['coefficient','rotation']
        self.chol = chol
        if iprint>0:
            print('2-body decomposition: ')

        density = None
        for i,L in enumerate(self.chol):
            ek,vk = np.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_ix = len(self.chol_basis)-1
            if trial is not None:
                density = trial.compute_density(U=vk)
                if iprint>1:
                    print('density up=',density[0])
                    print('density down=',density[1])

            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)

            if self.has_aa and self.has_ab and self.has_bb:
                if iprint>1:
                    print('all terms')
                self._decompose_chol(chol_ix,ek,dt,uniform,density=density)
            elif self.has_aa and (not self.has_ab) and (not self.has_bb):
                if iprint>1:
                    print('up only')
                self._decompose_chol_up_only(chol_ix,ek,dt,uniform,density=density)
            elif self.has_ab and (not self.has_aa) and (not self.has_bb):
                if iprint>1:
                    print('ab only')
                self._decompose_chol_ab_only(chol_ix,ek,dt,uniform,density=density)
            else:
                raise NotImplementedError

        self.chol = xp.asarray(self.chol)
        self.run_2body_first = True

    def _decompose_chol(self,chol_ix,ek,dt,uniform,density=None):
        a = 1./np.sqrt(dt)
        d = 1./a
        def get_coeffs(ep,eq,p,q,sp,sq,sign_p,sign_q):
            return get_coeffs_same_sign(ep,eq,p,q,sp,sq,sign_p,sign_q,a,uniform,density,self.thresh)
        for p in range(self.nbasis):
            ep = ek[p]
            if np.fabs(ep)<self.thresh:
                continue
            ixs = [(s,p) for s in (0,1)]
            d0,d1,app = get_coeffs(ep,ep,p,p,0,1,-1,1)
            self.add_term(app/2.,chol_ix,[p,p],[d0,d1],[0,1])
            self.add_term(app/2.,chol_ix,[p,p],[-d0,-d1],[0,1])

            for q in range(p+1,self.nbasis):
                eq = ek[q]
                if np.fabs(eq)<self.thresh:
                    continue
                #print('p,q,ep,eq=',p,q,ep,eq)

                if uniform=='coefficient':
                    ap,aq = a,a
                else:
                    ap = np.fabs(ep)*a
                    aq = np.fabs(eq)*a

                dp,dq = -ep/ap,eq/aq
                assert np.fabs(dp)<1.
                assert np.fabs(dq)<1.
                apq = ap*aq

                self.add_term(apq,chol_ix,[p,q],[dp,dq],[0,0])
                self.add_term(apq,chol_ix,[p,q],[-dp,-dq],[0,1])
                self.add_term(apq,chol_ix,[q,p],[-dq,-dp],[0,1])
                self.add_term(apq,chol_ix,[p,q],[dp,dq],[1,1])

    def _decompose_chol_up_only(self,chol_ix,ek,dt,uniform,density=None):
        a = 1./np.sqrt(dt)
        d = 1./a
        def get_coeffs(ep,eq,p,q,sp,sq,sign_p,sign_q):
            return get_coeffs_same_sign(ep,eq,p,q,sp,sq,sign_p,sign_q,a,uniform,density,self.thresh)
        for p in range(self.nbasis):
            ep = ek[p]
            if np.fabs(ep)<self.thresh:
                continue
            for q in range(p+1,self.nbasis):
                eq = ek[q]
                if np.fabs(eq)<self.thresh:
                    continue

                if ep*eq<0:
                    if uniform=='coefficient': 
                        ap,aq = a,a
                    else:
                        ap = np.fabs(ep)*a
                        aq = np.fabs(eq)*a

                    dp,dq = ep/ap,eq/aq
                    assert np.fabs(dp)<1.
                    assert np.fabs(dq)<1.
                    apq = ap*aq/2.
    
                    self.add_term(apq,chol_ix,[p,q],[-dp,dq],[0,0])
                    self.add_term(apq,chol_ix,[p,q],[dp,-dq],[0,0])
                else:
                    ixs = [(0,x) for x in (p,q)]
                    dp,dq,apq = get_coeffs(ep,eq,p,q,0,0,-1,1)
                    self.add_term(apq/2.,chol_ix,[p,q],[dp,dq],[0,0])
                    self.add_term(apq/2.,chol_ix,[p,q],[-dp,-dq],[0,0])

    def _decompose_chol_ab_only(self,chol_ix,ek,dt,uniform,density=None):
        a = 1./np.sqrt(dt)
        d = 1./a
        def get_coeffs(ep,eq,p,q,sp,sq,sign_p,sign_q):
            return get_coeffs_same_sign(ep,eq,p,q,sp,sq,sign_p,sign_q,a,uniform,density,self.thresh)

        for p in range(self.nbasis):
            ep = ek[p]
            if np.fabs(ep)<self.thresh:
                continue
            ixs = [(s,p) for s in (0,1)]
            d0,d1,app = get_coeffs(ep,ep,p,p,0,1,-1,1)
            self.add_term(app/2.,chol_ix,[p,p],[d0,d1],[0,1])
            self.add_term(app/2.,chol_ix,[p,p],[-d0,-d1],[0,1])

            for q in range(p+1,self.nbasis):
                eq = ek[q]
                if np.fabs(eq)<self.thresh:
                    continue
                #print('p,q,ep,eq=',p,q,ep,eq)

                if ep*eq<0:
                    if uniform=='coefficient':
                        ap,aq = a,a
                    else:
                        ap = np.fabs(ep)*a
                        aq = np.fabs(eq)*a

                    dp,dq = ep/ap,eq/aq
                    assert np.fabs(dp)<1.
                    assert np.fabs(dq)<1.
                    apq = ap*aq/2.

                    self.add_term(apq,chol_ix,[p,q],[-dp,dq],[0,1])
                    self.add_term(apq,chol_ix,[p,q],[dp,-dq],[0,1])
                    self.add_term(apq,chol_ix,[q,p],[dq,-dp],[0,1])
                    self.add_term(apq,chol_ix,[q,p],[-dq,dp],[0,1])
                else:
                    ixs = [(0,p),(1,q)]
                    dp,dq,apq = get_coeffs(ep,eq,p,q,0,1,-1,1)
                    self.add_term(apq/2.,chol_ix,[p,q],[dp,dq],[0,1])
                    self.add_term(apq/2.,chol_ix,[p,q],[-dp,-dq],[0,1])

                    ixs = [(1,p),(0,q)]
                    dp,dq,apq = get_coeffs_same_sign(ep,eq,p,q,1,0,-1,1,a,uniform,density,self.thresh)
                    self.add_term(apq/2.,chol_ix,[q,p],[dq,dp],[0,1])
                    self.add_term(apq/2.,chol_ix,[q,p],[-dq,-dp],[0,1])


