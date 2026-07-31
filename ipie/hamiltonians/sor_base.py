import numpy as np
from ipie.utils.backend import arraylib as xp

class SumOfRotationBase:

    def __init__(self,nbasis,thresh=1e-6):
        self.nbasis = nbasis
        self.thresh = thresh

        self.chol_basis = []
        self.chol_bands = []
        self.chol_ix = dict()
        self.term_dict = dict() 
        self.chol = None

        self.run_2body_first = False
        self.apply_spin_down = True
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

    def decompose_h1(self,h1e,at,iprint=0):
        if not self.run_2body_first:
            raise ValueError('Run 2-body decomposition first!')

        self.h1e = xp.asarray(h1e)
        ek,vk = xp.linalg.eigh(self.h1e+self.v0) 
        self.chol_basis.append(vk)
        self.chol_bands.append(ek)

        chol_ix = len(self.chol_basis)-1
        self.chol_ix['h1'] = [chol_ix]

        if iprint>0:
            print('bands=',ek)
            print('at=',at)

        for k in range(ek.size):
            self.add_term(at,chol_ix,[k],[-ek[k]/at],[0])
            if self.apply_spin_down:
                self.add_term(at,chol_ix,[k],[-ek[k]/at],[1])
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
                if chol_ix==self.chol_ix['h1'][0]:
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

    def decompose_h2(self,U,gu,iprint=0):
        self.hubbard_U = U
        self.chol_basis.append(xp.eye(self.nbasis))
        self.chol_bands.append(xp.ones(self.nbasis))
        chol_ix = len(self.chol_basis)-1
        self.chol_ix['U'] = [chol_ix]

        ai = self.hubbard_U/(np.cosh(gu)-1.)/4.
        dp = xp.exp(gu)-1.
        dm = xp.exp(-gu)-1.
        if iprint>0:
            print(f'eta={gu},ai={ai}')
        for i in range(self.nbasis): 
            self.add_term(ai,chol_ix,[i,i],[dp,dm],[0,1])
            self.add_term(ai,chol_ix,[i,i],[dm,dp],[0,1])

        self.v0 = np.eye(self.nbasis) * self.hubbard_U/2.
        self.run_2body_first = True

    def local_energy(self,walkers,trial):
        if walkers.fast_eloc:
            Lambda1,Lambda2,Lambda = self.Lambda
            E1 = Lambda1 - walkers.E1*Lambda
            E2 = Lambda2 - walkers.E2*Lambda
            return E1+E2,E1,E2

        if 'UDU' in walkers.buff_names:
            D = walkers.get_UDU()

            ix = self.chol_ix['h1'][0]
            E1 = xp.einsum('p,wpp->w',self.chol_bands[ix],D[0,0][:,ix])
            E1 += xp.einsum('p,wpp->w',self.chol_bands[ix],D[1,1][:,ix])
    
            ix = self.chol_ix['U'][0]
            E2 = xp.einsum('wpp,wpp->w',D[0,0][:,ix],D[1,1][:,ix])
            if (0,1) in D:
                E2 -= xp.einsum('wpp,wpp->w',D[0,1][:,ix],D[1,0][:,ix])
        else:
            SC = walkers.compute_SC(trial)
            E1 = walkers.compute_E1(trial,SC)
            Daa,Dbb,Dab,Dba = walkers.compute_1rdm_diag(trial,SC)
            E2 = xp.einsum('wp,wp->w',Daa,Dbb)
            if Dab is not None:
                E2 -= xp.einsum('wp,wp->w',Dab,Dba)
        E2 *= self.hubbard_U
        return E1+E2,E1,E2

class QCSOR(SumOfRotationBase):

    def decompose_h2_method_1(self,chol,ai,iprint=0,apply_spin_down=True):
        self.chol_ix['chol'] = []
        self.chol = xp.asarray(chol)
        aisq = ai**2/2.
        if iprint>0:
            print('ai=',ai,aisq)
        for i,L in enumerate(self.chol):
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            self.chol_bands.append(ek)
            chol_ix = len(self.chol_basis)-1
            self.chol_ix['chol'].append(chol_ix)

            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)

            for p in range(self.nbasis):
                dp = ek[p]/ai
                if apply_spin_down:
                    self.add_term(aisq,chol_ix,[p,p],[-dp,dp],[0,1])
                    self.add_term(aisq,chol_ix,[p,p],[dp,-dp],[0,1])
                for q in range(p+1,self.nbasis):
                    dq = ek[q]/ai
                    self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[0,0])
                    self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[0,0])
                    if apply_spin_down: 
                        self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[1,1])
                        self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[1,1])
                        self.add_term(aisq,chol_ix,[p,q],[-dp,dq],[0,1])
                        self.add_term(aisq,chol_ix,[p,q],[dp,-dq],[0,1])
                        self.add_term(aisq,chol_ix,[q,p],[dq,-dp],[0,1])
                        self.add_term(aisq,chol_ix,[q,p],[-dq,dp],[0,1])
        self.run_2body_first = True
        self.apply_spin_down = apply_spin_down

    def decompose_h2_method_2(self,chol,gu,iprint=0,thresh=1e-6):
        self.chol_ix['chol'] = []
        self.chol = xp.asarray(chol)
        ai = 1./(np.cosh(gu)-1.)/2.
        dp = xp.exp(gu)-1.
        dm = xp.exp(-gu)-1.
        if iprint>0:
            print('ai=',ai)
        for i,L in enumerate(self.chol):
            ek,vk = xp.linalg.eigh(L) 
            self.chol_basis.append(vk)
            self.chol_bands.append(ek)
            chol_ix = len(self.chol_basis)-1
            self.chol_ix['chol'].append(chol_ix)

            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)

            for p in range(self.nbasis):
                ep = ek[p]
                if xp.fabs(ep)<thresh:
                    continue

                epp = ep**2
                ap = epp*ai/2.
                self.add_term(ap,chol_ix,[p,p],[dp,dm],[0,1])
                self.add_term(ap,chol_ix,[p,p],[dm,dp],[0,1])

                for q in range(p+1,self.nbasis):
                    eq = ek[q]
                    if xp.fabs(eq)<thresh:
                        continue

                    epq = ep*eq
                    dp1,dp2 = dp,dm
                    if epq>0.:
                        dq1,dq2 = dm,dp
                    else:
                        dq1,dq2 = dp,dm

                    dpx,dpz = dp1,-dp1
                    dpy,dpw = dp2,-dp2
                    dqx,dqy = dq1,-dq1
                    dqz,dqw = dq2,-dq2
                    
                    apq = -epq/dpx/dqx#/2.
                    assert apq>0.
                    self.add_term(apq,chol_ix,[p,q],[dpx,dqx],[0,0])
                    apq = -epq/dpy/dqy#/2.
                    assert apq>0.
                    self.add_term(apq,chol_ix,[p,q],[dpy,dqy],[0,1])
                    apq = -epq/dpz/dqz#/2.
                    assert apq>0.
                    self.add_term(apq,chol_ix,[q,p],[dqz,dpz],[0,1])
                    apq = -epq/dpw/dqw#/2.
                    assert apq>0.
                    self.add_term(apq,chol_ix,[p,q],[dpw,dqw],[1,1])
            v0 = ek**2/2. 
            self.v0 += xp.dot(vk*v0[None,:],vk.T)
        self.run_2body_first = True

    def local_energy(self,walkers,trial):
        if walkers.fast_eloc:
            Lambda1,Lambda2,Lambda = self.Lambda
            E1 = Lambda1 - walkers.E1*Lambda
            E2 = Lambda2 - walkers.E2*Lambda
            return E1+E2,E1,E2

        if 'UDU' in walkers.buff_names:
            D = walkers.get_UDU()

            ix = self.chol_ix['h1'][0]
            E1 = xp.einsum('p,wpp->w',self.chol_bands[ix],D[0,0][:,ix])
            E1 += xp.einsum('p,wpp->w',self.chol_bands[ix],D[1,1][:,ix])

            ix = self.chol_ix['chol']
            Sigma = self.chol_bands[ix]
            Daa,Dbb = D[0,0][:,ix],D[1,1][:,ix]
            Daa = Sigma[None,:,:,None]*Daa
            Dbb = Sigma[None,:,:,None]*Dbb
            tr = xp.einsum('wdpp->wd',Daa)
            tr += xp.einsum('wdpp->wd',Dbb)
            E2 = (tr**2).sum(axis=1) 

            E2 -= xp.einsum('wdpq,wdqp->w',Daa,Daa)
            E2 -= xp.einsum('wdpq,wdqp->w',Dbb,Dbb)
            if (0,1) in D: 
                Dab,Dba = D[0,1][:,ix],D[1,0][:,ix]
                Dab = Sigma[None,:,:,None]*Dab
                Dba = Sigma[None,:,:,None]*Dba
                E2 -= 2.*xp.einsum('wdpq,wdqp->w',Dab,Dba)
            E2 *= .5
        else:
            SC = walkers.compute_SC(trial)
            E1 = walkers.compute_E1(trial,SC)
            E2 = walkers.compute_chol(trial,SC)
        return E1+E2,E1,E2
    
