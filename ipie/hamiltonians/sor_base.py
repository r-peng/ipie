import numpy as np
from ipie.utils.backend import arraylib as xp

def _get_coeffs(a,e,uniform):
    ep,eq = e 
    if uniform=='coefficient':
        ap,aq = a,a
    else:
        ap = np.fabs(ep)*a
        aq = np.fabs(eq)*a
    coeff = ap*aq
    dp,dq = -ep/ap,eq/aq
    return coeff,dp,dq

def _delta2d(delta,rho=0):
    return delta/(1.-delta*rho)

class SumOfRotationBase:

    def __init__(self,nbasis,thresh=1e-6,decomp_type='all',
            compute_local_magnetization=None):
        self.nbasis = nbasis
        self.thresh = thresh
        assert decomp_type in ['all','aa_only','ab_only']
        self.decomp_type = decomp_type

        self.chol_basis = []
        self.term_dict = dict() 
        self.chol = None
        self.compute_local_magnetization = compute_local_magnetization

        self.run_2body_first = False
        self.v0 = np.zeros((2,self.nbasis,self.nbasis)) 
        self.const = 0.

    def add_term(self,chol_ix,s,a,f,p,d):
        key = chol_ix,tuple(s)
        if key in self.term_dict:
            terms = self.term_dict[key]
        else:
            terms = {'a':[],'f':[],'p':[],'d':[]}
        terms['a'].append(a)
        terms['f'].append(f)
        terms['p'].append(p)
        terms['d'].append(d)
        self.term_dict[key] = terms

    def decompose_h1(self,h1e,dt,uniform='coefficient',iprint=0,trial=None):
        if iprint>0:
            print('1-body decomposition: ')
            print('a=',1./dt)
        if not self.run_2body_first:
            raise ValueError('Run 2-body decomposition first!')
        assert uniform in ['coefficient','rotation']

        for s,v0 in enumerate(self.v0):
            if s==1 and self.decomp_type=='aa_only':
                continue

            ek,vk = np.linalg.eigh(h1e+v0) 
            if iprint>0:
                print(f'spin={s} bands:',ek)
            self.chol_basis.append(vk)
            chol_ix = len(self.chol_basis)-1
            rho = np.zeros(self.nbasis)
            if trial is not None:
                rho = trial.compute_density(s,U=vk)

            for p,ep in enumerate(ek):
                if np.fabs(ep)<self.thresh:
                    continue

                if uniform=='coefficient':
                    ap = 1./dt
                else:
                    ap = np.fabs(ep/dt)
                delta_p = -ep/ap
                dp = _delta2d(delta_p,rho=rho[p])
                assert np.fabs(dp)<1.
                fp = 1.+dp*rho[p]
                self.const += ep*rho[p]
                self.add_term(chol_ix,[s],ap,fp,[p],[dp])
        self.h1e = xp.asarray(h1e)
        return ek

    def add_2body_term(self,chol_ix,s,a,p,delta,rho):
        d = [_delta2d(delta[i],rho=rho[i]) for i in (0,1)]
        assert np.fabs(d[0])<1.
        assert np.fabs(d[1])<1.
        f = [1.+d[i]*rho[i] for i in (0,1)]
        self.add_term(chol_ix,s,a,f[0]*f[1],p,d)

    def _add_2body_hubbard(self,chol_ix,s,a,pq,e,rho,uniform='coefficient'):
        coeff,delta_p,delta_q = _get_coeffs(a,e,uniform)
        coeff /= 2.
        self.add_2body_term(chol_ix,s,coeff,pq,[delta_p,delta_q],rho)
        self.add_2body_term(chol_ix,s,coeff,pq,[-delta_p,-delta_q],rho)

    def _compute_v0_hubbard(self,ek,rho):
        v0 = np.zeros((2,self.nbasis))
        if self.decomp_type=='aa_only':
            return v0
        eksq = ek**2
        for s in (0,1):
            v0[s] = eksq*rho[:,1-s]
        self.const -= (eksq*rho[:,0]*rho[:,1]).sum()
        return v0

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
        for key in self.term_dict:
            chol_ix,spin = key
            assert spin in [(0,),(1,),(0,0),(1,1),(0,1)]

            terms = self.term_dict[key]
            nterms = len(terms['a'])
            start = len(self.a)
            stop = start + nterms 

            self.a += terms['a']
            self.ix2key += [(key,i) for i in range(nterms)]
            ixs = list(range(start,stop))
            if len(spin)==1:
                E1_ixs += ixs
            else:
                E2_ixs += ixs
            ixs = xp.asarray(ixs) 
            f = xp.asarray(terms['f'])
            p = xp.asarray(terms['p'])
            d = xp.asarray(terms['d'])
            self.term_dict[key] = {'p':p,'d':d,'ix':ixs,'f':f}

            if iprint>1:
                #print('key=',key)
                #print('ixs=',ixs)
                #print('a=',terms['a'])
                print('f=',f)
                #print('p=',p)
                #print('d=',d)

        self.E1_ixs = xp.asarray(E1_ixs)
        self.E2_ixs = xp.asarray(E2_ixs)
        self.a = xp.asarray(self.a)
        self.asum1 = self.a[self.E1_ixs].sum()
        if self.E2_ixs.size==0:
            self.asum2 = 0.
        else:
            self.asum2 = self.a[self.E2_ixs].sum()
        self.asum = self.asum1+self.asum2
        self.denom = self.asum+self.const
        self.a /= self.denom
        self.nterms = self.a.size
        if iprint>0:
            print(f'asum1={self.asum1},asum2={self.asum2},const={self.const}')
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
        p,d,f = dat['p'][i],dat['d'][i],dat['f'][i]
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
        return p,d,f,u,u2

    def get_term(self,ix):
        key,i = self.ix2key[ix]
        chol_ix,spin = key
        dat = self.term_dict[key]
        p,d,f = dat['p'][i],dat['d'][i],dat['f'][i]
        return chol_ix,spin,p,d,f

    def get_rotation_matrix(self,ix):
        chol_ix,spin,ps,ds,f = self.get_term(ix)
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
        return U,f

class HubbardSOR(SumOfRotationBase):

    def decompose_h2(self,U,dt,iprint=0,trial=None):
        self.hubbard_U = U
        self.run_2body_first = True
        if self.decomp_type=='aa_only':
            raise NotImplementedError

        self.chol_basis.append(np.eye(self.nbasis))
        chol_ix = len(self.chol_basis)-1

        a = 1./np.sqrt(dt)
        e = np.sqrt(U),np.sqrt(U)
        if iprint>0:
            print('Hubbard 2-body decomposition: ')
            print('coefficient =',a)
            print('rotation =',np.sqrt(U*dt))
        rho = np.zeros((self.nbasis,2))
        if trial is not None:
            rho[:,0] = trial.compute_density(0)
            rho[:,1] = trial.compute_density(1)
        v0 = self._compute_v0_hubbard(e[0],rho)
        self.v0[0] = np.diag(v0[0])
        self.v0[1] = np.diag(v0[1])
        for i in range(self.nbasis): 
            self._add_2body_hubbard(chol_ix,(0,1),a,[i,i],e,rho[i])

class QCSOR(SumOfRotationBase):

    def decompose_h2(self,chol,dt,iprint=0,uniform='coefficient',trial=None):
        assert uniform in ['coefficient','rotation']
        if iprint>0:
            print('2-body decomposition: ')

        a = 1./np.sqrt(dt)
        for i,L in enumerate(chol):
            ek,vk = np.linalg.eigh(L) 
            self.chol_basis.append(vk)
            chol_ix = len(self.chol_basis)-1
            rho = np.zeros((self.nbasis,2))
            if trial is not None:
                rho[:,0] = trial.compute_density(0,U=vk)
                rho[:,1] = trial.compute_density(1,U=vk)
                if iprint>1:
                    print('density up=',rho[:,0])
                    print('density down=',rho[:,1])
            if iprint>0:
                print('nchol idx=',chol_ix)
                print('bands=',ek)

            v0 = self._compute_v0_hubbard(ek,rho)
            for p,ep in enumerate(ek):
                if np.fabs(ep)<self.thresh:
                    continue
                if self.decomp_type!='aa_only':
                    self._add_2body_hubbard(chol_ix,(0,1),a,[p,p],(ep,ep),rho[p])
                for q in range(p+1,self.nbasis):
                    eq = ek[q]
                    if np.fabs(eq)<self.thresh:
                        continue
                    epq = ep*eq
                    if self.decomp_type=='aa_only':
                        self._add_2body_hubbard(chol_ix,(0,0),a,[p,q],[ep,eq],[rho[p,0],rho[q,0]])

                        v0[0,p] += epq*rho[q,0]
                        v0[0,q] += epq*rho[p,0]
                        self.const -= epq*rho[p,0]*rho[q,0]
                    elif self.decomp_type=='ab_only':
                        self._add_2body_hubbard(chol_ix,(0,1),a,[p,q],[ep,eq],[rho[p,0],rho[q,1]])
                        self._add_2body_hubbard(chol_ix,(0,1),a,[q,p],[eq,ep],[rho[q,0],rho[p,1]])

                        v0[0,p] += epq*rho[q,1]
                        v0[1,p] += epq*rho[q,0]
                        v0[0,q] += epq*rho[p,1]
                        v0[1,q] += epq*rho[p,0]
                        self.const -= epq*(rho[p,0]*rho[q,1]+rho[p,1]*rho[q,0])
                    else:
                        coeff,delta_p,delta_q = _get_coeffs(a,[ep,eq],uniform)
                        self.add_2body_term(chol_ix,(0,0),coeff,[p,q],[delta_p,delta_q],[rho[p,0],rho[q,0]])
                        self.add_2body_term(chol_ix,(0,1),coeff,[p,q],[-delta_p,-delta_q],[rho[p,0],rho[q,1]])
                        self.add_2body_term(chol_ix,(0,1),coeff,[q,p],[-delta_q,-delta_p],[rho[q,0],rho[p,1]])
                        self.add_2body_term(chol_ix,(1,1),coeff,[p,q],[delta_p,delta_q],[rho[p,1],rho[q,1]])

                        v0[:,p] += epq*rho[q].sum()
                        v0[:,q] += epq*rho[p].sum()
                        self.const -= epq*np.outer(rho[p],rho[q]).sum()
            self.v0 += np.einsum('sp,xp,yp->sxy',v0,vk,vk) 

        self.chol = xp.asarray(chol)
        self.run_2body_first = True
