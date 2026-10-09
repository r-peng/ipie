import numpy as np
import pickle
from ipie.utils.backend import arraylib as xp
from ipie.utils.linalg import modified_cholesky
from ipie.hamiltonians.sor_chol import build_commuting_groups,pack_cholesky
from ipie.hamiltonians.sor_chol_opt import optimize_cholesky_gauge 
#from ipie.hamiltonians.sor_thc import fit_thc_from_eri
#from ipie.hamiltonians.sor_thc_stage1_comm import fit_thc_from_eri
#from ipie.hamiltonians.sor_thc_stage1_regularized import fit_thc_from_eri
from ipie.hamiltonians.sor_thc_stage2_weighted import fit_thc_from_eri

def _get_coeffs(a,g,uniform):
    sqrt_g = np.sqrt(np.fabs(g))
    if g>0.:
        ep, eq = sqrt_g, sqrt_g 
    else:
        ep, eq = sqrt_g, -sqrt_g 

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

def _compute_v0_hubbard(W,rho):
    v0 = np.zeros_like(rho).T
    const = 0.
    for s in (0,1):
        v0[s] = W*rho[:,1-s]
    const -= (W*rho[:,0]*rho[:,1]).sum()
    return v0,const

def compute_total_commutativity(ham):
    Us = [ham.get_rotation_matrix(ix) for ix in range(ham.nterms)]
    comm2 = 0.0
    weighted_comm2 = 0.0
    weight = 0.0
    for i in range(ham.nterms):
        for j in range(i + 1, ham.nterms):
            pair_comm2 = 0.0
            for Uis, Ujs in zip(Us[i], Us[j]):
                # If None means identity/no action in this spin block,
                # the commutator contribution from this block is zero.
                if Uis is None or Ujs is None:
                    continue
                Cij = np.dot(Uis,Ujs)-np.dot(Ujs,Uis)
                pair_comm2 += np.linalg.norm(Cij, 'fro')**2
            wij = ham.a[i] * ham.a[j]
            comm2 += pair_comm2
            weighted_comm2 += wij * pair_comm2
            weight += wij
    print('comm2=',comm2)
    print('weighted_comm2=',weighted_comm2)
    print('weight=',weight)
    print('weighted_comm2/weight=',weighted_comm2/weight)

class SizeGroup:
    def __init__(self):
        self.basis = []
        self.integrals = []
        self.integral_tags = []
        self.overlaps = []
        self.term_info = dict()

        self._integral_tags = 'h1','hubbard','thc'

    def add_basis(self,X,W,tag,isometry=True):
        if len(self.basis)>0:
            assert X.shape==self.basis[-1].shape
        self.basis.append(X)
        self.integrals.append(W)
        assert tag in self._integral_tags
        self.integral_tags.append(tag)
        S = None if isometry else np.dot(X.T,X)
        self.overlaps.append(S)

    def get_basis_index(self):
        ix = len(self.basis)-1
        assert len(self.integrals)==ix+1
        assert len(self.overlaps)==ix+1
        return ix

    def add_term(self,bix,s,a,p,d):
        key = bix,tuple(s)
        if key in self.term_info:
            term_info = self.term_info[key]
        else:
            term_info = {'a':[],'p':[],'d':[]}
        term_info['a'].append(a)
        term_info['p'].append(p)
        term_info['d'].append(d)
        self.term_info[key] = term_info 

    def decompose_1body(self,bix,s,dt,uniform='coefficient',trial=None,thresh=1e-6):
        nb1,nb2 = self.basis[bix].shape
        rho = np.zeros(nb2)
        if trial is not None:
            X = self.basis[bix]
            rho = trial.compute_density(s,X=X)
        const = 0.
        for p,ep in enumerate(self.integrals[bix]):
            if np.fabs(ep)<thresh:
                continue

            if uniform=='coefficient':
                ap = 1./dt
            else:
                ap = np.fabs(ep/dt)
            delta_p = -ep/ap
            dp = _delta2d(delta_p,rho=rho[p])
            assert np.fabs(dp)<1.
            fp = 1.+dp*rho[p]
            const += ep*rho[p]
            self.add_term(bix,[s],ap/fp,[p],[dp])
        return const

    def add_2body_term(self,bix,s,a,p,delta,rho):
        d = [_delta2d(delta[i],rho=rho[i]) for i in (0,1)]
        assert np.fabs(d[0])<1.
        assert np.fabs(d[1])<1.
        f = [1.+d[i]*rho[i] for i in (0,1)]
        self.add_term(bix,s,a/(f[0]*f[1]),p,d)

    def add_2body_hubbard(self,bix,s,a,g,pq,rho,uniform='coefficient'):
        coeff,delta_p,delta_q = _get_coeffs(a,g,uniform)
        coeff /= 2.
        self.add_2body_term(bix,s,coeff,pq,[delta_p,delta_q],rho)
        self.add_2body_term(bix,s,coeff,pq,[-delta_p,-delta_q],rho)

    def basis2cupy(self):
        self.num_basis = len(self.basis)
        assert len(self.integrals)==self.num_basis
        assert len(self.overlaps)==self.num_basis

        self.basis = xp.asarray(self.basis)
        self.integrals = [xp.asarray(X) for X in self.integrals]
        self.overlaps = [None if X is None else xp.asarray(X) for X in self.overlaps] 

    def decompose_hubbard(self,bix,dt,rho,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        Wdiag = W if len(W.shape)==1 else np.diag(W)
        v0,const = _compute_v0_hubbard(Wdiag,rho)
        a = 1./np.sqrt(dt)
        for i,Wi in enumerate(Wdiag):
            if np.fabs(Wi)<thresh:
                continue
            self.add_2body_hubbard(bix,(0,1),a,Wi,[i,i],rho[i])
        return v0,const

    def decompose_ab_only(self,bix,dt,rho,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        a = 1./np.sqrt(dt)
        v0 = np.zeros_like(rho).T
        const = 0.
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            self.add_2body_hubbard(bix,(0,1),a,Wpq,[p,q],[rho[p,0],rho[q,1]])
            self.add_2body_hubbard(bix,(0,1),a,Wpq,[q,p],[rho[q,0],rho[p,1]])

            v0[0,p] += Wpq*rho[q,1]
            v0[1,p] += Wpq*rho[q,0]
            v0[0,q] += Wpq*rho[p,1]
            v0[1,q] += Wpq*rho[p,0]
            const -= Wpq*(rho[p,0]*rho[q,1]+rho[p,1]*rho[q,0])
        return v0,const

    def decompose_aa_only(self,bix,dt,rho,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        S = self.overlaps[bix]
        a = 1./np.sqrt(dt)
        v0 = np.zeros_like(rho).T
        const = 0.
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            if S is None:
                self.add_2body_hubbard(bix,(0,0),a,Wpq,[p,q],[rho[p,0],rho[q,0]])
            else:
                self.add_2body_hubbard(bix,(0,0),.5*a,Wpq,[p,q],[rho[p,0],rho[q,0]])
                self.add_2body_hubbard(bix,(0,0),.5*a,Wpq,[q,p],[rho[q,0],rho[p,0]])

            v0[0,p] += Wpq*rho[q,0]
            v0[0,q] += Wpq*rho[p,0]
            const -= Wpq*rho[p,0]*rho[q,0]
        return v0,const

    def decompose_full_2body(self,bix,dt,rho,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        S = self.overlaps[bix]
        a = 1./np.sqrt(dt)
        v0 = np.zeros_like(rho).T
        const = 0.
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            coeff,delta_p,delta_q = _get_coeffs(a,Wpq,uniform)
            self.add_2body_term(bix,(0,1),coeff,[p,q],[-delta_p,-delta_q],[rho[p,0],rho[q,1]])
            self.add_2body_term(bix,(0,1),coeff,[q,p],[-delta_q,-delta_p],[rho[q,0],rho[p,1]])
            if S is None:
                self.add_2body_term(bix,(0,0),coeff,[p,q],[delta_p,delta_q],[rho[p,0],rho[q,0]])
                self.add_2body_term(bix,(1,1),coeff,[p,q],[delta_p,delta_q],[rho[p,1],rho[q,1]])
            else:
                self.add_2body_term(bix,(0,0),.5*coeff,[p,q],[delta_p,delta_q],[rho[p,0],rho[q,0]])
                self.add_2body_term(bix,(0,0),.5*coeff,[q,p],[delta_q,delta_p],[rho[q,0],rho[p,0]])
                self.add_2body_term(bix,(1,1),.5*coeff,[p,q],[delta_p,delta_q],[rho[p,1],rho[q,1]])
                self.add_2body_term(bix,(1,1),.5*coeff,[q,p],[delta_q,delta_p],[rho[q,1],rho[p,1]])

            v0[:,p] += Wpq*rho[q].sum()
            v0[:,q] += Wpq*rho[p].sum()
            const -= Wpq*np.outer(rho[p],rho[q]).sum()
        return v0,const

class SumOfRotationBase:

    def __init__(self,nbasis,thresh=1e-6,decomp_type='all'):
        self.nbasis = nbasis
        self.thresh = thresh
        assert decomp_type in ['all','aa_only','ab_only']
        self.decomp_type = decomp_type

        self.size_groups = dict() 

        self.run_2body_first = False
        self.v0 = np.zeros((2,self.nbasis,self.nbasis)) 
        self.const = 0.

    def decompose_h1(self,h1,dt,uniform='coefficient',iprint=0,trial=None):
        if iprint>0:
            print('1-body decomposition: ')
            print('a=',1./dt)
        if not self.run_2body_first:
            raise ValueError('Run 2-body decomposition first!')
        assert uniform in ['coefficient','rotation']
        self.h1 = xp.asarray(h1)

        if self.nbasis not in self.size_groups:
            self.size_groups[self.nbasis] = SizeGroup()
        sg = self.size_groups[self.nbasis]

        if np.linalg.norm(self.v0[0]-self.v0[1])<self.thresh:
            ek,vk = np.linalg.eigh(h1+self.v0[0]) 
            if iprint>0:
                print(f'bands for both spin:',ek)
            sg.add_basis(vk,ek,'h1')
            bix = sg.get_basis_index()
            for s in (0,1):
                self.const += sg.decompose_1body(bix,s,dt,uniform=uniform,trial=trial,thresh=self.thresh)
            self.size_groups[self.nbasis] = sg
            return

        for s,v0 in enumerate(self.v0):
            if s==1 and self.decomp_type=='aa_only':
                continue

            ek,vk = np.linalg.eigh(h1+v0) 
            if iprint>0:
                print(f'spin={s} bands:',ek)
            sg.add_basis(vk,ek,'h1')
            bix = sg.get_basis_index()
            self.const += sg.decompose_1body(bix,s,dt,uniform=uniform,trial=trial,thresh=self.thresh)
        self.size_groups[self.nbasis] = sg

    def parse_decomposition(self,iprint=0):
        self.size_keys = list(self.size_groups.keys())

        self.a = []
        self.ix2key = []
        E1_ixs = []
        E2_ixs = []

        self.term_dict = dict() 
        for size_key in self.size_keys:
            sg = self.size_groups[size_key]
            sg.basis2cupy()
            
            term_keys = list(sg.term_info.keys())
            for term_key in term_keys:
                bix,spin = term_key
                assert spin in [(0,),(1,),(0,0),(1,1),(0,1)]

                terms = sg.term_info[term_key]
                nterms = len(terms['a'])
                start = len(self.a)
                stop = start + nterms 

                a = terms['a']
                self.a += a
                self.ix2key += [(size_key,term_key,i) for i in range(nterms)]
                ixs = list(range(start,stop))
                if len(spin)==1:
                    E1_ixs += ixs
                else:
                    E2_ixs += ixs
                ixs = xp.asarray(ixs) 
                p = xp.asarray(terms['p'])
                d = xp.asarray(terms['d'])
                sg.term_info[term_key] = {'p':p,'d':d,'ix':ixs}
                # at this point not sure what saved ixs is for 
                # might not need if we never compute all overlap ratios

                if iprint>1:
                    print('key=',term_key)
                    print('ixs=',ixs)
                    print('a=',a)
                    print('p=',p)
                    print('d=',d)
            self.size_groups[size_key] = sg

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
            size_key,term_key,i = self.ix2key[ix]
            key = size_key,term_key
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

    def get_batch_info(self,key,i):
        size_key,term_key = key
        sg = self.size_groups[size_key]
        data = sg.term_info[term_key]
        p,d = data['p'][i],data['d'][i]

        bix,spin = term_key
        v = sg.basis[bix][:,p]
        S = sg.overlaps[bix]
        if spin in [(0,0),(1,1)]:
            d_ = xp.zeros((d.shape[0],2,2),dtype=d.dtype)
            d_[:,0,0] = d[:,0]
            d_[:,1,1] = d[:,1]
            if S is not None:
                d_[:,0,1] = d[:,0]*d[:,1]*S[p[:,0],p[:,1]]
        else:
            d_ = d
        return p,d_,v.transpose(1,0,2)

    def get_term(self,ix):
        size_key,term_key,i = self.ix2key[ix]
        sg = self.size_groups[size_key]
        dat = sg.term_info[term_key]
        p,d = dat['p'][i],dat['d'][i]
        return size_key,term_key,p,d

    def get_rotation_matrix(self,ix):
        size_key,term_key,ps,ds = self.get_term(ix)
        bix,spin = term_key
        sg = self.size_groups[size_key]
        v = sg.basis[bix]
        U = [None] * 2

        def rank1_matrix(p,d):
            vp = v[:,p]
            return xp.eye(v.shape[0]) + d*xp.outer(vp,vp)

        if spin==(0,1):
            for s,p in enumerate(ps):
                U[s] = rank1_matrix(p,ds[s])
        elif len(spin)==1:
            s = spin[0]
            U[s] = rank1_matrix(ps[0],ds[0])
        else:
            s = spin[0]
            U[s] = xp.dot(rank1_matrix(ps[0],ds[0]),rank1_matrix(ps[1],ds[1]))
        return U

class HubbardSOR(SumOfRotationBase):

    def decompose_h2(self,U,dt,iprint=0,trial=None):
        self.run_2body_first = True
        if self.decomp_type=='aa_only':
            raise NotImplementedError

        if self.nbasis not in self.size_groups:
            self.size_groups[self.nbasis] = SizeGroup()
        sg = self.size_groups[self.nbasis]
        sg.add_basis(np.eye(self.nbasis),np.ones(self.nbasis)*U,'hubbard')
        bix = sg.get_basis_index()

        if iprint>0:
            a = 1./np.sqrt(dt)
            print('Hubbard 2-body decomposition: ')
            print('coefficient =',a)
            print('rotation =',np.sqrt(U*dt))
        rho = np.zeros((self.nbasis,2))
        if trial is not None:
            rho[:,0] = trial.compute_density(0)
            rho[:,1] = trial.compute_density(1)
            if iprint>1:
                print('density up=',rho[:,0])
                print('density down=',rho[:,1])
        v0,const = sg.decompose_hubbard(bix,dt,rho)
        self.const += const
        self.v0[0] = np.diag(v0[0])
        self.v0[1] = np.diag(v0[1])
        self.size_groups[self.nbasis] = sg

class QCSOR(SumOfRotationBase):

    def from_cholesky(self,chol=None,eri=None,cmax=None,comm_tol=1e-10,basis_tol=1e-10,fname=None):
        # set comm_tol to negative to not do regrouping
        if chol is None:
            M = eri.reshape((self.nbasis**2, self.nbasis**2))
            if cmax is None:
                cmax = self.nbasis**2
            chol = modified_cholesky(M, cmax=cmax).reshape(-1, self.nbasis, self.nbasis)

        fit = optimize_cholesky_gauge(chol)
        chol = fit.chol

        if comm_tol<0.:
            result = pack_cholesky(chol)
            C = None
        else:
            print('buiding commuting groups...')
            result,C = build_commuting_groups(chol,comm_tol=comm_tol,basis_tol=basis_tol)
        if fname is not None:
            with open(fname+".pkl", "wb") as f:
                pickle.dump({"grouped": result,"commutator_matrix": C, 'chol':chol},f,protocol=pickle.HIGHEST_PROTOCOL)
        return result

    def from_thc(self,eri,rank,comm_lambda=1.,nstarts=4,maxiter=1000,rcond=1e-6,fname=None): 
        fit = fit_thc_from_eri(eri,rank,comm_lambda=comm_lambda,n_starts=nstarts,maxiter=maxiter,rcond=rcond)
        result = [{'X':fit.X,'W':fit.W,'isometry':False}]
        if fname is not None:
            with open(fname+".pkl", "wb") as f:
                pickle.dump({"grouped": result,'eri':eri},f,protocol=pickle.HIGHEST_PROTOCOL)
        return result,fit

    def decompose_h2(self,fname,dt,iprint=0,uniform='coefficient',trial=None):
        assert uniform in ['coefficient','rotation']
        if iprint>0:
            print('2-body decomposition: ')
        if isinstance(fname,str):
            with open(fname+".pkl", "rb") as f:
                data = pickle.load(f)
            grouped = data["grouped"]
        elif isinstance(fname,dict):
            grouped = fname["grouped"] if "grouped" in fname else [fname]
        else:
            grouped = fname
            if isinstance(fname,np.ndarray):
                grouped = pack_cholesky(fname)

        a = 1./np.sqrt(dt)
        for i,result in enumerate(grouped):
            X = result['X']
            W = result['W']
            isometry = result.get('isometry',False)
            nbasis = X.shape[1]
            if nbasis not in self.size_groups:
                self.size_groups[nbasis] = SizeGroup()
            sg = self.size_groups[nbasis]
            sg.add_basis(X,W,'thc',isometry=isometry)
            bix = sg.get_basis_index()
            if iprint>0:
                print('basis index=',bix)

            S = np.eye(nbasis) if isometry else sg.overlaps[-1]
            if self.decomp_type=='ab_only':
                self.v0 += .5*(np.dot(X,np.dot(W*S,X.T)))[None,:,:]
                v0 = np.zeros((2,nbasis))
            else:
                v0 = .5*np.diag(W)
                if not isometry:
                    v0 *= np.diag(sg.overlaps[-1])
                v0 = np.tile(v0[None,:],(2,1))

            rho = np.zeros((nbasis,2))
            if trial is not None:
                rho[:,0] = trial.compute_density(0,X=X)
                rho[:,1] = trial.compute_density(1,X=X)
                if iprint>1:
                    print('density up=',rho[:,0])
                    print('density down=',rho[:,1])

            if self.decomp_type!='aa_only':
                v0_,const = sg.decompose_hubbard(bix,dt,rho,uniform=uniform,thresh=self.thresh)
                v0 += v0_
                self.const += const

            if self.decomp_type=='aa_only':
                v0_,const = sg.decompose_aa_only(bix,dt,rho,uniform=uniform,thresh=self.thresh)
            elif self.decomp_type=='ab_only':
                v0_,const = sg.decompose_ab_only(bix,dt,rho,uniform=uniform,thresh=self.thresh)
            else:
                v0_,const = sg.decompose_full_2body(bix,dt,rho,uniform=uniform,thresh=self.thresh)
            v0 += v0_
            self.const += const

            self.v0 += np.einsum('sp,xp,yp->sxy',v0,X,X) 
        self.run_2body_first = True
