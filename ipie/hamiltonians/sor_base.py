import numpy as np
import pickle
import os
import json
import math
from ipie.utils.backend import arraylib as xp
# ipie.utils.linalg, h5py and the optimize_cholesky package are imported inside the
# functions that need them. Optimized decompositions are written by
# ipie.hamiltonians.optimize_cholesky.io.save_decomposition and read by QCSOR.from_file.

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
    print('weighted_comm2=',weighted_comm2*ham.asum**2)
    #print('weight=',weight)
    #print('weighted_comm2/weight=',weighted_comm2/weight)

def pack_cholesky(chol):
    assert chol.ndim==3
    grouped = []
    for L in chol:
        eps,X = np.linalg.eigh(L)
        grouped.append({"X":X,"W":np.outer(eps,eps),"isometry":True})
    return grouped

def _parse_mu(mu,ek,s):
    if mu is None:
        return 0.
    if mu=='mean':
        return ek.sum()/ek.size
    try:
        return mu[s]
    except TypeError:
        return mu

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

    def decompose_1body(self,bix,s,dt,uniform='coefficient',thresh=1e-6):
        nb1,nb2 = self.basis[bix].shape
        for p,ep in enumerate(self.integrals[bix]):
            if np.fabs(ep)<thresh:
                continue

            if uniform=='coefficient':
                ap = 1./dt
            else:
                ap = np.fabs(ep/dt)
            dp = -ep/ap
            assert np.fabs(dp)<1.
            self.add_term(bix,[s],ap,[p],[dp])

    def add_2body_term(self,bix,s,a,p,d):
        assert np.fabs(d[0])<1.
        assert np.fabs(d[1])<1.
        self.add_term(bix,s,a,p,d)

    def add_2body_hubbard(self,bix,s,a,g,pq,uniform='coefficient'):
        coeff,dp,dq = _get_coeffs(a,g,uniform)
        coeff /= 2.
        self.add_2body_term(bix,s,coeff,pq,[dp,dq])
        self.add_2body_term(bix,s,coeff,pq,[-dp,-dq])

    def basis2cupy(self):
        self.num_basis = len(self.basis)
        assert len(self.integrals)==self.num_basis
        assert len(self.overlaps)==self.num_basis

        self.basis = xp.asarray(self.basis)
        self.integrals = [xp.asarray(X) for X in self.integrals]
        self.overlaps = [None if X is None else xp.asarray(X) for X in self.overlaps] 

    def decompose_hubbard(self,bix,dt,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        Wdiag = W if len(W.shape)==1 else np.diag(W)
        a = 1./np.sqrt(dt)
        for i,Wi in enumerate(Wdiag):
            if np.fabs(Wi)<thresh:
                continue
            self.add_2body_hubbard(bix,(0,1),a,Wi,[i,i],uniform=uniform)

    def decompose_ab_only(self,bix,dt,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        a = 1./np.sqrt(dt)
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            self.add_2body_hubbard(bix,(0,1),a,Wpq,[p,q],uniform=uniform)
            self.add_2body_hubbard(bix,(0,1),a,Wpq,[q,p],uniform=uniform)

    def decompose_aa_only(self,bix,dt,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        S = self.overlaps[bix]
        a = 1./np.sqrt(dt)
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            if S is None:
                self.add_2body_hubbard(bix,(0,0),a,Wpq,[p,q],uniform=uniform)
            else:
                self.add_2body_hubbard(bix,(0,0),a,.5*Wpq,[p,q],uniform=uniform)
                self.add_2body_hubbard(bix,(0,0),a,.5*Wpq,[q,p],uniform=uniform)

    def decompose_full_2body(self,bix,dt,uniform='coefficient',thresh=1e-6):
        W = self.integrals[bix]
        S = self.overlaps[bix]
        a = 1./np.sqrt(dt)
        nbasis = W.shape[0]
        pairs = [(p,q) for p in range(nbasis) for q in range(p+1,nbasis)]
        for (p,q) in pairs:
            Wpq = W[p,q]
            if np.fabs(Wpq)<thresh:
                continue
            coeff,dp,dq = _get_coeffs(a,Wpq,uniform)
            self.add_2body_term(bix,(0,1),coeff,[p,q],[-dp,-dq])
            self.add_2body_term(bix,(0,1),coeff,[q,p],[-dq,-dp])
            if S is None:
                self.add_2body_term(bix,(0,0),coeff,[p,q],[dp,dq])
                self.add_2body_term(bix,(1,1),coeff,[p,q],[dp,dq])
            else:
                self.add_2body_term(bix,(0,0),.5*coeff,[p,q],[dp,dq])
                self.add_2body_term(bix,(0,0),.5*coeff,[q,p],[dq,dp])
                self.add_2body_term(bix,(1,1),.5*coeff,[p,q],[dp,dq])
                self.add_2body_term(bix,(1,1),.5*coeff,[q,p],[dq,dp])

class SumOfRotationBase:

    def __init__(self,nbasis,thresh=1e-6,decomp_type='all'):
        self.nbasis = nbasis
        self.thresh = thresh
        assert decomp_type in ['all','aa_only','ab_only']
        self.decomp_type = decomp_type
        self.v0 = np.zeros((2,self.nbasis,self.nbasis))
        self.mu = np.zeros(2)
        self.E0 = 0.

        self.size_groups = dict() 
        # for cholesky & double factorization:
        # should have only one size group with key=self.nbasis
        # for thc: should have two size groups
        # one with key=self.nbasis
        # one with key=number of auxiliary modes

    def decompose_h1(self,h1,dt,uniform='coefficient',iprint=0,mu='mean'):
        # for cholesky vectors, pass the effective h'
        if iprint>0:
            print('1-body decomposition: ')
            print('a=',1./dt)
        assert uniform in ['coefficient','rotation']
        self.h1 = xp.asarray(h1)

        if self.nbasis not in self.size_groups:
            self.size_groups[self.nbasis] = SizeGroup()
        sg = self.size_groups[self.nbasis]

        if np.linalg.norm(self.v0[0]-self.v0[1])<self.thresh:
            ek,vk = np.linalg.eigh(h1+self.v0[0]) 
            if iprint>0:
                print(f'bands for both spin:',ek)
            mu = _parse_mu(mu,ek,0)
            ek -= mu
            self.mu[0] = self.mu[1] = mu
            if iprint>0:
                print(f'mean of bands =',mu)
                print(f'shifted bands:',ek)

            sg.add_basis(vk,ek,'h1')
            bix = sg.get_basis_index()
            for s in (0,1):
                sg.decompose_1body(bix,s,dt,uniform=uniform,thresh=self.thresh)
            self.size_groups[self.nbasis] = sg
            return

        for s,v0 in enumerate(self.v0):
            if s==1 and self.decomp_type=='aa_only':
                continue

            ek,vk = np.linalg.eigh(h1+v0) 
            if iprint>0:
                print(f'spin={s} bands:',ek)
            mu = _parse_mu(mu,ek,s)
            ek -= mu
            self.mu[s] = mu
            if iprint>0:
                print(f'mean of bands =',mu)
                print(f'shifted bands:',ek)
            sg.add_basis(vk,ek,'h1')
            bix = sg.get_basis_index()
            sg.decompose_1body(bix,s,dt,uniform=uniform,thresh=self.thresh)
        self.size_groups[self.nbasis] = sg

    def parse_decomposition(self,nelec,iprint=0):
        """Concatenate all terms into one global index; normalize a by denom.

        sg.term_info[key] may hold Python lists (decompose_* paths, load_optimized)
        or numpy arrays (load_runtime). Both are handled without converting arrays
        to per-term Python objects: a, E1_ixs and E2_ixs are built from array
        chunks. The global term order is size_keys, then term_info insertion order.
        """
        self.size_keys = list(self.size_groups.keys())

        a_chunks = []
        E1_chunks = []
        E2_chunks = []
        self.ix2key = []
        nterms_total = 0

        self.term_dict = dict() 
        for size_key in self.size_keys:
            sg = self.size_groups[size_key]
            sg.basis2cupy()
            
            term_keys = list(sg.term_info.keys())
            for term_key in term_keys:
                bix,spin = term_key
                assert spin in [(0,),(1,),(0,0),(1,1),(0,1)]

                terms = sg.term_info[term_key]
                a = np.asarray(terms['a'],dtype=np.float64).reshape(-1)
                nterms = a.size
                start = nterms_total
                stop = start + nterms 
                nterms_total = stop

                a_chunks.append(a)
                self.ix2key += [(size_key,term_key,i) for i in range(nterms)]
                ixs = np.arange(start,stop,dtype=np.int64)
                if len(spin)==1:
                    E1_chunks.append(ixs)
                else:
                    E2_chunks.append(ixs)
                p = xp.asarray(terms['p'])
                d = xp.asarray(terms['d'])
                assert len(d[d<-1.+self.thresh])==0

                sg.term_info[term_key] = {'p':p,'d':d,'ix':xp.asarray(ixs)}
                # at this point not sure what saved ixs is for 
                # might not need if we never compute all overlap ratios

                if iprint>1:
                    print('\nkey=',term_key)
                    print('ixs=',ixs)
                    print('a=',a)
                    print('p=',p)
                    print('d=',d)
            self.size_groups[size_key] = sg

        def concatenate(chunks,dtype):
            return np.concatenate(chunks) if chunks else np.zeros(0,dtype=dtype)

        self.E1_ixs = xp.asarray(concatenate(E1_chunks,np.int64))
        self.E2_ixs = xp.asarray(concatenate(E2_chunks,np.int64))
        self.a = xp.asarray(concatenate(a_chunks,np.float64))
        del a_chunks
        self.asum1 = self.a[self.E1_ixs].sum()
        if self.E2_ixs.size==0:
            self.asum2 = 0.
        else:
            self.asum2 = self.a[self.E2_ixs].sum()
        self.asum = self.asum1+self.asum2
        self.denom = self.asum
        self.a /= self.denom
        self.nterms = self.a.size
        self.E0 = self.mu[0]*nelec[0]+self.mu[1]*nelec[1]
        if iprint>0:
            print(f'asum1={self.asum1},asum2={self.asum2}')
            print('normalization=',xp.fabs(self.a).sum())
            print('number of terms=',self.nterms)
            print('E0=',self.E0)

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

    def decompose_h2(self,U,dt,iprint=0):
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
        sg.decompose_hubbard(bix,dt)
        self.size_groups[self.nbasis] = sg

def _check_sector(policy,nelec):
    """hybrid is exact on every (N_up,N_down) sector; ab_only only on (1,1);
    aa_only only on sectors with N_down=0."""
    if policy=='ab_only' and tuple(nelec)!=(1,1):
        raise ValueError(f'ab_only decompositions are exact only for nelec=(1,1), not {tuple(nelec)}')
    if policy=='aa_only' and nelec[1]!=0:
        raise ValueError(f'aa_only decompositions are exact only for N_down=0, not {tuple(nelec)}')


def _read_decomposition_file(path):
    """Read and check a file written by optimize_cholesky.io.save_decomposition."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading optimized decompositions requires h5py; install it with "
                          "'pip install h5py'") from exc
    from ipie.hamiltonians.optimize_cholesky.io import FORMAT,FORMAT_VERSION,POLICY_TO_DECOMP_TYPE
    names = ('h1','v0','chol_X','chol_ell','h1_X','h1_eps','a','p1','d1','p2','d2','keys')
    with h5py.File(path,'r') as f:
        fmt,version = f.attrs.get('format'),int(f.attrs.get('format_version',-1))
        if fmt!=FORMAT or version!=FORMAT_VERSION:
            raise ValueError(f'{path} is not a {FORMAT} file of format_version {FORMAT_VERSION} '
                             f'(found format={fmt!r}, version={version}); save it with '
                             'optimize_cholesky.io.save_decomposition')
        A = f.attrs
        data = {name:f[name][()] for name in names}
        p = f['provenance']
        data['policy'] = str(A['policy'])
        data['decomp_type'] = str(A['decomp_type'])
        data['nelec'] = tuple(int(x) for x in A['nelec']) if bool(A['has_nelec']) else None
        data['nbasis'],data['nchol'] = int(A['nbasis']),int(A['nchol'])
        data['mu'] = np.array(A['mu'],dtype=np.float64)
        data['denom'] = float(A['denom'])
        data['lambda_electronic'] = float(A['lambda_electronic'])
        data['dropped_identity'] = int(A['dropped_identity'])
        data['metadata'] = {
            'path':os.fspath(path),'format':FORMAT,'format_version':version,'created':str(A['created']),
            'policy':data['policy'],'nelec':data['nelec'],'mu':tuple(float(x) for x in data['mu']),
            'tau1':float(A['tau1']),'tau2':float(A['tau2']),'partition':str(A['partition']),
            'amplitude_bound':float(A['amplitude_bound']),'denom':data['denom'],
            'denom_all':float(A['denom_all']),'lambda_electronic':data['lambda_electronic'],
            'drop_identity':bool(A['drop_identity']),'dropped_identity':data['dropped_identity'],
            'provenance':json.loads(str(p.attrs['record'])),'user':json.loads(str(p.attrs['user_metadata'])),
        }

    n,r = data['nbasis'],data['nchol']
    keys,T,T1 = data['keys'],data['a'].shape[0],data['p1'].shape[0]
    k = data['h1_eps'].shape[0] if data['h1_eps'].ndim==2 else -1
    def require(condition,message):
        if not condition:
            raise ValueError(f'{path}: {message}')
    require(data['policy'] in POLICY_TO_DECOMP_TYPE and data['decomp_type']==POLICY_TO_DECOMP_TYPE[data['policy']],
            'inconsistent policy/decomp_type')
    require(data['chol_X'].shape==(r,n,n) and data['chol_ell'].shape==(r,n),'invalid factor arrays')
    require(data['h1_X'].shape==(n,n) and k>=0 and data['h1_eps'].shape==(k,n),'invalid h1 basis arrays')
    require(data['h1'].shape==(n,n) and data['v0'].shape==(2,n,n),'invalid h1/v0 arrays')
    require(data['a'].ndim==1 and data['d1'].shape==(T1,) and data['p2'].shape==data['d2'].shape==(T-T1,2),
            'invalid term arrays')
    require(keys.ndim==2 and keys.shape[1]==5,'invalid key table')
    if keys.shape[0]==0:
        require(T==0,'terms without keys')
    else:
        require(keys[0,3]==0 and keys[-1,4]==T and np.all(keys[1:,3]==keys[:-1,4])
                and np.all(keys[:,4]>=keys[:,3]),'key ranges must be contiguous and ordered')
        one = keys[:,2]<0
        n_one = int(np.count_nonzero(one))
        require(np.all(one[:n_one]) and (keys[n_one-1,4]==T1 if n_one else T1==0),
                'one-body keys must come first and cover [0,T1)')
        require(np.all((keys[one,0]>=r)&(keys[one,0]<r+k)) and np.all((keys[~one,0]>=0)&(keys[~one,0]<r)),
                'key basis index out of range')
        require(np.all(np.isin(keys[one,1],(0,1))) and
                all(tuple(sp) in ((0,0),(1,1),(0,1)) for sp in keys[~one,1:3].tolist()),'invalid key spins')
        require(len({tuple(row[:3]) for row in keys.tolist()})==keys.shape[0],'duplicate keys')
    require(np.all((data['p1']>=0)&(data['p1']<n)) and np.all((data['p2']>=0)&(data['p2']<n)),
            'term modes out of range')
    require(np.all(np.isfinite(data['a'])&(data['a']>0)),'weights must be finite and positive')
    return data


class QCSOR(SumOfRotationBase):

    @classmethod
    def initialize_cholesky(cls,eri,cmax=None):
        from ipie.utils.linalg import modified_cholesky
        nbasis = eri.shape[0]
        M = eri.reshape((nbasis**2, nbasis**2))
        if cmax is None:
            cmax = nbasis**2
        return modified_cholesky(M, cmax=cmax).reshape(-1, nbasis, nbasis)

    @classmethod
    def optimize_from_cholesky(cls,h1e,eri,chol,nelecs,dt1,dt2,source,**kwargs):
        """Optimize the Cholesky gauge and save the result to the HDF5 file `source`.

        The file is written by optimize_cholesky.io.save_decomposition and is read
        back with QCSOR.from_file(source). Extra keyword arguments drop_identity,
        overwrite and metadata are passed to save_decomposition.
        """
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        from ipie.hamiltonians.optimize_cholesky.models import LBFGSOptions,InnerOptions
        from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem
        from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
        from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
        problem = prepare_problem(h1e,eri,chol,nelec=nelecs)
    
        r = problem.data.nchol
        theta = kwargs.get('theta',0.1)
        theta = theta*(np.random.rand(r * (r - 1) // 2)*2.-1.)
    
        partition = kwargs.get('partition','balanced_log')
        loss_kind = kwargs.get('loss_kind','weighted')
        evaluator = GaugeEvaluator(problem, tau1=dt1, tau2=dt2, loss_kind=loss_kind, partition=partition, backend='numpy')
    
        maxiter = kwargs.get('maxiter',100)
        maxsweep = kwargs.get('maxsweep',100)
        verbosity = kwargs.get('verbosity',2)
        result = optimize_gauge(evaluator,
            solver='lbfgs',
            theta0=theta,
            options=LBFGSOptions(max_iterations=maxiter,verbosity=verbosity, line_search='weak_wolfe'),
            inner_options=InnerOptions(max_sweeps=maxsweep, gradient_atol=1e-6, gradient_rtol=1e-6,optimize_shift=True),
    )
        save_decomposition(source,result,drop_identity=kwargs.get('drop_identity',False),
                           overwrite=kwargs.get('overwrite',False),metadata=kwargs.get('metadata',None))
        return result

    @classmethod
    def from_file(cls,path,*,nelec=None,parse=True,thresh=1e-6,iprint=0):
        """Build a QCSOR from a file written by optimize_cholesky.io.save_decomposition.

        Array-only: term_info holds numpy slices of the file arrays and no
        per-channel Python objects are built. One size group of key nbasis:
          * bix=gamma (0..r-1): factor eigenvectors (isometry, S=None), integrals
            W=outer(ell,ell), tag 'thc' (used by local_energy);
          * bix=r (and r+1 if mu_up!=mu_down): the h eigenbasis, integrals
            eps-mu_s, tag 'h1';
          * term_info keys in file order, so parse_decomposition gives the file's
            global term order (minus any truncated terms).

        Optional truncation (default None: no truncation, exact):
          * thresh1 drops one-body channel (k,s) if |eps_k - mu_s| < thresh1, i.e.
            the SHIFTED orbital energy, the term (eps_k-mu_s) n_ks it represents
            (equivalently |d| < thresh1*tau1);
          * thresh2 drops two-body channels with |W_ab| = |ell_a ell_b| < thresh2
            (equivalently |d_L d_R| < thresh2*tau2). All branches of a source share
            W, so a source is always dropped as a whole.
        Truncation is not exact: the walk then uses H minus the dropped terms, with
        Lambda = (kept sum a) + mu.N = ham.denom + ham.E0 after parsing. ham.h1, v0
        and the W integrals are NOT truncated, so local_energy still measures the
        full H. Bases are kept even if all of their terms are dropped, so bix stays
        valid. ham.truncation records the thresholds, counts, dropped weight and
        largest dropped |eps-mu| and |W|; optimized_metadata['lambda_electronic']
        is the file's untruncated value.

        Sets h1 (h' = h - 1/2 sum L^2), v0, mu, optimized_nelec, dropped_identity,
        truncation and optimized_metadata. With parse=True (default)
        parse_decomposition(nelec) is called, with nelec defaulting to the particle
        numbers of the optimization; a sector on which the stored policy is not
        exact raises ValueError (hybrid: any; ab_only: (1,1); aa_only: N_down=0),
        and the parsed denom is checked against the file minus the dropped weight.
        """
        data = _read_decomposition_file(path)
        n,r = data['nbasis'],data['nchol']
        ham = cls(n,thresh=thresh,decomp_type=data['decomp_type'])

        sg = SizeGroup()
        for g in range(r):
            ell = data['chol_ell'][g]
            sg.add_basis(data['chol_X'][g],np.outer(ell,ell),'thc',isometry=True)
        for k,eps in enumerate(data['h1_eps']):
            sg.add_basis(data['h1_X'] if k==0 else data['h1_X'].copy(),eps,'h1')

        a,p1,d1,p2,d2,keys = data['a'],data['p1'],data['d1'],data['p2'],data['d2'],data['keys']
        T1,T = p1.shape[0],a.shape[0]
        truncation = {'thresh':thresh,'dropped_one_body':0,'dropped_two_body':0,
                      'dropped_weight':0.,'max_dropped_eps':0.,'max_dropped_W':0.}
        if thresh is not None:
            # Per-term magnitudes. One-body keys come first and cover [0,T1).
            one = keys[:,2]<0
            counts = keys[:,4]-keys[:,3]
            keep = np.ones(T,dtype=bool)
            if thresh is not None and T1>0:
                bix1 = np.repeat(keys[one,0],counts[one])
                value1 = np.abs(data['h1_eps'][bix1-r,p1])
                keep[:T1] = value1>=thresh
                if not keep[:T1].all():
                    truncation['max_dropped_eps'] = float(value1[~keep[:T1]].max())
            if thresh is not None and T>T1:
                bix2 = np.repeat(keys[~one,0],counts[~one])
                ell = data['chol_ell']
                value2 = np.abs(ell[bix2,p2[:,0]]*ell[bix2,p2[:,1]])
                keep[T1:] = value2>=thresh
                if not keep[T1:].all():
                    truncation['max_dropped_W'] = float(value2[~keep[T1:]].max())
            truncation['dropped_one_body'] = int(T1-np.count_nonzero(keep[:T1]))
            truncation['dropped_two_body'] = int(T-T1-np.count_nonzero(keep[T1:]))
            if not keep.all():
                truncation['dropped_weight'] = math.fsum(a[~keep])
                # New key ranges from the number of kept terms before each old boundary.
                kept_before = np.concatenate([[0],np.cumsum(keep)])
                keys = np.column_stack([keys[:,:3],kept_before[keys[:,3]],kept_before[keys[:,4]]])
                keys = keys[keys[:,4]>keys[:,3]]
                a,p1,d1 = a[keep],p1[keep[:T1]],d1[keep[:T1]]
                p2,d2 = p2[keep[T1:]],d2[keep[T1:]]
                T1 = p1.shape[0]
                if a.size==0:
                    raise ValueError('thresh1/thresh2 dropped every term')

        for bix,s0,s1,start,stop in keys.tolist():
            if s1<0:
                key = (int(bix),(int(s0),))
                p,d = p1[start:stop].reshape(-1,1),d1[start:stop].reshape(-1,1)
            else:
                key = (int(bix),(int(s0),int(s1)))
                p,d = p2[start-T1:stop-T1],d2[start-T1:stop-T1]
            sg.term_info[key] = {'a':a[start:stop],'p':p,'d':d}
        ham.size_groups[n] = sg

        ham.h1 = xp.asarray(data['h1'])
        ham.v0 = np.array(data['v0'])
        ham.mu = np.array(data['mu'],dtype=np.float64)
        #ham.optimized_nelec = data['nelec']
        #ham.dropped_identity = data['dropped_identity']
        #ham.truncation = truncation
        #ham.optimized_metadata = data['metadata']
        #ham.optimized_metadata['truncation'] = truncation
        if iprint>0:
            print('optimized decomposition:',os.fspath(path))
            print('policy=',data['policy'],'decomp_type=',ham.decomp_type)
            print('number of bases=',len(sg.basis),'(thc:',r,', h1:',data['h1_eps'].shape[0],')')
            print('number of terms=',a.size,'dropped identity=',data['dropped_identity'])
            if thresh is not None:
                print('truncation: thresh=',thresh,
                      'dropped one-body=',truncation['dropped_one_body'],
                      'dropped two-body=',truncation['dropped_two_body'],
                      'dropped weight=',truncation['dropped_weight'])
                print('largest dropped |eps-mu|=',truncation['max_dropped_eps'],
                      'largest dropped |W|=',truncation['max_dropped_W'])
            print('mu=',data['mu'],'denom (file)=',data['denom'],'lambda_electronic (file)=',data['lambda_electronic'])
        policy,expected_denom = data['policy'],data['denom']
        dropped_weight = truncation['dropped_weight']
        if nelec is None:
            nelec = data['nelec']
        del data,a,p1,d1,p2,d2

        if parse:
            if nelec is None:
                raise ValueError('parse=True needs nelec: the optimization stored no particle numbers')
            nelec = tuple(int(x) for x in nelec)
            _check_sector(policy,nelec)
            ham.parse_decomposition(nelec,iprint=iprint)
            if not math.isclose(float(ham.denom),expected_denom-dropped_weight,
                                rel_tol=1e-12,abs_tol=1e-12*expected_denom):
                raise ValueError(f'Parsed denom {float(ham.denom)} disagrees with the file '
                                 f'({expected_denom} minus dropped weight {dropped_weight})')
        return ham

    def decompose_h2(self,fname,dt,iprint=0,uniform='coefficient'):
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
                #print(W)
            S = np.eye(nbasis) if isometry else sg.overlaps[-1]
            if self.decomp_type=='ab_only':
                self.v0 += .5*(np.dot(X,np.dot(W*S,X.T)))[None,:,:]
                v0 = np.zeros((2,nbasis))
            else:
                v0 = .5*np.diag(W)
                if not isometry:
                    v0 *= np.diag(sg.overlaps[-1])
                v0 = np.tile(v0[None,:],(2,1))

            if self.decomp_type!='aa_only':
                sg.decompose_hubbard(bix,dt,uniform=uniform,thresh=self.thresh)

            if self.decomp_type=='aa_only':
                sg.decompose_aa_only(bix,dt,uniform=uniform,thresh=self.thresh)
            elif self.decomp_type=='ab_only':
                sg.decompose_ab_only(bix,dt,uniform=uniform,thresh=self.thresh)
            else:
                sg.decompose_full_2body(bix,dt,uniform=uniform,thresh=self.thresh)
            self.v0 += np.einsum('sp,xp,yp->sxy',v0,X,X) 
