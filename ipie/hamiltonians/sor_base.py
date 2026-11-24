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
        self.E0 = 0.

        self.size_groups = dict() 
        # for cholesky & double factorization:
        # should have only one size group with key=self.nbasis
        # for thc: should have two size groups
        # one with key=self.nbasis
        # one with key=number of auxiliary modes

    def decompose_h1(self,h1,dt,uniform='coefficient',iprint=0):
        # for cholesky vectors, pass the effective h'
        if iprint>0:
            print('1-body decomposition: ')
            print('a=',1./dt)
        assert uniform in ['coefficient','rotation']

        if self.nbasis not in self.size_groups:
            self.size_groups[self.nbasis] = SizeGroup()
        sg = self.size_groups[self.nbasis]

        same_spin = True
        if isinstance(h1,(list,tuple)):
            same_spin = False
        else:
            h1 = [h1,h1]
        h1 = np.stack(h1,axis=0)
        self.h1 = xp.asarray(h1)

        if np.linalg.norm(self.v0[0]-self.v0[1])>self.thresh:
            same_spin = False
        if same_spin:
            ek,vk = np.linalg.eigh(h1[0]+self.v0[0]) 
            if iprint>0:
                print(f'bands for both spin:',ek)

            sg.add_basis(vk,ek,'h1')
            bix = sg.get_basis_index()
            for s in (0,1):
                sg.decompose_1body(bix,s,dt,uniform=uniform,thresh=self.thresh)
            self.size_groups[self.nbasis] = sg
            return

        for s,v0 in enumerate(self.v0):
            if s==1 and self.decomp_type=='aa_only':
                continue

            ek,vk = np.linalg.eigh(h1[s]+v0) 
            if iprint>0:
                print(f'spin={s} bands:',ek)
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
                assert len(d[d<-1.])==0

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
    """Read and check a file written by optimize_cholesky.io.save_decomposition (format version 2 or 3).

    Version 2 files have no Cholesky shift and one (n,n) h1_X; it is returned
    as (k,n,n) like version 3. chol_lam (the shifted channel eigenvalues) is
    None without a shift.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading optimized decompositions requires h5py; install it with "
                          "'pip install h5py'") from exc
    from ipie.hamiltonians.optimize_cholesky.io import FORMAT,READABLE_VERSIONS,POLICY_TO_DECOMP_TYPE
    names = ('h1','v0','chol_X','chol_ell','h1_X','h1_eps','a','p1','d1','p2','d2','keys')
    with h5py.File(path,'r') as f:
        fmt,version = f.attrs.get('format'),int(f.attrs.get('format_version',-1))
        if fmt!=FORMAT or version not in READABLE_VERSIONS:
            raise ValueError(f'{path} is not a {FORMAT} file of format_version {READABLE_VERSIONS} '
                             f'(found format={fmt!r}, version={version}); save it with '
                             'optimize_cholesky.io.save_decomposition')
        A = f.attrs
        data = {name:f[name][()] for name in names}
        shifted = version>=3 and bool(A['has_cholesky_shift'])
        data['chol_lam'] = f['chol_lam'][()] if shifted else None
        data['cholesky_shift'] = np.array(A['cholesky_shift'],dtype=np.float64) if shifted else None
        data['cholesky_energy'] = float(A['cholesky_energy']) if version>=3 else 0.
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
            'cholesky_shift':None if data['cholesky_shift'] is None else data['cholesky_shift'].tolist(),
            'cholesky_energy':data['cholesky_energy'],
            'provenance':json.loads(str(p.attrs['record'])),'user':json.loads(str(p.attrs['user_metadata'])),
        }

    n,r = data['nbasis'],data['nchol']
    keys,T,T1 = data['keys'],data['a'].shape[0],data['p1'].shape[0]
    k = data['h1_eps'].shape[0] if data['h1_eps'].ndim==2 else -1
    if version==2 and data['h1_X'].ndim==2 and k>=0:
        data['h1_X'] = np.repeat(data['h1_X'][None],k,axis=0)
    def require(condition,message):
        if not condition:
            raise ValueError(f'{path}: {message}')
    require(data['policy'] in POLICY_TO_DECOMP_TYPE and data['decomp_type']==POLICY_TO_DECOMP_TYPE[data['policy']],
            'inconsistent policy/decomp_type')
    require(data['chol_X'].shape==(r,n,n) and data['chol_ell'].shape==(r,n),'invalid factor arrays')
    require(k>=0 and data['h1_X'].shape==(k,n,n) and data['h1_eps'].shape==(k,n),'invalid h1 basis arrays')
    require(data['h1'].shape==(n,n) and data['v0'].shape==(2,n,n),'invalid h1/v0 arrays')
    if data['cholesky_shift'] is not None:
        require(data['cholesky_shift'].shape==(r,2) and data['chol_lam'].shape==(2,r,n),
                'invalid Cholesky shift arrays')
        require(data['nelec'] is not None,'a Cholesky shift needs the stored particle numbers')
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

def compute_cholesky_shift(chol):
    # computes cholesky shifts as medians of bands
    ell = np.linalg.eigvalsh(chol)                          # (r, n)
    c = np.median(ell, axis=1)
    c[np.abs(c) < 1e-10*np.abs(ell).max()] = 0.0            # medians that are only roundoff
    return ell,c

def compute_h1e_shift(h1e,eri,chol,nelec,dt1,dt2):
    from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem, build_gauge_state
    from ipie.hamiltonians.optimize_cholesky.channels import build_fixed_decomposition
    from ipie.hamiltonians.optimize_cholesky.shift import optimal_shift
    problem = prepare_problem(h1e, eri, chol, nelec=nelec)      # bare h; nelec=None for decomp_type='all'
    dec = build_fixed_decomposition(problem, build_gauge_state(problem.data), dt1, dt2, 0.0)
    mu = optimal_shift(dec, loss_kind="weighted")                # (mu_up, mu_down)
    return mu

class QCSOR(SumOfRotationBase):

    def make_shift(self,h1e,chol,nelec,dt1,dt2,iprint=0,fname=None):
        """One-shot Cholesky shift (median of each factor's eigenvalues) plus one-body shift.

        h1e is the bare h. With L'_g = L_g - c_g I and N = N_up + N_down, on the
        (N_up, N_down) sector
            H = bare(h + (N-1) sum_g c_g L'_g, L' x L') + 1/2 N(N-1) sum_g c_g^2,
        and mu = compute_h1e_shift(...) is chosen for that bare Hamiltonian.
        Returns (h_t, L') with
            h_t = h + (N-1) sum_g c_g L'_g - 1/2 sum_g L'_g^2 - mu
        (a pair of matrices if mu_up != mu_down), and adds
        1/2 N(N-1) sum_g c_g^2 + mu.N to self.E0, so that
        H = sum_s h_t,s n_s + 1/2 sum_g rho'_g^2 + E0 on the sector.
        """
        ell,c = compute_cholesky_shift(chol)
        if iprint>0:
            print('cholesky bands =',ell)
            print('shift =',c)
        ell -= c[:,None]
        if iprint>0:
            print('shifted min =',ell[:,0])
            print('shifted max =',ell[:,-1])

        n = h1e.shape[0]
        N = sum(nelec)
        Lp  = chol - c[:, None, None]*np.eye(n)
        h_t = h1e + (N-1)*np.einsum("g,gpq->pq", c, Lp)     # bare h for the factors Lp
        if iprint>0:
            w = np.linalg.eigvalsh(h_t)
            print('shifted h1e band =',w)
        self.E0 += 0.5*N*(N-1)*np.sum(c**2)                 # add to energies measured with (h_t, Lp)
        eri_new = np.einsum("gpr,gqs->prqs", Lp, Lp)

        mu = compute_h1e_shift(h_t,eri_new,Lp,nelec,dt1,dt2)
        self.E0 += mu[0]*nelec[0]+mu[1]*nelec[1]
        h_t -= 0.5 * np.einsum("gpr,grs->ps",Lp,Lp)         # h' = bare h - 1/2 sum_g L'_g^2
        if np.fabs(mu[0]-mu[1])<1e-16:
            h_t -= mu[0]*np.eye(n)
        else:
            h_t = h_t - mu[0]*np.eye(n), h_t - mu[1]*np.eye(n)
        if fname is not None:
            np.savez_compressed(fname,h1e_eff=h_t,chol=Lp,eri=eri_new,dt1=dt1,dt2=dt2,E0=self.E0)
        return h_t,Lp

    def load_shift(self,fname):
        data = np.load(fname+'.npz')
        h1e_eff = data['h1e_eff']
        chol = data['chol']
        dt1 = data['dt1']
        dt2 = data['dt2']
        self.E0 = data['E0']
        return h1e_eff,chol,dt1,dt2

    @classmethod
    def initialize_cholesky(cls,eri,cmax=None):
        from ipie.utils.linalg import modified_cholesky
        nbasis = eri.shape[0]
        M = eri.reshape((nbasis**2, nbasis**2))
        if cmax is None:
            cmax = nbasis**2
        return modified_cholesky(M, cmax=cmax).reshape(-1, nbasis, nbasis)

    @classmethod
    def optimize_from_cholesky(cls,h1e,chol,nelecs,dt1,dt2,source,*,
                               theta=0.1,seed=None,partition='balanced_log',loss_kind='weighted',
                               maxiter=None,verbosity=None,gradient_atol=None,gradient_rtol=None,
                               maxsweep=None,inner_atol=None,inner_rtol=None,
                               amplitude_limit=None,one_body_limit=None,
                               options=None,inner_options=None,
                               cholesky_shift=None,optimize_cholesky_shift='tied',
                               drop_identity=False,overwrite=False,metadata=None):
        """Optimize the Cholesky gauge and save the result to the HDF5 file `source`.

        The file is written by optimize_cholesky.io.save_decomposition and is read
        back with QCSOR.from_file(source). Returns the OuterOptimizationResult.
        All options are keyword-only; an unknown name raises TypeError.

        h1e is the bare one-body matrix h. The two-electron integrals are built as
        eri = einsum("gpr,gqs->prqs", chol, chol), so the decomposition is exact for
        the Hamiltonian that chol represents: a truncated Cholesky factorization
        gives the correspondingly truncated Hamiltonian.

        Start and objective:
          theta=0.1         scale of the random starting angles, uniform in [-theta, theta]
          seed=None         seed for those angles (numpy RandomState); None draws from the
                            global np.random state, so np.random.seed(s) still works
          partition, loss_kind   amplitude map and objective of GaugeEvaluator

        Outer L-BFGS. Based on `options` (an LBFGSOptions; default
        LBFGSOptions(verbosity=2, line_search='weak_wolfe')), with each of these
        overriding its field when not None:
          maxiter -> max_iterations (default 100), verbosity (default 2),
          gradient_atol (default 1e-8), gradient_rtol (default 1e-6).
          The outer loop stops at gradient_atol + gradient_rtol*|initial gradient|_inf.

        Inner partition and shift solve at every trial gauge. Based on `inner_options`
        (an InnerOptions; default InnerOptions(optimize_shift=True)), with each of these
        overriding its field when not None:
          maxsweep -> max_sweeps (default 500), inner_atol -> gradient_atol (default
          1e-11), inner_rtol -> gradient_rtol (default 1e-9), amplitude_limit (default
          (1-1e-8)*amplitude_bound), one_body_limit (default 1-1e-8).

        Cholesky shift (needs nelecs). By default a tied shift mu_g of the rotated
        factors is optimized together with the gauge, from the median start:
          optimize_cholesky_shift='tied' (default), 'split' or False: optimize the shift
                            with the gauge ('tied': one mu_g for both spins; 'split':
                            per spin, which measured no better in roadmap step 13 and
                            costs more), or hold it fixed (False).
          cholesky_shift=None, 'median', 'zero' or an (r,) / (r,2) array: the start of
                            the optimized shift ('median' when None), or the fixed shift
                            with optimize_cholesky_shift=False (None: no shift, the
                            behaviour before roadmap step 13). 'median' is each rotated
                            factor's spectral median at the starting gauge
                            (optimize_cholesky.optimizers.median_cholesky_shift).
        With a shift the saved file carries it (format version 3) and is exact only on
        the nelecs sector; its energy integrals stay the physical ones. Without
        nelecs pass optimize_cholesky_shift=False. make_shift is a separate
        preprocessing route and is unchanged.

        drop_identity, overwrite and metadata are passed to save_decomposition.
        """
        from dataclasses import replace
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        from ipie.hamiltonians.optimize_cholesky.models import LBFGSOptions,InnerOptions
        from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem
        from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
        from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge,_shift_mode
        from ipie.hamiltonians.optimize_cholesky.channels import cholesky_shift_array

        def overridden(base,fields):
            given = {name:value for name,value in fields.items() if value is not None}
            return replace(base,**given) if given else base

        # Options first: invalid settings fail before any numerical work.
        if options is None:
            options = LBFGSOptions(verbosity=2,line_search='weak_wolfe')
        elif not isinstance(options,LBFGSOptions):
            raise TypeError("options must be LBFGSOptions")
        options = overridden(options,{'max_iterations':maxiter,'verbosity':verbosity,
                                      'gradient_atol':gradient_atol,'gradient_rtol':gradient_rtol})
        if inner_options is None:
            inner_options = InnerOptions(optimize_shift=True)
        elif not isinstance(inner_options,InnerOptions):
            raise TypeError("inner_options must be InnerOptions")
        inner_options = overridden(inner_options,{'max_sweeps':maxsweep,'gradient_atol':inner_atol,
                                                  'gradient_rtol':inner_rtol,'amplitude_limit':amplitude_limit,
                                                  'one_body_limit':one_body_limit})

        mode = _shift_mode(optimize_cholesky_shift)
        if isinstance(cholesky_shift,str):
            if cholesky_shift not in ('median','zero'):
                raise ValueError("cholesky_shift must be None, 'median', 'zero' or an array")
        elif cholesky_shift is not None:
            cholesky_shift_array(cholesky_shift,np.shape(chol)[0])          # shape and finiteness
        if (mode is not None or cholesky_shift is not None) and nelecs is None:
            raise ValueError('A Cholesky shift requires particle numbers (nelecs); pass '
                             'optimize_cholesky_shift=False (and no cholesky_shift) to optimize without one')

        eri = np.einsum("gpr,gqs->prqs", chol, chol)
        problem = prepare_problem(h1e,eri,chol,nelec=nelecs)
        r = problem.data.nchol
        draws = np.random.rand(r*(r-1)//2) if seed is None else np.random.RandomState(seed).rand(r*(r-1)//2)
        theta0 = theta*(draws*2.-1.)

        evaluator = GaugeEvaluator(problem,tau1=dt1,tau2=dt2,loss_kind=loss_kind,partition=partition,
                                   backend='numpy')
        result = optimize_gauge(evaluator,solver='lbfgs',theta0=theta0,options=options,
                                inner_options=inner_options,optimize_cholesky_shift=optimize_cholesky_shift,
                                cholesky_shift0=cholesky_shift)
        save_decomposition(source,result,drop_identity=drop_identity,overwrite=overwrite,metadata=metadata)
        return result

    @classmethod
    def from_file(cls,path,*,nelec=None,parse=True,thresh=1e-6,iprint=0):
        """Build a QCSOR from a file written by optimize_cholesky.io.save_decomposition.

        Array-only: term_info holds numpy slices of the file arrays and no
        per-channel Python objects are built. One size group of key nbasis:
          * bix=gamma (0..r-1): factor eigenvectors (isometry, S=None), integrals
            W=outer(ell,ell), tag 'thc' (used by local_energy);
          * bix=r (and r+1 if the spins need separate bases): the one-body
            channel basis, integrals eps-mu_s, tag 'h1'. It is the eigenbasis of
            h, or of h_s = h + sum_g (m_g - mu[g,s]) L_g with a Cholesky shift;
          * term_info keys in file order, so parse_decomposition gives the file's
            global term order (minus any truncated terms).

        Truncation (default thresh=1e-6; thresh=None keeps every term, exact):
          * thresh drops one-body channel (k,s) if |eps_k - mu_s| < thresh, i.e.
            the SHIFTED orbital energy, the term (eps_k-mu_s) n_ks it represents
            (equivalently |d| < thresh*tau1);
          * thresh drops two-body channels with |W_ab| = |ell_a ell_b| < thresh
            (equivalently |d_L d_R| < thresh*tau2). All branches of a source share
            W, so a source is always dropped as a whole. With a Cholesky shift the
            channel's own |lambda_a,s lambda_b,s'| (its leg spins) is compared, so
            the branches of a split quartet can be dropped separately.
        Truncation is not exact: the walk then uses H minus the dropped terms.
        ham.h1, v0 and the W integrals are NOT truncated, so local_energy still
        measures the full H. Bases are kept even if all of their terms are dropped,
        so bix stays valid. iprint>0 prints the thresholds, counts, dropped weight
        and largest dropped |eps-mu| and |W|.

        Energies. On the (N_up, N_down) sector, sum_i a_i U_i = Lambda I - H with
        Lambda = ham.denom + mu.N + ham.cholesky_energy (ham.denom: the parsed,
        kept weight sum; mu = ham.mu; cholesky_energy = E_c, 0 without a Cholesky
        shift). The file's lambda_electronic attribute is the untruncated Lambda.
        ham.E0 is the constant added to a local energy measured with ham.h1. The
        energy integrals (ham.h1, v0 and the 'thc' W = ell ell^T) are the
        physical, unshifted ones even when the channels carry a Cholesky shift,
        so E0 stays 0.

        Sets h1 = h' = h - 1/2 sum L^2 for both spins (shape (2,n,n), as
        decompose_h1 does), v0, mu (the file's one-body (mu_up, mu_down)),
        cholesky_shift ((r,2) or None) and cholesky_energy. Version 2 and 3
        files load. With parse=True (default) parse_decomposition(nelec) is
        called, with nelec defaulting to the particle numbers of the
        optimization; a sector on which the stored policy is not exact raises
        ValueError (hybrid: any; ab_only: (1,1); aa_only: N_down=0; with a
        Cholesky shift: only the stored sector), and the parsed denom is
        checked against the file minus the dropped weight.
        """
        data = _read_decomposition_file(path)
        n,r = data['nbasis'],data['nchol']
        ham = cls(n,thresh=thresh,decomp_type=data['decomp_type'])

        sg = SizeGroup()
        for g in range(r):
            ell = data['chol_ell'][g]
            sg.add_basis(data['chol_X'][g],np.outer(ell,ell),'thc',isometry=True)
        for k,eps in enumerate(data['h1_eps']):
            sg.add_basis(np.array(data['h1_X'][k]),eps,'h1')

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
                if data['chol_lam'] is None:
                    ell = data['chol_ell']
                    value2 = np.abs(ell[bix2,p2[:,0]]*ell[bix2,p2[:,1]])
                else:
                    # Shifted channels: lambda of each leg's own spin (leg 0 is up for (0,1) keys).
                    lam = data['chol_lam']
                    s0,s1 = np.repeat(keys[~one,1],counts[~one]),np.repeat(keys[~one,2],counts[~one])
                    value2 = np.abs(lam[s0,bix2,p2[:,0]]*lam[s1,bix2,p2[:,1]])
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
                    raise ValueError('thresh dropped every term')

        for bix,s0,s1,start,stop in keys.tolist():
            if s1<0:
                key = (int(bix),(int(s0),))
                p,d = p1[start:stop].reshape(-1,1),d1[start:stop].reshape(-1,1)
            else:
                key = (int(bix),(int(s0),int(s1)))
                p,d = p2[start-T1:stop-T1],d2[start-T1:stop-T1]
            sg.term_info[key] = {'a':a[start:stop],'p':p,'d':d}
        ham.size_groups[n] = sg

        ham.h1 = xp.asarray(np.stack([data['h1'],data['h1']]))
        ham.v0 = np.array(data['v0'])
        ham.mu = np.array(data['mu'],dtype=np.float64)
        ham.cholesky_shift = None if data['cholesky_shift'] is None else np.array(data['cholesky_shift'])
        ham.cholesky_energy = float(data['cholesky_energy'])
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
            if data['cholesky_shift'] is not None:
                print('Cholesky shift: E_c=',data['cholesky_energy'],'max|mu_up-mu_down|=',
                      float(np.max(np.abs(data['cholesky_shift'][:,0]-data['cholesky_shift'][:,1]))))
        policy,expected_denom = data['policy'],data['denom']
        shifted_sector = data['nelec'] if data['cholesky_shift'] is not None else None
        dropped_weight = truncation['dropped_weight']
        if nelec is None:
            nelec = data['nelec']
        del data,a,p1,d1,p2,d2

        if parse:
            if nelec is None:
                raise ValueError('parse=True needs nelec: the optimization stored no particle numbers')
            nelec = tuple(int(x) for x in nelec)
            _check_sector(policy,nelec)
            if shifted_sector is not None and nelec!=tuple(shifted_sector):
                raise ValueError(f'This decomposition has a Cholesky shift and is exact only on the sector it '
                                 f'was built for, nelec={tuple(shifted_sector)}, not {nelec}')
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

        #a = 1./np.sqrt(dt)
        #if isinstance(dt,float):
        #    dt = dt*np.ones(len(grouped))
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
            if iprint>1:
                print('basis index=',bix)
                print(W)
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
