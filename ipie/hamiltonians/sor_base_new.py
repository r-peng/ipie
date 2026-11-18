import numpy as np
import pickle
import os
import json
import math
from datetime import datetime, timezone
from ipie.utils.backend import arraylib as xp
# ipie.utils.linalg (scipy), h5py and the optimize_cholesky package are imported lazily
# inside the functions that need them, so a QMC run that only reads a runtime file
# (QCSOR.from_runtime) imports neither scipy nor the optimizer.

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
                sg.term_info[term_key] = {'p':p,'d':d,'ix':xp.asarray(ixs)}
                # at this point not sure what saved ixs is for 
                # might not need if we never compute all overlap ratios

                if iprint>1:
                    print('key=',term_key)
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

_POLICY_TO_DECOMP_TYPE = {'hybrid':'all','ab_only':'ab_only','aa_only':'aa_only'}

SOR_RUNTIME_FORMAT = 'lafqmc-sor-runtime'
SOR_RUNTIME_VERSION = 1
# Two-body spin classes after normalization; (down,up) is stored as (0,1) with legs swapped.
_SPIN_CLASSES = ((0,0),(1,1),(0,1))


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading/writing decomposition files requires h5py; install it with "
                          "'pip install h5py'") from exc
    return h5py


def _file_format(path):
    """The 'format' attribute of an HDF5 file, or None if it is not a readable HDF5 file."""
    h5py = _h5py()
    try:
        with h5py.File(path,'r') as handle:
            value = handle.attrs.get('format',None)
    except OSError:
        return None
    return None if value is None else str(value)


def _optimized_source(source):
    """Return (FixedDecomposition, metadata) from a path or an optimize_cholesky object.

    Accepts an HDF5 path written by optimize_cholesky.io.save_decomposition, a
    SavedDecomposition, a FixedDecomposition, or any result exposing
    .decomposition (outer result, evaluation, gauge gradient, partition result).
    """
    from ipie.hamiltonians.optimize_cholesky.io import SavedDecomposition, load_decomposition
    from ipie.hamiltonians.optimize_cholesky.models import FixedDecomposition
    if isinstance(source, (str, os.PathLike)):
        source = load_decomposition(source)
    if isinstance(source, SavedDecomposition):
        return source.decomposition, source.metadata
    if isinstance(source, FixedDecomposition):
        return source, {}
    decomposition = getattr(source, 'decomposition', None)
    if isinstance(decomposition, FixedDecomposition):
        return decomposition, {}
    raise TypeError('source must be a path, SavedDecomposition, FixedDecomposition or an '
                    'optimize_cholesky result exposing .decomposition')


def _check_sector(policy,nelec):
    """hybrid is exact on every (N_up,N_down) sector (H = B + mu.N - sum a U as operators);
    ab_only only on (1,1); aa_only only on sectors with N_down=0."""
    if policy=='ab_only' and tuple(nelec)!=(1,1):
        raise ValueError(f'ab_only decompositions are exact only for nelec=(1,1), not {nelec}')
    if policy=='aa_only' and nelec[1]!=0:
        raise ValueError(f'aa_only decompositions are exact only for N_down=0, not {nelec}')


def _group_by_first_occurrence(key_ids):
    """Stable grouping of integer keys, with groups ordered by first occurrence.

    Returns (perm, keys, counts): key_ids[perm] is grouped, groups appear in the
    order their key first occurs, and the original order is kept inside a group.
    This is exactly the term order that SizeGroup.add_term followed by
    parse_decomposition produces when channels are added one by one.
    """
    key_ids = np.asarray(key_ids,dtype=np.int64)
    if key_ids.size==0:
        return np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.int64)
    unique,first = np.unique(key_ids,return_index=True)
    order = np.argsort(first,kind='stable')
    rank_of_unique = np.empty(unique.size,dtype=np.int64)
    rank_of_unique[order] = np.arange(unique.size,dtype=np.int64)
    rank = rank_of_unique[np.searchsorted(unique,key_ids)]
    perm = np.argsort(rank,kind='stable')
    counts = np.bincount(rank,minlength=unique.size)
    return perm,unique[order],counts


def _require(condition,message):
    if not condition:
        raise ValueError(message)


def _read_decomposition_arrays(path):
    """Read a save_decomposition file as plain arrays (no Channel objects)."""
    from ipie.hamiltonians.optimize_cholesky.io import FORMAT, FORMAT_VERSION
    h5py = _h5py()
    data = dict()
    with h5py.File(path,'r') as f:
        _require(f.attrs.get('format')==FORMAT,f'{path} is not a {FORMAT} file')
        version = int(f.attrs.get('format_version',-1))
        _require(version==FORMAT_VERSION,
                 f'Unsupported decomposition format_version {version}; expected {FORMAT_VERSION}')
        data['format_version'] = version
        data['policy'] = str(f.attrs['policy'])
        data['nelec'] = tuple(int(x) for x in f.attrs['nelec']) if bool(f.attrs['has_nelec']) else None
        data['denom'] = float(f.attrs['denom'])
        data['created'] = str(f.attrs.get('created',''))

        g = f['one_body']
        data['h1_X'] = np.asarray(g['X'][()],dtype=np.float64)
        data['eps'] = np.asarray(g['eigenvalues'][()],dtype=np.float64)
        data['tau1'] = float(g.attrs['tau1'])
        data['shift'] = tuple(float(x) for x in g.attrs['shift'])
        c = g['channels']
        data['s1'] = np.asarray(c['spin'][()],dtype=np.int64).reshape(-1)
        data['a1'] = np.asarray(c['a_i'][()],dtype=np.float64).reshape(-1)
        data['m1'] = np.asarray(c['mode'][()],dtype=np.int64).reshape(-1)
        data['d1'] = np.asarray(c['d'][()],dtype=np.float64).reshape(-1)

        g = f['gauge']
        data['O'] = np.asarray(g['O'][()],dtype=np.float64)
        data['chol'] = np.asarray(g['chol'][()],dtype=np.float64)
        data['ell'] = np.asarray(g['eigenvalues'][()],dtype=np.float64)
        data['X'] = np.asarray(g['X'][()],dtype=np.float64)

        g = f['two_body']
        data['tau2'] = float(g.attrs['tau2'])
        data['src_bix'] = np.asarray(g['sources/bix'][()],dtype=np.int64).reshape(-1)
        data['src_modes'] = np.asarray(g['sources/modes'][()],dtype=np.int64).reshape(-1,2)
        data['amplitudes'] = np.asarray(g['amplitudes'][()],dtype=np.float64).reshape(-1,2)
        data['slices'] = np.asarray(g['source_slices'][()],dtype=np.int64).reshape(-1,2)
        c = g['channels']
        data['bix2'] = np.asarray(c['bix'][()],dtype=np.int64).reshape(-1)
        data['spins2'] = np.asarray(c['spins'][()],dtype=np.int64).reshape(-1,2)
        data['a2'] = np.asarray(c['a_i'][()],dtype=np.float64).reshape(-1)
        data['modes2'] = np.asarray(c['modes'][()],dtype=np.int64).reshape(-1,2)
        data['d2'] = np.asarray(c['d'][()],dtype=np.float64).reshape(-1,2)

        p = f['provenance']
        data['record'] = str(p.attrs.get('record','{}'))
        data['user_metadata'] = str(p.attrs.get('user_metadata','{}'))
    return data


def _check_decomposition_arrays(data,rtol=1e-12):
    """Vectorized version of io._check_loaded plus the asserts load_optimized relies on."""
    X1,eps,chol,ell,X = data['h1_X'],data['eps'],data['chol'],data['ell'],data['X']
    n = X1.shape[0]
    r = chol.shape[0]
    _require(data['policy'] in _POLICY_TO_DECOMP_TYPE,f"Unknown policy {data['policy']!r}")
    _require(X1.shape==(n,n) and eps.shape==(n,),'Invalid one-body eigensystem shape')
    _require(chol.shape==(r,n,n) and ell.shape==(r,n) and X.shape==(r,n,n) and data['O'].shape==(r,r),
             'Invalid Cholesky gauge shapes')
    _require(np.allclose(X1.T@X1,np.eye(n),rtol=0,atol=1e-10),'One-body eigenvectors are not orthonormal')
    scale = max(1.,float(np.max(np.abs(chol),initial=0.)))
    _require(np.allclose(np.einsum('gpa,ga,gqa->gpq',X,ell,X),chol,rtol=0,atol=1e-10*scale),
             'Stored factor eigenpairs do not reconstruct the stored factors')
    _require(np.allclose(data['O']@data['O'].T,np.eye(r),rtol=0,atol=1e-10),'Stored gauge is not orthogonal')

    s1,a1,m1,d1 = data['s1'],data['a1'],data['m1'],data['d1']
    _require(s1.size==a1.size==m1.size==d1.size,'Inconsistent one-body channel arrays')
    _require(np.all((s1==0)|(s1==1)) and np.all((m1>=0)&(m1<n)),'Invalid one-body spins or modes')
    _require(np.all(np.isfinite(a1)&(a1>0)),'One-body weights must be finite and positive')
    shift = data['shift']
    expected = (np.asarray(shift)[s1]-eps[m1])*data['tau1']
    _require(np.all(np.isclose(d1,expected,rtol=1e-13,atol=1e-300)),
             'One-body amplitudes disagree with the stored shift')
    _require(np.all(1.+d1>0),'One-body amplitudes leave the real-exponential domain 1+d>0')
    _require(data['nelec'] is not None or not any(shift),'A nonzero shift requires particle numbers')

    bix2,spins2,a2,modes2,d2 = data['bix2'],data['spins2'],data['a2'],data['modes2'],data['d2']
    nch = bix2.size
    _require(spins2.shape==modes2.shape==d2.shape==(nch,2) and a2.size==nch,'Inconsistent two-body channel arrays')
    _require(np.all((bix2>=0)&(bix2<r)),'Two-body basis index out of range')
    _require(np.all((spins2==0)|(spins2==1)) and np.all((modes2>=0)&(modes2<n)),'Invalid two-body spins or modes')
    _require(np.all(np.isfinite(a2)&(a2>0)),'Two-body weights must be finite and positive')
    _require(np.all(np.abs(d2)<1.),'Two-body amplitudes must satisfy |d|<1')

    src_bix,src_modes,amplitudes,slices = data['src_bix'],data['src_modes'],data['amplitudes'],data['slices']
    nsrc = src_bix.size
    _require(src_modes.shape==(nsrc,2) and amplitudes.shape==(nsrc,2) and slices.shape==(nsrc,2),
             'Inconsistent source arrays')
    if nsrc==0:
        _require(nch==0,'Channels without sources')
    else:
        _require(slices[0,0]==0 and slices[-1,1]==nch and np.all(slices[1:,0]==slices[:-1,1])
                 and np.all(slices[:,1]>=slices[:,0]),'Source channel slices must be contiguous and ordered')
        W = ell[src_bix,src_modes[:,0]]*ell[src_bix,src_modes[:,1]]
        _require(np.all(np.isclose(amplitudes[:,0]*amplitudes[:,1],-W*data['tau2'],rtol=rtol,atol=1e-300)),
                 'A source violates d_left*d_right = -W*tau2')
        owner = np.repeat(np.arange(nsrc),slices[:,1]-slices[:,0])
        _require(np.all(bix2==src_bix[owner]),'Channel basis index disagrees with its source')
        reference = np.abs(amplitudes[owner])
        _require(np.all(np.abs(np.abs(d2)-reference)<=rtol*reference),'Channel amplitudes disagree with their source')

    total = math.fsum(a1)+math.fsum(a2)
    _require(math.isclose(total,data['denom'],rel_tol=rtol),'Stored denom does not match the stored channel weights')
    return n,r


def export_sor_runtime(decomposition_path,runtime_path,*,drop_identity=False,overwrite=False,iprint=0):
    """Convert a save_decomposition file into a compact QCSOR runtime file (run once, serially).

    The decomposition file is read as plain arrays, checked with vectorized
    versions of the load-time checks, and rewritten in the layout QCSOR uses, so
    that QCSOR.from_runtime never builds per-channel Python objects:

      attrs      format/version, policy, decomp_type, nelec, mu, tau1, tau2,
                 denom and lambda_electronic of the retained terms, drop_identity
      h1, v0     h' = h - 1/2 sum_g L_g^2 and v0 = 1/2 sum_g L_g^2 (both spins)
      chol_X, chol_ell   rotated factor eigenvectors (r,n,n) and eigenvalues (r,n)
      h1_X, h1_eps       h eigenvectors (n,n) and eps-mu_s for each h1 basis (k,n);
                         k=1 if mu_up==mu_down or a single spin, else 2 (up, down)
      a          all weights (T,), in the final global term order
      p1, d1     one-body modes/amplitudes (T1,), global terms [0,T1)
      p2, d2     two-body modes/amplitudes (T2,2), global terms [T1,T)
      keys       (K,5) rows [bix, spin0, spin1, start, stop]; spin1=-1 for one-body
      provenance source path/created time, the decomposition provenance record,
                 user metadata and the gauge O

    The term order equals what load_optimized + parse_decomposition produce
    (one-body keys first; keys in first-occurrence order; channel order kept
    within a key), so both loaders give identical a, ix2key and sampling.
    (down,up) channels are stored under (0,1) with their legs swapped.
    drop_identity=True omits channels whose amplitudes are all exactly zero (exact;
    Lambda shrinks by their weight). Returns runtime_path.
    """
    h5py = _h5py()
    runtime_path = os.fspath(runtime_path)
    decomposition_path = os.fspath(decomposition_path)
    if os.path.abspath(runtime_path)==os.path.abspath(decomposition_path):
        raise ValueError('runtime_path must differ from decomposition_path')
    if os.path.exists(runtime_path) and not overwrite:
        raise FileExistsError(f'{runtime_path} exists; pass overwrite=True to replace it')

    data = _read_decomposition_arrays(decomposition_path)
    n,r = _check_decomposition_arrays(data)
    policy,nelec,shift = data['policy'],data['nelec'],data['shift']

    # h1 bases, exactly as load_optimized builds them.
    spins = [int(s) for s in np.unique(data['s1'])]
    if len(spins)==0:
        h1_eps = np.zeros((0,n))
        h1_bix = dict()
    elif len(spins)==1 or shift[0]==shift[1]:
        h1_eps = (data['eps']-shift[spins[0]])[None,:]
        h1_bix = {s:r for s in spins}
    else:
        h1_eps = np.stack([data['eps']-shift[s] for s in spins])
        h1_bix = {s:r+i for i,s in enumerate(spins)}

    # One-body terms: key = spin.
    s1,a1,m1,d1 = data['s1'],data['a1'],data['m1'],data['d1']
    keep1 = (d1!=0.) if drop_identity else np.ones(s1.size,dtype=bool)
    idx1 = np.flatnonzero(keep1)
    perm1,keys1,counts1 = _group_by_first_occurrence(s1[idx1])
    idx1 = idx1[perm1]

    # Two-body terms: normalize (down,up) to (0,1) by swapping legs; key = (bix, spin class).
    spins2,modes2,d2 = data['spins2'],data['modes2'].copy(),data['d2'].copy()
    swap = (spins2[:,0]==1)&(spins2[:,1]==0)
    modes2[swap] = modes2[swap][:,::-1]
    d2[swap] = d2[swap][:,::-1]
    spin_class = np.where(spins2[:,0]==spins2[:,1],spins2[:,0],2)
    keep2 = ~((d2[:,0]==0.)&(d2[:,1]==0.)) if drop_identity else np.ones(spin_class.size,dtype=bool)
    idx2 = np.flatnonzero(keep2)
    perm2,keys2,counts2 = _group_by_first_occurrence(data['bix2'][idx2]*3+spin_class[idx2])
    idx2 = idx2[perm2]

    T1,T2 = idx1.size,idx2.size
    rows = []
    start = 0
    for key,count in zip(keys1,counts1):
        s = int(key)
        rows.append((h1_bix[s],s,-1,start,start+int(count)))
        start += int(count)
    for key,count in zip(keys2,counts2):
        sp = _SPIN_CLASSES[int(key)%3]
        rows.append((int(key)//3,sp[0],sp[1],start,start+int(count)))
        start += int(count)
    assert start==T1+T2
    keys = np.array(rows,dtype=np.int64).reshape(-1,5)

    a = np.concatenate([a1[idx1],data['a2'][idx2]])
    p1,d1_out = m1[idx1],d1[idx1]
    p2,d2_out = modes2[idx2],d2[idx2]
    del modes2,d2

    # Same reductions parse_decomposition performs, for a load-time consistency check.
    denom = float(a[:T1].sum())+(float(a[T1:].sum()) if T2>0 else 0.)
    shift_energy = 0. if nelec is None else shift[0]*nelec[0]+shift[1]*nelec[1]

    X1,eps,chol = data['h1_X'],data['eps'],data['chol']
    L2 = np.einsum('gpr,grq->pq',chol,chol)
    h = (X1*eps)@X1.T
    h1 = h-.5*L2
    v0 = np.tile(.5*L2[None,:,:],(2,1,1))

    with h5py.File(runtime_path,'w') as f:
        f.attrs['format'] = SOR_RUNTIME_FORMAT
        f.attrs['format_version'] = SOR_RUNTIME_VERSION
        f.attrs['created'] = datetime.now(timezone.utc).isoformat()
        f.attrs['policy'] = policy
        f.attrs['decomp_type'] = _POLICY_TO_DECOMP_TYPE[policy]
        f.attrs['has_nelec'] = nelec is not None
        f.attrs['nelec'] = np.array(nelec if nelec is not None else (-1,-1),dtype=np.int64)
        f.attrs['nbasis'] = n
        f.attrs['nchol'] = r
        f.attrs['mu'] = np.array(shift,dtype=np.float64)
        f.attrs['tau1'] = data['tau1']
        f.attrs['tau2'] = data['tau2']
        f.attrs['denom'] = denom
        f.attrs['lambda_electronic'] = denom+shift_energy
        f.attrs['drop_identity'] = bool(drop_identity)
        f.attrs['dropped_identity'] = int(s1.size+spin_class.size-T1-T2)
        f.attrs['n_one_body'] = T1
        f.attrs['n_two_body'] = T2
        f.create_dataset('h1',data=h1)
        f.create_dataset('v0',data=v0)
        f.create_dataset('chol_X',data=data['X'])
        f.create_dataset('chol_ell',data=data['ell'])
        f.create_dataset('h1_X',data=X1)
        f.create_dataset('h1_eps',data=h1_eps.reshape(-1,n))
        f.create_dataset('a',data=a)
        f.create_dataset('p1',data=p1.astype(np.int64))
        f.create_dataset('d1',data=d1_out)
        f.create_dataset('p2',data=p2.astype(np.int64).reshape(-1,2))
        f.create_dataset('d2',data=d2_out.reshape(-1,2))
        f.create_dataset('keys',data=keys)
        g = f.create_group('provenance')
        g.attrs['source_path'] = os.path.abspath(decomposition_path)
        g.attrs['source_created'] = data['created']
        g.attrs['source_format_version'] = data['format_version']
        g.attrs['source_denom'] = data['denom']
        g.attrs['record'] = data['record']
        g.attrs['user_metadata'] = data['user_metadata']
        g.create_dataset('O',data=data['O'])
    if iprint>0:
        print('runtime file:',runtime_path)
        print('policy=',policy,'terms: one-body=',T1,'two-body=',T2,'keys=',keys.shape[0])
        print('dropped identity=',int(s1.size+spin_class.size-T1-T2),'denom=',denom,'mu=',shift)
    return runtime_path


def _read_sor_runtime(path):
    """Read and validate a runtime file written by export_sor_runtime."""
    h5py = _h5py()
    data = dict()
    with h5py.File(path,'r') as f:
        _require(f.attrs.get('format')==SOR_RUNTIME_FORMAT,f'{path} is not a {SOR_RUNTIME_FORMAT} file')
        version = int(f.attrs.get('format_version',-1))
        _require(version==SOR_RUNTIME_VERSION,
                 f'Unsupported runtime format_version {version}; expected {SOR_RUNTIME_VERSION}')
        attrs = {key:f.attrs[key] for key in f.attrs.keys()}
        for name in ('h1','v0','chol_X','chol_ell','h1_X','h1_eps','a','p1','d1','p2','d2','keys'):
            data[name] = f[name][()]
        p = f['provenance']
        provenance = {key:p.attrs[key] for key in p.attrs.keys()}
    data['policy'] = str(attrs['policy'])
    data['decomp_type'] = str(attrs['decomp_type'])
    data['nelec'] = tuple(int(x) for x in attrs['nelec']) if bool(attrs['has_nelec']) else None
    data['nbasis'] = int(attrs['nbasis'])
    data['nchol'] = int(attrs['nchol'])
    data['mu'] = np.array(attrs['mu'],dtype=np.float64)
    data['denom'] = float(attrs['denom'])
    data['lambda_electronic'] = float(attrs['lambda_electronic'])
    data['drop_identity'] = bool(attrs['drop_identity'])
    data['dropped_identity'] = int(attrs['dropped_identity'])
    data['metadata'] = {
        'format':SOR_RUNTIME_FORMAT,'format_version':version,'created':str(attrs.get('created','')),
        'policy':data['policy'],'nelec':data['nelec'],'mu':tuple(float(x) for x in data['mu']),
        'tau1':float(attrs['tau1']),'tau2':float(attrs['tau2']),'denom':data['denom'],
        'lambda_electronic':data['lambda_electronic'],'drop_identity':data['drop_identity'],
        'dropped_identity':data['dropped_identity'],
        'source':{'path':str(provenance.get('source_path','')),
                  'created':str(provenance.get('source_created','')),
                  'format_version':int(provenance.get('source_format_version',-1)),
                  'denom':float(provenance.get('source_denom',float('nan')))},
        'provenance':json.loads(str(provenance.get('record','{}'))),
        'user':json.loads(str(provenance.get('user_metadata','{}'))),
    }

    n,r = data['nbasis'],data['nchol']
    keys,a = data['keys'].reshape(-1,5),data['a']
    T1,T = data['p1'].shape[0],a.shape[0]
    _require(data['chol_X'].shape==(r,n,n) and data['chol_ell'].shape==(r,n),'Invalid factor arrays')
    _require(data['h1_X'].shape==(n,n) and data['h1_eps'].ndim==2 and data['h1_eps'].shape[1]==n,
             'Invalid h1 basis arrays')
    _require(data['h1'].shape==(n,n) and data['v0'].shape==(2,n,n),'Invalid h1/v0 arrays')
    _require(data['d1'].shape==(T1,) and data['p2'].shape==data['d2'].shape==(T-T1,2),'Invalid term arrays')
    k = data['h1_eps'].shape[0]
    if keys.shape[0]>0:
        _require(keys[0,3]==0 and keys[-1,4]==T and np.all(keys[1:,3]==keys[:-1,4])
                 and np.all(keys[:,4]>=keys[:,3]),'Runtime key ranges must be contiguous and ordered')
        one = keys[:,2]<0
        _require(np.all(one[:np.count_nonzero(one)]) and keys[np.count_nonzero(one)-1,4]==T1
                 if np.any(one) else T1==0,'One-body keys must come first and cover [0,T1)')
        _require(np.all((keys[one,0]>=r)&(keys[one,0]<r+k)) and np.all((keys[~one,0]>=0)&(keys[~one,0]<r)),
                 'Runtime key basis index out of range')
        _require(len({tuple(row[:3]) for row in keys.tolist()})==keys.shape[0],'Duplicate runtime keys')
    else:
        _require(T==0,'Terms without keys')
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
        """Optimize the Cholesky gauge, save the decomposition to `source`.

        Extra keyword arguments: runtime_path (if given, also write the compact
        runtime file with export_sor_runtime), drop_identity and
        overwrite_runtime (both passed to export_sor_runtime).
        """
        from ipie.hamiltonians.optimize_cholesky.io import save_decomposition
        from ipie.hamiltonians.optimize_cholesky.models import OuterOptions,InnerOptions
        from ipie.hamiltonians.optimize_cholesky.problem import prepare_problem
        from ipie.hamiltonians.optimize_cholesky.gradients import GaugeEvaluator
        from ipie.hamiltonians.optimize_cholesky.optimizers import optimize_gauge
        problem = prepare_problem(h1e,eri,chol,nelec=nelecs)
    
        r = problem.data.nchol
        theta = kwargs.get('theta',0.1)
        theta = theta*(np.random.rand(r * (r - 1) // 2)*2.-1.)
    
        partition = kwargs.get('partition','balanced_log')
        evaluator = GaugeEvaluator(problem, tau1=dt1, tau2=dt2, partition=partition, backend='numpy')
    
        maxiter = kwargs.get('maxiter',100)
        maxsweep = kwargs.get('maxsweep',100)
        verbosity = kwargs.get('verbosity',2)
        result = optimize_gauge(evaluator,
            solver='lbfgs',
            theta0=theta,
            options=OuterOptions(max_iterations=maxiter,verbosity=verbosity, line_search='weak_wolfe'),
            inner_options=InnerOptions(max_sweeps=maxsweep, gradient_atol=1e-6, gradient_rtol=1e-6,optimize_shift=True),
    )
        save_decomposition(source,result,problem=problem)
        runtime_path = kwargs.get('runtime_path',None)
        if runtime_path is not None:
            export_sor_runtime(source,runtime_path,drop_identity=kwargs.get('drop_identity',False),
                               overwrite=kwargs.get('overwrite_runtime',False),
                               iprint=1 if verbosity>0 else 0)
        return result

    @classmethod
    def from_runtime(cls,path,*,nelec=None,parse=True,thresh=1e-6,iprint=0):
        """Build a QCSOR from a runtime file written by export_sor_runtime.

        Array-only: no FixedDecomposition or per-channel Python objects are built.
        With parse=True (default) parse_decomposition(nelec) is called, with nelec
        defaulting to the particle numbers of the optimization; a sector on which
        the stored policy is not exact raises ValueError. The parsed denom is
        checked against the value stored at conversion.
        """
        data = _read_sor_runtime(path)
        ham = cls(data['nbasis'],thresh=thresh,decomp_type=data['decomp_type'])
        ham.load_runtime(data,iprint=iprint)
        del data
        if parse:
            if nelec is None:
                nelec = ham.optimized_nelec
            if nelec is None:
                raise ValueError('parse=True needs nelec: the optimization stored no particle numbers')
            nelec = tuple(int(x) for x in nelec)
            _check_sector(ham.runtime_metadata['policy'],nelec)
            ham.parse_decomposition(nelec,iprint=iprint)
            stored = ham.runtime_metadata['denom']
            if not math.isclose(float(ham.denom),stored,rel_tol=1e-12):
                raise ValueError(f'Parsed denom {float(ham.denom)} disagrees with the runtime file ({stored})')
        return ham

    def load_runtime(self,source,*,iprint=0):
        """Fill this (empty) QCSOR from a runtime file path (or its already-read arrays).

        Builds the same size group, bases, integrals, term_info keys and term
        order as load_optimized, but term_info holds numpy slices (views) of the
        file arrays instead of Python lists. Sets h1, v0, mu, optimized_nelec,
        dropped_identity, runtime_metadata and optimized_metadata (the latter
        with the same 'provenance'/'user' entries as a decomposition file);
        optimized_decomposition is None.
        """
        data = source if isinstance(source,dict) else _read_sor_runtime(source)
        n,r = data['nbasis'],data['nchol']
        if self.size_groups:
            raise ValueError('load_runtime requires an empty QCSOR (no prior decompose_* calls)')
        if n != self.nbasis:
            raise ValueError(f'nbasis={self.nbasis} does not match the runtime basis size {n}')
        if self.decomp_type != data['decomp_type']:
            raise ValueError(f"decomp_type='{self.decomp_type}' does not match policy "
                             f"'{data['policy']}' (expected '{data['decomp_type']}')")

        sg = SizeGroup()
        X,ell = data['chol_X'],data['chol_ell']
        for g in range(r):
            sg.add_basis(X[g],np.outer(ell[g],ell[g]),'thc',isometry=True)
            assert sg.get_basis_index()==g
        for k,eps in enumerate(data['h1_eps']):
            sg.add_basis(data['h1_X'] if k==0 else data['h1_X'].copy(),eps,'h1')
            assert sg.get_basis_index()==r+k

        a,p1,d1,p2,d2 = data['a'],data['p1'],data['d1'],data['p2'],data['d2']
        T1 = p1.shape[0]
        for bix,s0,s1,start,stop in data['keys'].tolist():
            if s1<0:
                key = (int(bix),(int(s0),))
                p,d = p1[start:stop].reshape(-1,1),d1[start:stop].reshape(-1,1)
            else:
                key = (int(bix),(int(s0),int(s1)))
                p,d = p2[start-T1:stop-T1],d2[start-T1:stop-T1]
            sg.term_info[key] = {'a':a[start:stop],'p':p,'d':d}
        self.size_groups[n] = sg

        self.h1 = xp.asarray(data['h1'])
        self.v0 = np.array(data['v0'])
        self.mu = np.array(data['mu'],dtype=np.float64)
        self.optimized_decomposition = None
        self.optimized_nelec = data['nelec']
        self.dropped_identity = data['dropped_identity']
        self.runtime_metadata = data['metadata']
        self.optimized_metadata = data['metadata']
        if iprint>0:
            print('runtime decomposition: policy=',data['policy'],'decomp_type=',self.decomp_type)
            print('number of bases=',len(sg.basis),'(thc:',r,', h1:',data['h1_eps'].shape[0],')')
            print('number of terms=',a.size,'dropped identity=',self.dropped_identity)
            print('mu=',self.mu,'denom=',data['denom'],'lambda_electronic=',data['lambda_electronic'])
        return self

    @classmethod
    def from_optimized(cls,source,*,drop_identity=False,thresh=1e-6,nelec=None,parse=False,
                       keep_decomposition=False,iprint=0):
        """Build a QCSOR from an optimize_cholesky result or a saved HDF5 file.

        If source is a runtime file (export_sor_runtime), this forwards to
        from_runtime; drop_identity was then fixed at conversion, and requesting
        it for a file converted without it raises ValueError.

        Otherwise the optimized channels replace decompose_h2/decompose_h1 and are
        copied verbatim (no threshold is applied; thresh is only stored). This path
        builds per-channel Python objects and is meant for analysis and small
        systems; use export_sor_runtime + from_runtime for production runs.
        Afterwards parse_decomposition(nelec) works unchanged. With parse=True it
        is called here, with nelec defaulting to the particle numbers of the
        optimization; a sector on which the stored policy is not exact raises
        ValueError (hybrid: any; ab_only: (1,1); aa_only: N_down=0). See
        load_optimized for the mapping and keep_decomposition.
        """
        if isinstance(source,(str,os.PathLike)) and _file_format(source)==SOR_RUNTIME_FORMAT:
            ham = cls.from_runtime(source,nelec=nelec,parse=parse,thresh=thresh,iprint=iprint)
            if drop_identity and not ham.runtime_metadata['drop_identity']:
                raise ValueError('drop_identity is fixed when the runtime file is written; this file '
                                 'was exported with drop_identity=False')
            return ham
        decomposition,metadata = _optimized_source(source)
        two = decomposition.two_body
        ham = cls(decomposition.one_body.X.shape[0],thresh=thresh,
                  decomp_type=_POLICY_TO_DECOMP_TYPE[two.policy])
        ham.load_optimized((decomposition,metadata),drop_identity=drop_identity,
                           keep_decomposition=keep_decomposition,iprint=iprint)
        del decomposition
        if parse:
            if nelec is None:
                nelec = two.nelec
            if nelec is None:
                raise ValueError('parse=True needs nelec: the optimization stored no particle numbers')
            nelec = tuple(int(x) for x in nelec)
            _check_sector(two.policy,nelec)
            ham.parse_decomposition(nelec,iprint=iprint)
        return ham

    def load_optimized(self,source,*,drop_identity=False,keep_decomposition=False,iprint=0):
        """Fill this (empty) QCSOR from an optimized decomposition.

        Mapping into one size group of key nbasis:
          * bix=gamma for every rotated Cholesky factor: basis X_gamma
            (orthonormal eigenvectors, isometry=True so S=None), integrals
            W_gamma=outer(ell_gamma,ell_gamma), tag 'thc' (used by local_energy).
          * then the h eigenbasis with tag 'h1' and integrals epsilon-mu_s: one
            shared basis if mu_up==mu_down (or aa_only), else one per spin, as in
            decompose_h1.
          * every channel copied verbatim as (a_i, p, d). Same-spin and
            (up,down) channels keep their order; a (down,up) channel is stored
            under SOR's (0,1) key with p=[beta,alpha], d=[d_R,d_L].
          * self.h1 = h - 1/2*sum_gamma L_gamma^2 (h' = effective h, unshifted),
            because trial.build contracts ham.h1 and the 'thc' local energy adds
            back 1/2*diag(W).D; self.v0 = 1/2*sum_gamma L_gamma^2 (h1+v0 = h).
          * self.mu = (mu_up, mu_down), so parse_decomposition gives
            E0 = mu.N and denom = sum a_i, i.e. denom+E0 = lambda_electronic.
        drop_identity=True omits channels whose amplitudes are all exactly zero
        (U=I). This is exact: parse_decomposition recomputes denom from the
        retained terms, lowering Lambda by their weight. Nonzero amplitudes are
        never thresholded.
        keep_decomposition=False (default) does not retain the FixedDecomposition
        (self.optimized_decomposition is None): its per-channel Python objects
        cost several hundred bytes per term for the lifetime of the run.
        """
        if isinstance(source,tuple) and len(source)==2 and isinstance(source[1],dict):
            decomposition,metadata = source
            decomposition,_ = _optimized_source(decomposition)
        else:
            decomposition,metadata = _optimized_source(source)
        one,two = decomposition.one_body,decomposition.two_body
        gauge = two.gauge
        n = one.X.shape[0]
        if self.size_groups:
            raise ValueError('load_optimized requires an empty QCSOR (no prior decompose_* calls)')
        if n != self.nbasis:
            raise ValueError(f'nbasis={self.nbasis} does not match the optimized basis size {n}')
        expected = _POLICY_TO_DECOMP_TYPE[two.policy]
        if self.decomp_type != expected:
            raise ValueError(f"decomp_type='{self.decomp_type}' does not match policy "
                             f"'{two.policy}' (expected '{expected}')")

        sg = SizeGroup()
        nchol = gauge.chol.shape[0]
        for g in range(nchol):
            ell = np.array(gauge.eigenvalues[g],dtype=np.float64)
            sg.add_basis(np.array(gauge.X[g],dtype=np.float64),np.outer(ell,ell),'thc',isometry=True)
            assert sg.get_basis_index()==g

        shift = tuple(float(m) for m in one.shift)
        eps = np.array(one.eigenvalues,dtype=np.float64)
        X1 = np.array(one.X,dtype=np.float64)
        spins = one.spins
        if len(spins)==1 or shift[0]==shift[1]:
            sg.add_basis(X1,eps-shift[spins[0]],'h1')
            h1_bix = {s:sg.get_basis_index() for s in spins}
        else:
            h1_bix = dict()
            for s in spins:
                sg.add_basis(X1.copy(),eps-shift[s],'h1')
                h1_bix[s] = sg.get_basis_index()

        dropped = 0
        for c in one.channels:
            if drop_identity and c.d==0.:
                dropped += 1
                continue
            # One-body amplitudes may exceed 1 (single branch); bypass decompose_1body's assert.
            sg.add_term(h1_bix[c.spin],[c.spin],float(c.a_i),[int(c.mode)],[float(c.d)])
        for c in two.channels:
            if drop_identity and c.d[0]==0. and c.d[1]==0.:
                dropped += 1
                continue
            (sl,sr),(pl,pr),(dl,dr) = c.spins,c.modes,c.d
            if sl==sr or (sl,sr)==(0,1):
                sg.add_2body_term(int(c.bix),(sl,sr),float(c.a_i),[int(pl),int(pr)],[float(dl),float(dr)])
            else:  # (down, up) -> SOR's (0,1): leg 0 is always spin up
                sg.add_2body_term(int(c.bix),(0,1),float(c.a_i),[int(pr),int(pl)],[float(dr),float(dl)])
        self.size_groups[n] = sg

        L2 = np.einsum('gpr,grq->pq',np.asarray(gauge.chol),np.asarray(gauge.chol))
        h = (X1*eps) @ X1.T
        self.h1 = xp.asarray(h-.5*L2)
        self.v0 = np.tile(.5*L2[None,:,:],(2,1,1))
        self.mu = np.array(shift,dtype=np.float64)
        self.optimized_decomposition = decomposition if keep_decomposition else None
        self.optimized_metadata = metadata
        self.optimized_nelec = two.nelec
        self.dropped_identity = dropped
        if iprint>0:
            print('optimized decomposition: policy=',two.policy,'decomp_type=',self.decomp_type)
            print('number of bases=',len(sg.basis),'(thc:',nchol,', h1:',len(set(h1_bix.values())),')')
            print('number of channels=',decomposition.nterms,'dropped identity=',dropped)
            print('mu=',self.mu,'denom (all channels)=',decomposition.denom)
        return self

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
