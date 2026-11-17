import numpy as np
import pickle
from ipie.utils.backend import arraylib as xp
from ipie.utils.linalg import modified_cholesky
#from ipie.hamiltonians.sor_chol import build_commuting_groups,pack_cholesky
#from ipie.hamiltonians.sor_chol_opt import optimize_cholesky_gauge 
#from ipie.hamiltonians.sor_thc import fit_thc_from_eri
#from ipie.hamiltonians.sor_thc_stage1_comm import fit_thc_from_eri
#from ipie.hamiltonians.sor_thc_stage1_regularized import fit_thc_from_eri
#from ipie.hamiltonians.sor_thc_stage2_weighted import fit_thc_from_eri
#from ipie.hamiltonians.sor_df import optimize_df_commutator,save_grouped 

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


def _optimized_source(source):
    """Return (FixedDecomposition, metadata) from a path or an optimize_cholesky object.

    Accepts an HDF5 path written by optimize_cholesky.io.save_decomposition, a
    SavedDecomposition, a FixedDecomposition, or any result exposing
    .decomposition (outer result, evaluation, gauge gradient, partition result).
    """
    from ipie.hamiltonians.optimize_cholesky.io import SavedDecomposition, load_decomposition
    from ipie.hamiltonians.optimize_cholesky.models import FixedDecomposition
    import os
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


class QCSOR(SumOfRotationBase):

    @classmethod
    def from_optimized(cls,source,*,drop_identity=False,thresh=1e-6,nelec=None,parse=False,iprint=0):
        """Build a QCSOR from an optimize_cholesky result or a saved HDF5 file.

        The optimized channels replace decompose_h2/decompose_h1 and are copied
        verbatim (no threshold is applied; thresh is only stored). Afterwards
        parse_decomposition(nelec) works unchanged. With parse=True it is called
        here, with nelec defaulting to the particle numbers of the optimization;
        a sector on which the stored policy is not exact raises ValueError
        (hybrid: any; ab_only: (1,1); aa_only: N_down=0). See load_optimized for
        the mapping.
        """
        decomposition,metadata = _optimized_source(source)
        two = decomposition.two_body
        ham = cls(decomposition.one_body.X.shape[0],thresh=thresh,
                  decomp_type=_POLICY_TO_DECOMP_TYPE[two.policy])
        ham.load_optimized((decomposition,metadata),drop_identity=drop_identity,iprint=iprint)
        if parse:
            if nelec is None:
                nelec = two.nelec
            if nelec is None:
                raise ValueError('parse=True needs nelec: the optimization stored no particle numbers')
            nelec = tuple(int(x) for x in nelec)
            _check_sector(two.policy,nelec)
            ham.parse_decomposition(nelec,iprint=iprint)
        return ham

    def load_optimized(self,source,*,drop_identity=False,iprint=0):
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
        self.optimized_decomposition = decomposition
        self.optimized_metadata = metadata
        self.optimized_nelec = two.nelec
        self.dropped_identity = dropped
        if iprint>0:
            print('optimized decomposition: policy=',two.policy,'decomp_type=',self.decomp_type)
            print('number of bases=',len(sg.basis),'(thc:',nchol,', h1:',len(set(h1_bix.values())),')')
            print('number of channels=',decomposition.nterms,'dropped identity=',dropped)
            print('mu=',self.mu,'denom (all channels)=',decomposition.denom)
        return self

    def initialize_cholesky(self,eri,cmax=None):
        M = eri.reshape((self.nbasis**2, self.nbasis**2))
        if cmax is None:
            cmax = self.nbasis**2
        return modified_cholesky(M, cmax=cmax).reshape(-1, self.nbasis, self.nbasis)

    def from_cholesky(self,chol=None,eri=None,cmax=None,comm_tol=1e-10,basis_tol=1e-10,fname=None):
        # set comm_tol to negative to not do regrouping
        if chol is None:
            chol = self.initialize_cholesky(eri,cmax=cmax)

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
        return result,fit

    def from_thc(self,eri,rank,comm_lambda=1.,nstarts=4,maxiter=1000,rcond=1e-6,fname=None): 
        fit = fit_thc_from_eri(eri,rank,comm_lambda=comm_lambda,n_starts=nstarts,maxiter=maxiter,rcond=rcond)
        result = [{'X':fit.X,'W':fit.W,'isometry':False}]
        if fname is not None:
            with open(fname+".pkl", "wb") as f:
                pickle.dump({"grouped": result,'eri':eri},f,protocol=pickle.HIGHEST_PROTOCOL)
        return result,fit

    def from_df(self,eri,nt,chol=None,comm_lambda=1.,fname=None,cmax=None):
        if chol is None:
            chol = self.initialize_cholesky(eri,cmax=cmax)
        fit = optimize_df_commutator(eri,nt,chol=chol,comm_lambda=comm_lambda)
        result = save_grouped(fit.U,fit.Z)
        if fname is not None:
            with open(fname+".pkl", "wb") as f:
                pickle.dump({"grouped": result,'eri':eri},f,protocol=pickle.HIGHEST_PROTOCOL)
        return result,fit

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
