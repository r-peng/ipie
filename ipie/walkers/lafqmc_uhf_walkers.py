import numpy as np
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp
from ipie.walkers.base_walkers import BaseWalkers 

def qr(phi,UC=None,thresh=1e-3):
    Q,R = xp.linalg.qr(phi,mode='reduced')
    Rdiag = xp.einsum('wii->wi',R)

    Rabs = xp.fabs(Rdiag)
    assert Rabs[Rabs<thresh].size==0

    sign = xp.sign(Rdiag)
    Q *= sign[:,None,:]

    if UC is not None:
        Rinv = xp.linalg.inv(R)
        UC = xp.einsum('wdpi,wij->wdpj',UC,Rinv)
        UC *= sign[:,None,None,:]
    return Q,UC

def compute_uBS(uB,S,s):
    ne = uB.shape[-1]
    if S.shape[1]==ne:
        S_ = S
    else:
        if s==0:
            S_ = S[:,:ne]
        else:
            S_ = S[:,-ne:]
    return xp.einsum('kri,wij->kwrj',uB,S_)

def compute_uDu(uBS,uC,s):
    ne = uC.shape[-1]
    if uBS.shape[-1]==ne:
        uBS_ = uBS
    else:
        if s==0:
            uBS_ = uBS[:,:,:,:ne]
        else:
            uBS_ = uBS[:,:,:,-ne:]
    return xp.einsum('kwrj,kwsj->kwrs',uBS_,uC)

def compute_SC(S,C,s,w=None,i=None):
    ne = C.shape[-1]
    if S.shape[-1]==ne:
        S_ = S
    else:
        if s==0:
            S_ = S[:,:,:ne]
        else:
            S_ = S[:,:,-ne:]
    if w is None:
        return xp.einsum('wij,wxj->wix',S_,C)
    else:
        return xp.einsum('wij,wrj->wir',S_[w],C[i,w]) 

def update_phi(C,u,d,uC,w,i):
    duC = d[:,:,None]*uC[w,i]
    C[w] += xp.einsum('wxr,wri->wxi',u,duC)
    return C,duC

def update_Uphi(UC,u2,duC,w):
    UC[w] += xp.einsum('dwpr,wri->dwpi',u2,duC)
    return UC

class UHFWalkers(BaseWalkers):

    def __init__(
        self,
        initial_walker: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        mpi_handler,
        write_filepath=None,
        write_restart=False,
        write_freq=None,
        write_time=None,
        verbose: bool = False,
    ):
        assert len(initial_walker.shape) == 2
        self.nup = nup
        self.ndown = ndown
        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        super().__init__(
            nwalkers,
            write_filepath=write_filepath,
            write_restart=write_restart,
            write_freq=write_freq,
            write_time=write_time,
            verbose=verbose,
        )

        self.phi = xp.array([initial_walker.copy() for iw in range(self.nwalkers)])

        self.buff_names = ['phi','weight','phase']
        self.phase = xp.ones(walkers.nwalkers) 

        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = xp.zeros(self.buff_size)

    def get_phi(self,UC=False):
        phi = [self.phi[:,:,:self.nup],self.phi[:,:,self.nup:]]
        if not UC:
            return phi
        UC = [self.UC[:,:,:,:self.nup],self.UC[:,:,:,self.nup:]]
        return phi,UC

    def set_phi(self,phi,UC=None):
        self.phi[:,:,:self.nup] = phi[0]
        self.phi[:,:,self.nup:] = phi[1]
        if UC is None:
            return
        self.UC[:,:,:,:self.nup] = UC[0]
        self.UC[:,:,:,self.nup:] = UC[1]

    #def get_S_right(self,axis1,axis2):
    #    if self.S is None:
    #        return [self.Sa,self.Sb]
    #    return [self.S[:,:self.nup],self.S[:,self.nup:]]

    #def get_S_left(self):
    #    if self.S is None:
    #        return [self.Sa,self.Sb]
    #    return [self.S[:,:,:self.nup],self.S[:,:,self.nup:]]

    @plum.dispatch
    def build_trial(self,trial:SingleDet,set_buffnames=False):
        phi = self.get_phi()
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,trial.psi)] 
        self.Sa,self.Sb = [xp.linalg.inv(Si) for Si in CB]
        self.S = None
        if set_buffnames:
            self.buff_names += ['Sa','Sb']
    
    @plum.dispatch
    def build_trial(self,trial:SingleDetGHF,set_buffnames=False):
        phi = self.get_phi()
        B = [trial.psi[:self.nbasis],trial.psi[self.nbasis:]]
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,B)] 
        CB = xp.concatenate(CB,axis=1)
        self.S = xp.linalg.inv(CB)
        if set_buffnames:
            self.buff_names += ['S']

    def build_hamiltonian(self,hamiltonian):
        self.UC = xp.einsum('dxp,wxi->wdpi',hamiltonian.chol_basis,self.phi)

    def get_uC(self,key,p):
        chol_idx,typ = key

        uC = self.UC[:,chol_idx]
        uC = [uC[:,:,:self.nup],uC[:,:,self.nup:]

        if typ=='h2ab':
            p = [p[:,:1],p[:,1:]]
            uC = [xp.asarray([uC[s][:,pi] for pi in p[s]]) for s in (0,1)]
        else:
            s = {'a':0,'b':1}[typ[-1]]
            uC = xp.asarray([uC[s][:,pi] for pi in p])
        self.uC[key] = uC
        return uC

    def compute_uBS(self,key,uB):
        _,typ = key
        if self.S is None:
            S = [self.Sa,self.Sb]
        else:
            S = [self.S,self.S]

        if typ=='h2ab':
            uBS = [compute_uBS(uB[s],S[s]) for s in (0,1)]
        else:
            s = {'a':0,'b':1}[typ[-1]]
            uBS = compute_uBS(uB,S[s],s)
        self.uBS[key] = uBS

    def compute_M(self,key,d):
        _,typ = key
        uBS = self.uBS[key]
        uC = self.uC[key]
        if typ=='h2ab':
            M = [compute_uDu(uBS[s],uC[s],s)[:,:,0,0] for s in (0,1)] 
            M = d[:,None,:]*xp.stack(M,axis=2) + 1.
            ovlp = M.prod(axis=2)

            M = (1./M)*d[:,None,:]
        else:
            s = {'a':0,'b':1}[typ[-1]]
            M = compute_uDu(uBS,uC,s)
            M = d[:,None,:,None]*M + np.eye(d.shape[1])[None,None,:,:]
            ovlp = xp.linalg.det(M)

            M = xp.linalg.inv(M) * d[:,None,None,:]
        self.M2[key] = M
        return ovlp

    def compute_ovlp_ratio(self,hamiltonian,trial):
        ovlp = xp.zeros((hamiltonian.nterms,self.nwalkers))
        self.uC = dict()
        self.uBS = dict()
        self.M2 = dict()
        for key,p in hamiltonian.p_dict.items():
            d = hamiltonian.d_dict[key]
            kix = hamiltonian.kix_dict[key]

            self.get_uC(key,p)
            uB = trial.get_uB(key,p)
            self.compute_uBS(key,uB)
            ovlp[kix] = self.compute_M1(key,d)
        return ovlp

    def parse_sampled_rotations(self,kixs):
        ix1 = dict()
        ix2 = dict()
        for kix in kixs:
            key,i = self.kix2key[kix]
            if key not in ix1:
                ix1[key] = []
            ix1[key].append(kix)
            if key not in ix2:
                ix2[key] = []
            ix2[key].append(i)
        for key in ix1:
            ix1[key] = xp.asarray(self.ix1[key])
        for key in ix2:
            ix2[key] = xp.asarray(self.ix2[key])
        return ix1,ix2

    def update_walkers(self,kixs,hamiltonian):
        ix1,ix2 = self.parse_sampled_rotations(kixs)
        for key,w in ix1.items():
            _,typ = key
            i = self.ix2[key]
            u,d,u2 = hamiltonian.get_ud(key,i)

            self.update_phi(key,w,i,u,d,u2)
            self.update_S(key,w,i) 

    def update_phi(self,key,w,i,u,d,u2):
        _,typ = key
        uC = self.uC[key]
        phi,UC = self.get_phi(UC=True)
        if typ=='h2ab':
            d = [d[:,:1],d[:,1:]]
            u = [u[:,:,:1],u[:,:,1:]]
            u2 = [u2[:,:,:,:1],u2[:,:,:,1:]]
            for s in (0,1):
                phi[s],duC = update_phi(phi[s],u[s],d[s],uC[s],w,i)
                UC[s] = update_Uphi(UC[s],u2[s],duC,w)
        else:
            s = {'a':0,'b':1}[typ[-1]]
            phi[s],duC = update_phi(phi[s],u,d,uC,w,i)
            UC[s] = update_Uphi(UC[s],u2,duC,w)
        self.set_phi(phi,UC)

    def update_S(self,key,w,i):
        _,typ = key
        uC = self.uC[key]
        if self.S is None:
            S = [Sa,Sb]
        else:
            S = [self.S,self.S]
        if typ=='h2ab':
            SCu = [compute_SC(S[s],uC[s],s,w=w,i=i) for s in (0,1)]
            self.update_S2(key,w,i,SCu)
        else:
            s = {'a':0,'b':1}[typ[-1]]
            SCu = compute_SC(S[s],uC,s,w=w,i=i)
            self.update_S1(key,w,i,SCu)

    def update_S1(self,key,w,i,SCu):
        _,typ = key
        s = {'a':0,'b':1}[typ[-1]]
        M = self.M2[key][i,w]
        uBS = self.uBS[key][i,w]
        MuBS = xp.einsum('wrs,wsi->wri',M,uBS)
        S1 = xp.einsum('wir,wrj->wij',SCu,MuBS)
        if self.S is None:
            if s==0:
                self.Sa[w] -= S1
            else:
                self.Sb[w] -= S1 
        else:
            self.S[w] -= S1 

    def update_S2(self,key,w,i,SCu):
        M = self.M2[key][i,w]
        uBS = self.uBS[key]
        uBS = [uBS[0][i,w],uBS[1][i,w]]
        if self.S is None:
            S = [self.Sa,self.Sb]
            M = [M[:,:1],M[:,1:]]
            for s in (0,1):
                MuBS = M[s][:,:,None]*uBS[s]
                S[s][w] -= xp.einsum('wir,wrj->wij',SCu[s],MuBS)
            self.Sa,self.Sb = S[0],S[1]
            return
        uBS = xp.concatenate(uBS,axis=1)
        MuBS = M[:,:,None]*uBS
        SCu = xp.concatenate(SCu,axis=2)
        self.S[w] -= xp.einsum('wir,wrj->wij',SCu,MuBS)

    def compute_SC(self):
        phi = self.get_phi()
        if self.S is None:
            S = [Sa,Sb]
        else:
            S = [self.S,self.S]
        return [compute_SC(S[s],phi[s],s) for s in (0,1)]

    def reortho(self):
        phi,UC = self.get_phi(UC=True)
        for s in (0,1):
            phi[s],UC[s] = qr(phi[s],UC[s])
        self.set_phi(phi,UC)

    def save(self,comm,dirname):
        RANK,SIZE = comm.rank,comm.size
        if RANK>0:
            obj = to_host(self.phi),to_host(self.weight)
            comm.send(obj,0)
            return
        phi = [to_host(self.phi)] + ([None] * (SIZE-1))
        weights = [to_host(self.weight)] + ([None] * (SIZE-1))
        for r in range(1,SIZE):
            phi[r],weights[r] = comm.recv(source=r)
        with h5py.File(f'{dirname}/walkers.hdf5','w') as f:
            f.create_dataset('phi',data=np.concatenate(phi,axis=0))
            f.create_dataset('weights',data=np.concatenate(weights,axis=0))
    
    def load_walkers(self,comm,dirname):
        with h5py.File(f'{dirname}/walkers.hdf5','r') as f:
            phi = f['phi'][:]
            weights = f['weights'][:]
        RANK,SIZE = comm.rank,comm.size
    
        nw = weights.size
        b,r = nw//SIZE,nw%SIZE
        counts = np.array([b]*SIZE)
        if r>0:
            counts[:r] += 1
        counts = np.cumsum(counts)
        start = 0 if RANK==0 else counts[RANK-1]
        stop = counts[RANK]
        print(f'RANK={RANK},start={start},stop={stop}')
        self.phi = xp.asarray(phi[start:stop])
        self.weight = xp.asarray(weights[start:stop])
        #_check_nan(walkers.phi,'phi','loaded')
        #nu = walkers.nup
        #phi = walkers.phi[:,:,:nu]
        #ovlp = xp.einsum('wxi,wxj->wij',phi,phi)
        #print(np.linalg.norm(ovlp-xp.eye(nu)[None,:,:]))
        #phi = walkers.phi[:,:,nu:]
        #ovlp = xp.einsum('wxi,wxj->wij',phi,phi)
        #print(np.linalg.norm(ovlp-xp.eye(nu)[None,:,:]))
        #exit()
    
