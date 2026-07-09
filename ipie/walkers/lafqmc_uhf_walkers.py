import numpy as np
import plum,h5py
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.walkers.base_walkers import BaseWalkers 
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

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

def get_uBS_left(uBS,ne,s):
    if uBS.shape[-1]==ne:
        return uBS
    if s==0:
        return uBS[:,:,:,:ne]
    assert s==1
    if ne==0:
        return None 
    return uBS[:,:,:,-ne:]

def uBS_dot_Cu(uBS,uC,s,scalar=False):
    uBS = get_uBS_left(uBS,uC.shape[-1],s)
    if uBS is None:
        return 0
    uDu = xp.einsum('kwrj,kwsj->kwrs',uBS,uC)
    if scalar:
        uDu = uDu[:,:,0,0]
    return uDu

def UBS_dot_Cu(UBS,uC,s,scalar=False):
    UBS = get_uBS_left(UBS,uC.shape[-1],s)
    if UBS is None:
        return 0
    left = xp.einsum('wdpi,wri->wdpr',UBS,uC)
    if scalar:
        left = left[:,:,:,0]
    return left

def CU_dot_UBS(UBS,UC,s,tr=True,ixs=None):
    UBS = get_uBS_left(UBS,UC.shape[-1],s)
    if UBS is None:
        return 0
    if ixs is not None:
        UC = UC[:,ixs]
        UBS = UBS[:,ixs]
    if tr:
        return xp.einsum('wdpi,wdpi->wd',UC,UBS)
    else:
        return xp.einsum('wdpi,wdpj->wdij',UC,UBS)

def update_phi(C,u,d,uC):
    duC = d[:,:,None]*uC
    C += xp.einsum('wxr,wri->wxi',u,duC)
    return C,duC

def update_UC(UC,u2,duC):
    UC += xp.einsum('dwpr,wri->wdpi',u2,duC)
    return UC

def update_UBS(UBS,left,uBS):
    UBS -= xp.einsum('wdpr,wri->wdpi',left,uBS)
    return UBS

def compute_1rdm_diag(BS,C):
    return xp.einsum('wpi,wpi->wp',BS,C)

def compute_lowdin(d,uC,thresh=1e-3):
    Cu = uC.transpose(0,2,1)
    q,s = xp.linalg.qr(Cu,mode='reduced')
    delta = np.einsum('wrs,ws,wts->wrt',s,d**2+2*d,s)
    delta,v = xp.linalg.eigh(delta)
    q = xp.einsum('wir,wrs->wis',q,v)
    delta += 1.
    if delta[delta<thresh.size]>0:
        print('delta=',delta.T)
        print('s=',s.T)
        raise ValueError

    delta = xp.sqrt(delta)
    return q,delta

def lowdin_phi(C,q,delta):
    left = xp.einsum('wxi,wir->wxr',C,q)
    right = (1./delta-1.)[:,None,:]*q
    C += xp.einsum('wxr,wir->wxi',left,right)
    return C

def lowdin_UC(C,q,delta):
    left = xp.einsum('wdxi,wir->wdxr',C,q)
    right = (1./delta-1.)[:,None,:]*q
    C += xp.einsum('wdxr,wir->wdxi',left,right)
    return C

class UHFWalkers(BaseWalkers):

    def __init__(
        self,
        initial_walker: np.ndarray,
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

        self.phase = xp.ones(self.nwalkers) 

    def get_phi(self):
        nu = self.nup
        return [self.phi[:,:,:nu],self.phi[:,:,nu:]]

    def set_phi(self,phi):
        nu = self.nup
        self.phi[:,:,:nu] = phi[0]
        self.phi[:,:,nu:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDet):
        phi = self.get_phi()
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,trial.psi)] 
        return [xp.linalg.inv(Si) for Si in CB]
    
    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF):
        phi = self.get_phi()
        B = [trial.psi[:self.nbasis],trial.psi[self.nbasis:]]
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,B)] 
        CB = xp.concatenate(CB,axis=1)
        return xp.linalg.inv(CB)

    def compute_UC(self,hamiltonian):
        U = hamiltonian.chol_basis
        self.UC = xp.einsum('dxp,wxi->wdpi',U,self.phi)

    def build(self,hamiltonian,trial):
        self.compute_UC(hamiltonian)
        UB = trial.compute_UB(hamiltonian)
        S = self.compute_S(trial)
        if isinstance(S,list):
            UBS = [xp.einsum('dpi,wij->wdpj',UBi,Si) for UBi,Si in zip(UB,S)]
            self.UBS = xp.concatenate(UBS,axis=3)
        else:
            self.UBS = xp.einsum('dpi,wij->wdpj',UB,S)

        self.buff_names = ['phi','weight','phase','UC','UBS']
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = xp.zeros(self.buff_size)

    def get_UC(self):
        nu = self.nup
        return [self.UC[:,:,:,:nu],self.UC[:,:,:,nu:]]

    def set_UC(self,UC):
        nu = self.nup
        self.UC[:,:,:,:nu] = UC[0]
        self.UC[:,:,:,nu:] = UC[1]

    def get_UBS(self):
        nb = self.nbasis
        if self.UBS.shape[2]==nb:
            nu = self.nup
            return [self.UBS[:,:,:,:nu],self.UBS[:,:,:,nu:]]
        else:
            return [self.UBS[:,:,:nb],self.UBS[:,:,nb:]]

    def set_UBS(self,UBS):
        nb = self.nbasis
        if self.UBS.shape[2]==nb:
            nu = self.nup
            self.UBS[:,:,:,:nu] = UBS[0]
            self.UBS[:,:,:,nu:] = UBS[1]
        else:
            self.UBS[:,:,:nb] = UBS[0]
            self.UBS[:,:,nb:] = UBS[1]

    def get_walkers_component(self,key,p,typ):
        chol_idx,spin = key
        if typ=='UC':
            UX = self.get_UC()
        elif typ=='UBS':
            UX = self.get_UBS()
        else:
            raise ValueError
        uX = [UXi[:,chol_idx] for UXi in UX]

        if spin==(0,1):
            p = [p[:,:1],p[:,1:]]
            uX = [xp.asarray([uX[s][:,pi] for pi in p[s]]) for s in (0,1)]
        else:
            s = spin[0]
            uX = xp.asarray([uX[s][:,pi] for pi in p])
        return uX

    def compute_M(self,key,d):
        _,spin = key
        uBS = self.uBS[key]
        uC = self.uC[key]
        if spin==(0,1):
            nk,nw,_,_ = uC[0].shape
            M = xp.zeros((nk,nw,2,2))
            for s in (0,1):
                M[:,:,s,s] = uBS_dot_Cu(uBS[s],uC[s],s,scalar=True)
            if uBS[0].shape[-1]>self.nup:
                M[:,:,0,1] = uBS_dot_Cu(uBS[0],uC[1],1,scalar=True)
            if uBS[1].shape[-1]>self.nup:
                M[:,:,1,0] = uBS_dot_Cu(uBS[1],uC[0],0,scalar=True)
        else:
            s = spin[0]
            M = uBS_dot_Cu(uBS,uC,s)
        M = d[:,None,:,None]*M + xp.eye(d.shape[1])[None,None,:,:]
        ovlp = xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,None,:]
        self.M2[key] = M
        return ovlp

    def compute_ovlp_ratio(self,hamiltonian,trial):
        ovlp = xp.zeros((hamiltonian.nterms,self.nwalkers))
        self.uC = dict()
        self.uBS = dict()
        self.M2 = dict()
        keys = 'p','d','ix'
        for key,dat in hamiltonian.term_dict.items():
            p,d,ix = [dat[k] for k in keys]

            self.uC[key] = self.get_walkers_component(key,p,'UC')
            self.uBS[key] = self.get_walkers_component(key,p,'UBS')
            ovlp[ix] = self.compute_M(key,d)
        return ovlp

    def update_walkers(self,samples):
        keys = 'w','i','u','d','u2'
        for key in samples:
            dat = samples[key]
            w,i,u,d,u2 = [dat[k] for k in keys]

            uC,duC = self.update_phi(key,w,i,u,d)
            self.update_UC(key,w,u2,duC)
            self.update_UBS(key,w,i,uC) 

            samples[key] = {'w':w,'d':d,'uC':uC}
        return samples

    def update_phi(self,key,w,i,u,d):
        _,spin = key
        uC = self.uC[key]
        phi = self.get_phi()
        if spin==(0,1):
            d = [d[:,:1],d[:,1:]]
            u = [u[:,:,:1],u[:,:,1:]]
            uC = [uCi[i,w] for uCi in uC]
            duC = [None] * 2
            for s in (0,1):
                phi[s][w],duC[s] = update_phi(phi[s][w],u[s],d[s],uC[s])
        else:
            s = spin[0]
            uC = uC[i,w]
            phi[s][w],duC = update_phi(phi[s][w],u,d,uC)
        self.set_phi(phi)
        return uC,duC

    def update_UC(self,key,w,u2,duC):
        _,spin = key
        UC = self.get_UC()
        if spin==(0,1):
            u2 = [u2[:,:,:,:1],u2[:,:,:,1:]]
            for s in (0,1):
                UC[s][w] = update_UC(UC[s][w],u2[s],duC[s])
        else:
            s = spin[0]
            UC[s][w] = update_UC(UC[s][w],u2,duC)
        self.set_UC(UC)

    def update_UBS_1(self,key,w,i,uC):
        _,spin = key
        s = spin[0]

        M = self.M2[key][i,w]
        MuC = xp.einsum('wri,wrs->wsi',uC,M)
        uBS = self.uBS[key]
        if self.UBS.shape[2]==self.nbasis:
            UBS = self.get_UBS()
            left = UBS_dot_Cu(UBS[s][w],MuC,s)
            UBS[s][w] = update_UBS(UBS[s][w],left,uBS[i,w])
            self.set_UBS(UBS)
        else:
            left = UBS_dot_Cu(self.UBS[w],MuC,s)
            self.UBS[w] = update_UBS(self.UBS[w],left,uBS[i,w])

    def update_UBS_2(self,key,w,i,uC):
        M = self.M2[key][i,w]
        M = M[:,:1],M[:,1:]
        MuC = [xp.einsum('wri,wrs->wsi',uC[s],M[s]) for s in (0,1)]

        uBS = self.uBS[key]
        uBS = [uBSi[i,w] for uBSi in uBS]
        if self.UBS.shape[2]==self.nbasis:
            UBS = self.get_UBS()
            for s in (0,1):
                left = UBS_dot_Cu(UBS[s][w],MuC[s],s)
                UBS[s][w] = update_UBS(UBS[s][w],left,uBS[s])
            self.set_UBS(UBS)
        else:
            MuC = np.concatenate(MuC,axis=2)
            left = UBS_dot_Cu(self.UBS[w],MuC,None)
            uBS = xp.concatenate(uBS,axis=1)
            self.UBS[w] = update_UBS(self.UBS[w],left,uBS)

    def update_UBS(self,key,w,i,uC):
        _,spin = key
        if spin==(0,1):
            self.update_UBS_2(key,w,i,uC)
        else:
            self.update_UBS_1(key,w,i,uC)

    def compute_lowdin(self,samples):
        keys = 'w','d','uC'
        for key in samples.items():
            dat = samples[key]
            w,d,uC = [dat[k] for k in keys]
            q,delta = self._compute_lowdin(key,d,uC)
            samples[key] = {'w':w,'q':q,'delta':delta}
        return samples

    def _compute_lowdin(self,key,d,uC):
        _,spin = key
        if spin==(0,1):
            q = [None] * 2
            delta = [None] * 2
            d = [d[:,:1],d[:,1:]]
            for s in (0,1):
                q[s],delta[s] = compute_lowdin(d[s],uC[s])
        else:
            q,delta = compute_lowdin(d,uC) 
        return q,delta

    def lowdin(self,samples):
        keys = 'w','q','delta'
        for key,dat in samples.items():
            w,q,delta = [dat[k] for k in keys]
            self.lowdin_phi(key,w,q,delta)

    def lowdin_phi(self,key,w,q,delta):
        _,spin = key
        phi = self.get_phi()
        if spin==(0,1):
            for s in (0,1):
                phi[s][w] = lowdin_phi(phi[s][w],q[s],delta[s])
        else:
            s = spin[0]
            phi[s][w] = lowdin_phi(phi[s][w],q,delta)
        self.set_phi(phi)

    def compute_1rdm_diag(self,chol_ix):
        phi = self.get_phi()
        BS = [UBSi[:,chol_ix] for UBSi in self.get_UBS()]
        D = dict()
        if BS[0].shape[-1]==phi[0].shape[-1]:
            D[0,0] = compute_1rdm_diag(BS[0],phi[0])
            D[1,1] = compute_1rdm_diag(BS[1],phi[1])
            return D
        nu = self.nup
        D[0,0] = compute_1rdm_diag(BS[0][:,:,:,:nu],phi[0])
        D[1,1] = compute_1rdm_diag(BS[1][:,:,:,nu:],phi[1])
        D[0,1] = compute_1rdm_diag(BS[0][:,:,:,nu:],phi[1])
        D[1,0] = compute_1rdm_diag(BS[1][:,:,:,:nu],phi[0])
        return D

    def compute_local_energy_intermediates(self,bands,tr_ixs=None,mat_ixs=None):
        UBS = [bands[None,:,:,None]*UBSi for UBSi in self.get_UBS()]
        UC = self.get_UC()

        tr = [CU_dot_UBS(UBS[s],UC[s],s,tr=True,ixs=tr_ixs) for s in (0,1)] 
        if mat_ixs is None:
            return tr
        mat = [CU_dot_UBS(UBS[s],UC[s],s,tr=False,ixs=mat_ixs) for s in (0,1)] 
        return tr,mat 

    def reortho(self):
        phi = self.get_phi()
        UC = self.get_UC()
        for s in (0,1):
            phi[s],UC[s] = qr(phi[s],UC[s])
        self.set_phi(phi)
        self.set_UC(UC)

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
    
    def reortho_batched(self):
        pass


