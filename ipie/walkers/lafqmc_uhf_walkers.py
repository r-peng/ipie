import numpy as np
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp
from ipie.walkers.base_walkers import BaseWalkers 

def qr(phi,thresh=1e-3):
    Q,R = xp.linalg.qr(phi,mode='reduced')
    Rdiag = xp.einsum('wii->wi',R)

    Rabs = xp.fabs(Rdiag)
    assert Rabs[Rabs<thresh].size==0

    sign = xp.sign(Rdiag)
    Q *= sign[:,None,:]
    return Q

def det_ovlp(C,B=None):
    if C is None:
        return None
    if B is None:
        return xp.einsum('wxi,wxj->wij',C,C)
    return xp.einsum('wxi,xj->wij',C,B)

def matrix_inverse(ovlp):
    if ovlp is None:
        return None
    return xp.linalg.inv(ovlp)

def compute_SC(S,C):
    if C is None:
        return None
    return xp.einsum('wij,wxj->wix',S,C) 

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

    def get_phi(self):
        if self.ndown==0:
            phi = [self.phi,None]
        else:
            phi = [self.phi[:,:,:self.nup],self.phi[:,:,self.nup:]]
        return phi

    @plum.dispatch
    def build_trial(self,trial:SingleDet,set_buffnames=False):
        phi = self.get_phi()
        CB = [det_ovlp(Ci,Bi) for Ci,Bi in zip(phi,trial.psi)] 
        self.Sa,self.Sb = [matrix_inverse(Si) for Si in CB]
        self.S = None
        if set_buffnames:
            self.buff_names += ['Sa','Sb']
    
    @plum.dispatch
    def build_trial(self,trial:SingleDetGHF,set_buffnames=False):
        phi = self.get_phi()
        B = [trial.psi[:self.nbasis],trial.psi[self.nbasis:]]
        CB = [det_ovlp(Ci,Bi) for Ci,Bi in zip(phi,B)] 
        if CB[1] is None:
            CB = CB[0]
        else:
            CB = xp.concatenate(CB,axis=1)
        self.S = xp.linalg.inv(CB)
        if set_buffnames:
            self.buff_names += ['S']

    def build_hamiltonian(self,hamiltonian):
        self.UC = xp.einsum('dxp,wxi->wdpi',hamiltonian.chol_basis,self.phi)

    def compute_ovlp_ratio(self,hamiltonian,trial):
        b = xp.zeros(hamiltonian.nterms)
        self.Cua = dict()
        self.Cub = dict()
        uB = dict()
        M1_dict = dict()
        for key,p in hamiltonian.p_dict.items():
            d = hamiltonian.d_dict[key]
            kix = hamiltonian.kix_dict[key]
            chol_idx,typ = key

            Cua = self.UC[:,chol_idx,:,:self.nup].transpose(0,2,1)
            Cub = self.UC[:,chol_idx,:,self.nup:].transpose(0,2,1)
            if typ=='h2ab':
                Cua = xp.asarray([Cua[:,:,pi[:1]] for pi in p])
                Cub = xp.asarray([Cua[:,:,pi[1:]] for pi in p]) 
            elif typ[-1]=='a':
                Cua = xp.asarray([Cua[:,:,pi] for pi in p])
                Cub = None
            else:
                Cub = xp.asarray([Cua[:,:,pi] for pi in p])
                Cua = None
            self.Cua[key] = Cua
            self.Cub[key] = Cub 

            


    def compute_SC(self):
        phi = self.get_phi()
        if self.S is None:
            S = [self.Sa,self.Sb]
        else:
            S = [self.S[:,:,:self.nup],self.S[:,:,self.nu:]]
        return [_SC(Si,Ci) for Si,Ci in zip(S,phi)]

    def reortho(self):
        if self.ndown>0:
            self..phi[:,:,:self.nup] = qr(self.phi[:,:,:self.nup])
            self..phi[:,:,self.nup:] = qr(self.phi[:,:,self.nup:])
        else:
            self.phi = qr(self.phi)

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
    
