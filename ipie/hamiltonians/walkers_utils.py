import numpy as np
import scipy,itertools,plum,h5py
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from mpi4py import MPI
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

@plum.dispatch
def walkers2tensor(walkers:UHFWalkers):
    return xp.concatenate((walkers.phia.real,walkers.phib.real),axis=2)

@plum.dispatch
def walkers2tensor(walkers:GHFWalkers):
    return walkers.phi.real

@plum.dispatch
def tensor2walkers(walkers:UHFWalkers,phi):
    phi = xp.asarray(phi)
    walkers.phia = phi[0]
    walkers.phib = phi[1]
    return walkers

@plum.dispatch
def tensor2walkers(walkers:GHFWalkers,phi):
    walkers.phi = xp.asarray(phi)
    return walkers 

def save_walkers(walkers,comm,dirname):
    if comm.rank>0:
        obj = to_host(walkers2tensor(walkers)),to_host(walkers.weight),to_host(walkers.sgn_ovlp)
        comm.send(obj,0)
        return
    phi = [to_host(walkers2tensor(walkers))] + ([None] * (SIZE-1))
    weights = [to_host(walkers.weight)] + ([None] * (SIZE-1))
    sgn_ovlp = [to_host(walkers.sgn_ovlp)] + ([None] * (SIZE-1))
    for r in range(1,SIZE):
        phi[r],weights[r],sgn_ovlp[r] = comm.recv(source=r)
    f = h5py.File(f'{dirname}/walkers.hdf5','w')
    f.create_dataset('phi',data=np.concatenate(phi,axis=0))
    f.create_dataset('log_weights',data=np.concatenate(weights,axis=0))
    f.create_dataset('sgn_ovlp',data=np.concatenate(sgn_ovlp,axis=0))
    f.close()

def load_walkers(walkers,dirname):
    f = h5py.File(f'{dirname}/walkers.hdf5','r')
    phi = f['phi'][:]
    log_weights = f['log_weights'][:]
    sgn_ovlp = f['sgn_ovlp'][:]
    f.close()
    print(phi.shape)
    print(log_weights.shape)
    print(sgn_ovlp.shape)
    exit()

    nw = log_weights.size
    b,r = nw//SIZE,nw%SIZE
    counts = np.array([b]*SIZE)
    if r>0:
        counts[:r] += 1
    counts = np.cumsum(counts)
    start = 0 if RANK==0 else counts[RANK-1]
    stop = counts[RANK]
    print(f'RANK={RANK},start={start},stop={stop}')
    walkers = tensor2walkers(walkers,phi[start:stop])
    walkers.weight = xp.exp(xp.asarray(log_weights[start:stop]))
    walkers.sgn_ovlp = xp.asarray(sgn_ovlp[start:stop])
    return walkers

