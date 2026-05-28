import numpy as np
import plum,h5py
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers

def make_full(Daa,Dbb,Dab=None,Dba=None):
    nw,n1,n2 = Daa.shape 
    D = xp.zeros((nw,n1*2,n2*2))
    D[:,:n1,:n2] = Daa
    if Dbb is not None:
        D[:,n1:,n2:] = Dbb
    if Dab is not None:
        D[:,:n1,n2:] = Dab
    if Dba is not None:
        D[:,n1:,:n2] = Dba
    return D

def _conjugate_chol(U,D,axis):
    if D is None:
        return None
    if U is None:
        return D
    if axis==1:
        return xp.einsum('xp,wxy->wpy',U,D)
    elif axis==2:
        return xp.einsum('wxy,yq->wxq',D,U)
    else:
        raise ValueError

def conjugate_chol(U,D,axis,full=False):
    if U is None:
        if isinstance(D,list) and full:
            return make_full(D[0],D[1])
        return D
    if isinstance(D,list):
        D = [_conjugate_chol(U,Di,axis) for Di in D]
        if full:
            D = make_full(D[0],D[1])
        return D
    nb,_ = U.shape
    if axis==1:
        D = D[:,:nb],D[:,nb:]
    elif axis==2:
        D = D[:,:,:nb],D[:,:,nb:]
    else:
        raise ValueError
    D = [_conjugate_chol(U,Di,axis) for Di in D]
    return xp.concatenate(D,axis=axis)

@plum.dispatch
def walkers2tensor(walkers:UHFWalkers):
    if walkers.ndown==0:
        return walkers.phia
    return xp.concatenate((walkers.phia.real,walkers.phib.real),axis=2)

@plum.dispatch
def walkers2tensor(walkers:GHFWalkers):
    return walkers.phi.real

@plum.dispatch
def tensor2walkers(walkers:UHFWalkers,phi):
    phi = xp.asarray(phi)
    walkers.phia = phi[:,:,:walkers.nup]
    walkers.phib = None
    if walkers.ndown>0:
        walkers.phib = phi[:,:,walkers.nup:]
    return walkers

@plum.dispatch
def tensor2walkers(walkers:GHFWalkers,phi):
    walkers.phi = xp.asarray(phi)
    return walkers 

def save_walkers(walkers,comm,dirname):
    RANK,SIZE = comm.rank,comm.size
    if RANK>0:
        obj = to_host(walkers2tensor(walkers)),to_host(walkers.weight),to_host(walkers.sgn_ovlp)
        comm.send(obj,0)
        return
    phi = [to_host(walkers2tensor(walkers))] + ([None] * (SIZE-1))
    weights = [to_host(walkers.weight)] + ([None] * (SIZE-1))
    sgn_ovlp = [to_host(walkers.sgn_ovlp)] + ([None] * (SIZE-1))
    for r in range(1,SIZE):
        phi[r],weights[r],sgn_ovlp[r] = comm.recv(source=r)
    with h5py.File(f'{dirname}/walkers.hdf5','w') as f:
        f.create_dataset('phi',data=np.concatenate(phi,axis=0))
        f.create_dataset('log_weights',data=np.concatenate(weights,axis=0))
        f.create_dataset('sgn_ovlp',data=np.concatenate(sgn_ovlp,axis=0))

def load_walkers(walkers,comm,dirname):
    with h5py.File(f'{dirname}/walkers.hdf5','r') as f:
        phi = f['phi'][:]
        log_weights = f['log_weights'][:]
        sgn_ovlp = f['sgn_ovlp'][:]
    print(phi.shape)
    print(log_weights.shape)
    print(sgn_ovlp.shape)
    exit()

    RANK,SIZE = comm.rank,comm.size

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

