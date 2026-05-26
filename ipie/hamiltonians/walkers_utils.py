import numpy as np
import plum,h5py
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers

@plum.dispatch
def walkers2uhf(walkers:UHFWalkers):
    print(walkers.phia)
    print(walkers.phib)
    return xp.stack((walkers.phia.real,walkers.phib.real))

@plum.dispatch
def walkers2ghf(walkers:UHFWalkers):
    nw,nb = walkers.nwalkers,walkers.nbasis
    nu,nd = walkers.nup,walkers.ndown
    C = xp.zeros((nw,nb*2,nu+nd))
    C[:,:nb,:nu] = walkers.phia.real
    C[:,nb:,nu:] = walkers.phib.real
    return C 

@plum.dispatch
def walkers2ghf(walkers:GHFWalkers):
    return walkers.phi.real

def make_full(Daa,Dbb,Dab=None,Dba=None):
    nw,n1,n2 = Daa.shape 
    D = xp.zeros((nw,n1*2,n2*2))
    D[:,:n1,:n2] = Daa
    D[:,n1:,n2:] = Dbb
    if Dab is not None:
        D[:,:n1,n2:] = Dab
    if Dba is not None:
        D[:,n1:,:n2] = Dba
    return D

def conjugate_chol_left(U,D,full=False):
    if U is None:
        if len(D.shape)==4 and full:
            return make_full(D[0],D[1])
        return D

    if len(D.shape)==4:
        UD = xp.einsum('xp,swxy->swpy',U,D)
        if full:
            UD = make_full(UD[0],UD[1])
        return UD
    nw,_,sh2 = D.shape
    nb,_ = U.shape
    UD = xp.zeros((nw,nb*2,sh2))
    UD[:,:nb] = xp.einsum('xp,wxy->wpy',U,D[:,:nb])
    UD[:,nb:] = xp.einsum('xp,wxy->wpy',U,D[:,nb:])
    return UD

def conjugate_chol_right(U,D,full=False):
    if U is None:
        if len(D.shape)==4 and full:
            return make_full(D[0],D[1])
        return D

    if len(D.shape)==4:
        DU = xp.einsum('swxy,yq->swxq',D,U)
        if full:
            DU = make_full(DU[0],DU[1])
        return DU
    nw,sh1,_ = D.shape
    nb,_ = U.shape
    DU = xp.zeros((nw,sh1,nb*2))
    DU[:,:,:nb] = xp.einsum('wxy,yq->wxq',D[:,:,:nb],U)
    DU[:,:,nb:] = xp.einsum('wxy,yq->wxq',D[:,:,nb:],U)
    return DU

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

