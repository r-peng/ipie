import numpy as np
import plum,h5py
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

def _preprocess_walkers(walkers):
    walkers.buff_names += ['weight','phase','ovlp']
    walkers.phase = xp.ones(walkers.nwalkers) 
    walkers.ovlp = xp.ones(walkers.nwalkers) 
    walkers.R = None 
    walkers.sgn_ovlp = None
    walkers.eloc = None
    walkers.buff_size = round(walkers.set_buff_size_single_walker() / float(walkers.nwalkers))
    walkers.walker_buffer = xp.zeros(walkers.buff_size)

@plum.dispatch
def preprocess_walkers(walkers:UHFWalkers):
    walkers.buff_names = ['phia','phib']
    walkers.phia = walkers.phia.real
    if walkers.phib is not None:
        walkers.phib = walkers.phib.real
    walkers.G = None
    walkers.Ga = None
    walkers.Gb = None
    walkers.Ghalfa = None
    walkers.Ghalfb = None
    _preprocess_walkers(walkers)

@plum.dispatch
def preprocess_walkers(walkers:GHFWalkers):
    walkers.buff_names = ['phi']
    walkers.phi = walkers.phi.real
    walkers.G = None
    _preprocess_walkers(walkers)

def _preprocess_trial(trial):
    trial.psi0 = None
    trial.psi0a = None
    trial.psi0b = None
    trial.G = None
    trial.Ghalf = None

@plum.dispatch
def preprocess_trial(trial:SingleDet):
    trial.psi = [trial.psi0a.real,trial.psi0b.real]
    _preprocess_trial(trial)

@plum.dispatch
def preprocess_trial(trial:SingleDetGHF):
    trial.psi = trial.psi0.real
    _preprocess_trial(trial)

@plum.dispatch
def parse_phi(walkers:UHFWalkers):
    if walkers.ndown==0:
        return walkers.phia
    return xp.concatenate((walkers.phia,walkers.phib),axis=2)

@plum.dispatch
def parse_phi(walkers:GHFWalkers):
    return walkers.phi

@plum.dispatch
def load_phi(walkers:UHFWalkers,phi):
    phi = xp.asarray(phi)
    walkers.phia = phi[:,:,:walkers.nup]
    walkers.phib = None
    if walkers.ndown>0:
        walkers.phib = phi[:,:,walkers.nup:]
    return walkers

@plum.dispatch
def load_phi(walkers:GHFWalkers,phi):
    walkers.phi = xp.asarray(phi)
    return walkers 

def save_walkers(walkers,comm,dirname):
    RANK,SIZE = comm.rank,comm.size
    if RANK>0:
        obj = to_host(parse_phi(walkers)),to_host(walkers.weight)
        comm.send(obj,0)
        return
    phi = [to_host(parse_phi(walkers))] + ([None] * (SIZE-1))
    weights = [to_host(walkers.weight)] + ([None] * (SIZE-1))
    for r in range(1,SIZE):
        phi[r],weights[r] = comm.recv(source=r)
    with h5py.File(f'{dirname}/walkers.hdf5','w') as f:
        f.create_dataset('phi',data=np.concatenate(phi,axis=0))
        f.create_dataset('weights',data=np.concatenate(weights,axis=0))

def load_walkers(walkers,comm,dirname):
    with h5py.File(f'{dirname}/walkers.hdf5','r') as f:
        phi = f['phi'][:]
        weights = f['weights'][:]
    print(phi.shape)
    print(weights.shape)
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
    walkers = load_phi(walkers,phi[start:stop])
    walkers.weight = xp.asarray(weights[start:stop])
    return walkers

def _ovlp(C,B=None):
    if C is None:
        return None
    if B is None:
        return xp.einsum('wxi,wxj->wij',C,C)
    return xp.einsum('wxi,xj->wij',C,B)

def _inv(ovlp):
    if ovlp is None:
        return None
    return xp.linalg.inv(ovlp)

#@plum.dispatch
#def compute_ovlp(walkers:UHFWalkers,inv=True):
#    C = walkers.phia,walkers.phib
#    CdC = [_ovlp(Ci) for Ci in C]
#    if not inv:
#        return CdC
#    return [_inv(oi) for oi in CdC]

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True,scalar_ovlp=False):
    C = walkers.phia.real,walkers.phib.real
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,trial.psi)] 
    if scalar_ovlp:
        walkers.ovlp = xp.linalg.det(CdB[0])
        if CdB[1] is not None:
            walkers.ovlp *= xp.linalg.det(CdB[1])
    if inv:
        CdB = [_inv(oi) for oi in CdB]
    return CdB

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True,scalar_ovlp=False):
    C = walkers.phia,walkers.phib
    nb = trial.nbasis
    B = trial.psi[:nb],trial.psi[nb:]
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,B)] 
    CdB = xp.concatenate(CdB,axis=1)
    if scalar_ovlp:
        walkers.ovlp = xp.linalg.det(CdB)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

@plum.dispatch
def compute_ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True,scalar_ovlp=False):
    CdB = _ovlp(walkers.phi,B=trial.psi)
    if scalar_ovlp:
        walkers.ovlp = xp.linalg.det(CdB)
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB

def _rdm1(ovlp,C,B=None):
    if C is None:
        return None
    if B is None:
        D = xp.einsum('wxi,wij->wxj',C,ovlp) 
    else:
        D = xp.einsum('xi,wij->wxj',B,ovlp) 
    return xp.einsum('wxj,wyj->wxy',D,C) 

#@plum.dispatch
#def compute_rdm1(walkers:UHFWalkers):
#    CdCinv = compute_ovlp(walkers)
#    C = walkers.phia,walkers.phib
#    walkers.D = [_rdm1(oi,Ci) for oi,Ci in zip(CdCinv,C)] 

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDet,scalar_ovlp=False,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp)
    C = walkers.phia,walkers.phib
    walkers.D = [_rdm1(oi,Ci,B=Bi) for oi,Ci,Bi in zip(CdBinv,C,trial.psi)] 
    compute_regularization(walkers,eps_sq=eps_sq)

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDetGHF,scalar_ovlp=False,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp)
    tmp = xp.einsum('xi,wij->wxj',trial.psi,CdBinv) 

    nw = walkers.nwalkers
    nu,nd = trial.nelec
    nb = trial.nbasis
    D = xp.zeros((nw,nb*2,nb*2))
    D[:,:,:nb] = xp.einsum('wxj,wyj->wxy',tmp[:,:,:nu],walkers.phia)
    D[:,:,nb:] = xp.einsum('wxj,wyj->wxy',tmp[:,:,nu:],walkers.phib)
    walkers.D = D
    compute_regularization(walkers,eps_sq=eps_sq)

@plum.dispatch
def compute_rdm1(walkers:GHFWalkers,trial:SingleDetGHF,scalar_ovlp=False,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp)
    walkers.D = _rdm1(CdBinv,walkers.phi.real,B=trial.psi)
    compute_regularization(walkers,eps_sq=eps_sq)

def _trace(D):
    if D is None:
        return 0
    return xp.einsum('wxy->w',D**2)

def compute_trace(D):
    if isinstance(D,list):
        return  _trace(D[0])+_trace(D[1]) 
    return _trace(D)

def compute_regularization(walkers,eps_sq=None):
    if eps_sq is None:
        return 
    tr = compute_trace(walkers.D)
    walkers.R = 1./xp.sqrt(1.+eps_sq*tr)

def _update_walkers(C,d,u,w):
    if C is None:
        return C
    if d is None:
        return C
    right = xp.einsum('wr,wxr,wxi->wri',d,u,C[w])
    C[w] += xp.einsum('wxr,wri->wxi',u,right)
    return C

@plum.dispatch
def update_walkers(walkers:UHFWalkers,dmap,umap,wmap):
    phi = [walkers.phia,walkers.phib]
    for (s,r),d in dmap.items():
        phi[s] = _update_walkers(phi[s],d,umap[s,r],wmap[s,r])
    walkers.phia = phi[0]
    walkers.phib = phi[1]

@plum.dispatch
def update_walkers(walkers:GHFWalkers,dmap,umap,wmap):
    nb = walkers.nbasis
    phi = [walkers.phi[:,:nb],walkers.phi[:,nb:]]
    for (s,r),d in dmap.items():
        phi[s] = _update_walkers(phi[s],d,umap[s,r],wmap[s,r])
    walkers.phi = xp.concatenate(phi,axis=1)

@plum.dispatch
def update_walkers_slow(walkers:UHFWalkers,Us):
    phia = walkers.phia.real.copy()
    phib = None
    if walkers.phib is not None:
        phib = walkers.phib.real.copy()
    for w,U in enumerate(Us):
        if U[0] is not None:
            phia[w] = xp.dot(U[0],phia[w])
        if phib is not None:
            if U[1] is not None:
                phib[w] = xp.dot(U[1],phib[w])
    return phia,phib

@plum.dispatch
def update_walkers_slow(walkers:GHFWalkers,Us):
    phi = walkers.phi.real.copy()
    nb = walkers.nbasis
    for w,U in enumerate(Us):
        if U[0] is not None:
            phi[w,:nb] = xp.dot(U[0],phi[w,:nb])
        if U[1] is not None:
            phi[w,nb:] = xp.dot(U[1],phi[w,nb:])
    return phi

def _get_Dr(D,nb,s):
    if isinstance(D,list):
        return D[s]
    if s==0:
        return D[:,:nb]
    else:
        return D[:,nb:]

def _get_Dl(D,nb,s):
    if isinstance(D,list):
        return D[s]
    if s==0:
        return D[:,:,:nb]
    else:
        return D[:,:,nb:]

def _update_rdm1_ovlp_sd(D,b,d,u,w,s):
    if d is None:
        return D,b
    _,nb,r = u.shape
    idx = xp.arange(r)
    if isinstance(D,list):
        if D[s] is None:
            return D,b
        Dw = D[s][w]
        Du = xp.einsum('wxy,wyr->wxr',Dw,u)
        uD = xp.einsum('wxr,wxy->wry',u,Dw)
        uDu = xp.einsum('wxr,wxs->wrs',u,Du)
    else:
        Dw = D[w]
        if s==0:
            Du = xp.einsum('wxy,wyr->wxr',Dw[:,:,:nb],u)
            uD = xp.einsum('wxr,wxy->wry',u,Dw[:,:nb])
            uDu = xp.einsum('wxr,wxs->wrs',u,Du[:,:nb])
        else:
            Du = xp.einsum('wxy,wyr->wxr',Dw[:,:,nb:],u)
            uD = xp.einsum('wxr,wxy->wry',u,Dw[:,nb:])
            uDu = xp.einsum('wxr,wxs->wrs',u,Du[:,nb:])
    M = uDu.copy()
    M[:,idx,idx] += 1./d
    b[w] *= d.prod(axis=1)*xp.linalg.det(M)

    M = xp.linalg.inv(M)
    tmp = xp.einsum('wxr,wrs->wxs',Du,M)
    tmp = xp.einsum('wxr,wry->wxy',tmp,uD)
    if isinstance(D,list):
        D[s][w] -= tmp 
    else:
        D[w] -= tmp 

    M = xp.eye(r)[None,:,:] - xp.einsum('wrs,wsm->wrm',M,uDu) 
    M *= d[:,None,:]
    tmp = xp.einsum('wxr,wrs->wxs',Du,M)
    tmp = xp.einsum('wxr,wyr->wxy',tmp,u)
    if isinstance(D,list):
        D[s][w] += tmp 
    else:
        if s==0:
            D[w,:,:nb] += tmp
        else:
            D[w,:,nb:] += tmp
    return D,b

def update_rdm1_and_ovlp(walkers,b,dmap,umap,wmap,eps_sq=None):
    for (s,r),d in dmap.items():
        walkers.D,b = _update_rdm1_ovlp_sd(walkers.D,b,d,umap[s,r],wmap[s,r],s)
    if eps_sq is None:
        return b
    Rold = walkers.R.copy()
    walkers.R = compute_regularization(walkers.D,eps_sq=eps_sq)
    b *= Rold/walkers.R
    return b

