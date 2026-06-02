import numpy as np
import plum,h5py
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

def _preprocess_walkers(walkers):
    walkers.buff_names += ['weight','phase']
    walkers.phase = walkers.phase.real
    #walkers.ovlp = None
    #walkers.log_ovlp = None
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
    trial.psi = None
    trial.psi0 = None
    trial.psi0a = None
    trial.psi0b = None
    trial.G = None
    trial.Ghalf = None

@plum.dispatch
def preprocess_trial(trial:SingleDet):
    trial.B = [trial.psi0a.real,trial.psi0b.real]
    _preprocess_trial(trial)

@plum.dispatch
def preprocess_trial(trial:SingleDetGHF):
    trial.B = trial.psi0.real
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

def update_walkers(C,d,u,w):
    if d is None:
        return C
    right = xp.einsum('wr,wxr,wxi->wri',d,u,C[w])
    C[w] += xp.einsum('wxr,wri->wxi',u,right)
    return C

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
def compute_ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True,det=False):
    C = walkers.phia.real,walkers.phib.real
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,trial.B)] 
    if inv:
        CdB = [_inv(oi) for oi in CdB]
    if not det:
        return CdB
    det = xp.linalg.det(CdB[0])
    if CdB[1] is not None:
        det *= xp.linalg.det(CdB[1])
    return CdB,det

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True,det=False):
    C = walkers.phia,walkers.phib
    nb = trial.nbasis
    B = trial.B[:nb],trial.B[nb:]
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,B)] 
    CdB = xp.concatenate(CdB,axis=1)
    if inv:
        CdB = xp.linalg.inv(CdB)
    if not det:
        return CdB
    return CdB,xp.linalg.det(CdB)

@plum.dispatch
def compute_ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True,det=False):
    CdB = _ovlp(walkers.phi,B=trial.B)
    if inv:
        CdB = xp.linalg.inv(CdB)
    if not det:
        return CdB
    return CdB,xp.linalg.det(CdB)

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
def compute_rdm1(walkers:UHFWalkers,trial:SingleDet,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial)
    C = walkers.phia,walkers.phib
    walkers.D = [_rdm1(oi,Ci,B=Bi) for oi,Ci,Bi in zip(CdBinv,C,trial.B)] 
    walkers.R = compute_regularization(walkers.D,eps_sq=eps_sq)

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDetGHF,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial)
    tmp = xp.einsum('xi,wij->wxj',trial.B,CdBinv) 

    nw = walkers.nwalkers
    nu,nd = trial.nelec
    nb = trial.nbasis
    D = xp.zeros((nw,nb*2,nb*2))
    D[:,:,:nb] = xp.einsum('wxj,wyj->wxy',tmp[:,:,:nu],walkers.phia)
    D[:,:,nb:] = xp.einsum('wxj,wyj->wxy',tmp[:,:,nu:],walkers.phib)
    walkers.D = D
    walkers.R = compute_regularization(walkers.D,eps_sq=eps_sq)

@plum.dispatch
def compute_rdm1(walkers:GHFWalkers,trial:SingleDetGHF,eps_sq=None):
    CdBinv = compute_ovlp(walkers,trial)
    walkers.D = _rdm1(CdBinv,walkers.phi.real,B=trial.B)
    walkers.R = compute_regularization(walkers.D,eps_sq=eps_sq)

def _trace(D):
    if D is None:
        return 0
    return xp.einsum('wxy->w',D**2)

def compute_trace(D):
    if isinstance(D,list):
        return  _trace(D[0])+_trace(D[1]) 
    return _trace(D)

def compute_regularization(D,eps_sq=None):
    if eps_sq is None:
        return None
    tr = compute_trace(D)
    return 1./xp.sqrt(1.+eps_sq*tr)

def _get_Dr(D,nb,s):
    sh = D.shape[1]
    if sh==nb:
        return D
    if sh==nb*2:
        Dr = D[:,:nb] if s==0 else D[:,nb:]
        return Dr
    raise NotImplementedError

def _get_Dl(D,nb,s):
    sh = D.shape[2]
    if sh==nb:
        return D
    if sh==nb*2:
        Dr = D[:,:,:nb] if s==0 else D[:,:,nb:]
        return Dr
    raise NotImplementedError

def _update_rdm1_sd(D,d,u,s):
    _,nb,r = u.shape
    idx = xp.arange(r)

    Du = xp.einsum('wxy,wyr->wxr',_get_Dl(D,nb,s),u)
    uD = xp.einsum('wxr,wxy->wry',u,_get_Dr(D,nb,s))
    uDu = xp.einsum('wxr,wxs->wrs',u,_get_Dr(Du,nb,s))
    M = uDu.copy()
    M[:,idx,idx] += 1./d
    ratio = d.prod(axis=1)*xp.linalg.det(M)

    M = xp.linalg.inv(M)
    D1 = xp.einsum('wxr,wrs->wxs',Du,M)
    D1 = xp.einsum('wxr,wry->wxy',D1,uD)

    M = xp.eye(r)[None,:,:] - xp.einsum('wrs,wsm->wrm',M,uDu) 
    M *= d[:,None,:]
    D2 = xp.einsum('wxr,wrs->wxs',Du,M)
    D2 = xp.einsum('wxr,wyr->wxy',D2,u)
    return D1,D2,ratio

def update_rdm1(D,ovlp_ratio,d,u,w,s):
    if d is None:
        return D,ovlp_ratio
    if isinstance(D,list):
        if D[s] is None:
            return D,ovlp_ratio
        D1,D2,r = _update_rdm1_sd(D[s][w],d,u,s)
        D[s][w] -= D1
        D[s][w] += D2
        ovlp_ratio[w] *= r
        return D,ovlp_ratio
    nb = u.shape[1]
    D1,D2,r = _update_rdm1_sd(D[w],d,u,s)
    if s==0:
        D[w][:,:,:nb] += D2
    else:
        D[w][:,:,nb:] += D2
    D[w] -= D1
    ovlp_ratio[w] *= r
    return D,ovlp_ratio
