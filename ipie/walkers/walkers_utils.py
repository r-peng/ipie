import numpy as np
import plum,h5py
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

def _preprocess_walkers(walkers,scalar_ovlp=True):
    walkers.buff_names += ['weight','phase']
    if scalar_ovlp:
        walkers.buff_names += ['ovlp']
    walkers.phase = xp.ones(walkers.nwalkers) 
    walkers.ovlp = xp.ones(walkers.nwalkers) 
    walkers.R = None 
    walkers.sgn_ovlp = None
    walkers.eloc = None
    walkers.buff_size = round(walkers.set_buff_size_single_walker() / float(walkers.nwalkers))
    walkers.walker_buffer = xp.zeros(walkers.buff_size)

@plum.dispatch
def preprocess_walkers(walkers:UHFWalkers,scalar_ovlp=True):
    walkers.buff_names = ['phia','phib']
    walkers.phia = walkers.phia.real
    if walkers.phib is not None:
        walkers.phib = walkers.phib.real
    walkers.G = None
    walkers.Ga = None
    walkers.Gb = None
    walkers.Ghalfa = None
    walkers.Ghalfb = None
    _preprocess_walkers(walkers,scalar_ovlp=scalar_ovlp)

@plum.dispatch
def preprocess_walkers(walkers:GHFWalkers,scalar_ovlp=True):
    walkers.buff_names = ['phi']
    walkers.phi = walkers.phi.real
    walkers.G = None
    _preprocess_walkers(walkers,scalar_ovlp=scalar_ovlp)

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
def compute_ovlp(walkers:UHFWalkers,trial:SingleDet,inv=True,scalar_ovlp=False,iws=None):
    if iws is None:
        C = [walkers.phia,walkers.phib]
    else:
        C = [walkers.phia[iws],None]
        if walkers.phib is not None:
            C[1] = walkers.phib[iws]

    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,trial.psi)] 
    if scalar_ovlp:
        det = xp.linalg.det(CdB[0])
        if CdB[1] is not None:
            det *= xp.linalg.det(CdB[1])
        if iws is None:
            walkers.ovlp = det
        else:
            walkers.ovlp[iws] = det
    if inv:
        CdB = [_inv(oi) for oi in CdB]
    return CdB,C

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDetGHF,inv=True,scalar_ovlp=False,iws=None):
    if iws is None:
        C = [walkers.phia,walkers.phib]
    else:
        C = [walkers.phia[iws],None]
        if walkers.phib is not None:
            C[1] = walkers.phib[iws]

    nb = trial.nbasis
    B = trial.psi[:nb],trial.psi[nb:]
    CdB = [_ovlp(Ci,B=Bi) for Ci,Bi in zip(C,B)] 
    CdB = xp.concatenate(CdB,axis=1)
    if scalar_ovlp:
        det = xp.linalg.det(CdB)
        if iws is None:
            walkers.ovlp = det
        else:
            walkers.ovlp[iws] = det
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB,C

@plum.dispatch
def compute_ovlp(walkers:GHFWalkers,trial:SingleDetGHF,inv=True,scalar_ovlp=False,iws=None):
    C = walkers.phi if iws is None else walkers.phi[iws]
    CdB = _ovlp(C,B=trial.psi)
    if scalar_ovlp:
        det = xp.linalg.det(CdB)
        if iws is None:
            walkers.ovlp = det
        else:
            walkers.ovlp[iws] = det
    if inv:
        CdB = xp.linalg.inv(CdB)
    return CdB,C

def _rdm1(ovlp,C,B=None):
    if C is None:
        return None
    if xp.count_nonzero(xp.isnan(ovlp))>0:
        print('ovlp inv')
        print(ovlp)
        exit()

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
def compute_rdm1(walkers:UHFWalkers,trial:SingleDet,scalar_ovlp=False,eps_sq=None,iws=None):
    CdBinv,C = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp,iws=iws)
    D = [_rdm1(oi,Ci,B=Bi) for oi,Ci,Bi in zip(CdBinv,C,trial.psi)] 
    if iws is None:
        walkers.D = D
    else:
        walkers.D[0][iws] = D[0]
        if D[1] is not None:
            walkers.D[1][iws] = D[1]
    compute_regularization(walkers,eps_sq=eps_sq,iws=iws)

@plum.dispatch
def compute_rdm1(walkers:UHFWalkers,trial:SingleDetGHF,scalar_ovlp=False,eps_sq=None,iws=None):
    CdBinv,C = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp,iws=iws)
    tmp = xp.einsum('xi,wij->wxj',trial.psi,CdBinv) 

    nw = walkers.nwalkers if iws is None else iws.size
    nu,nd = trial.nelec
    nb = trial.nbasis
    D = xp.zeros((nw,nb*2,nb*2))
    D[:,:,:nb] = xp.einsum('wxj,wyj->wxy',tmp[:,:,:nu],C[0])
    if C[1] is not None:
        D[:,:,nb:] = xp.einsum('wxj,wyj->wxy',tmp[:,:,nu:],C[1])
    if iws is None:
        walkers.D = D
    else:
        walkers.D[iws] = D
    compute_regularization(walkers,eps_sq=eps_sq,iws=iws)

@plum.dispatch
def compute_rdm1(walkers:GHFWalkers,trial:SingleDetGHF,scalar_ovlp=False,eps_sq=None,iws=None):
    CdBinv,C = compute_ovlp(walkers,trial,scalar_ovlp=scalar_ovlp,iws=iws)
    D = _rdm1(CdBinv,C,B=trial.psi)
    if iws is None:
        walkers.D = D
    else:
        walkers.D[iws] = D
    compute_regularization(walkers,eps_sq=eps_sq,iws=iws)

def _trace(D,iws=None):
    if D is None:
        return 0
    Dw = D if iws is None else D[iws]
    return xp.einsum('wxy->w',Dw**2)

def compute_trace(D,iws=None):
    if isinstance(D,list):
        return  _trace(D[0],iws=iws)+_trace(D[1],iws=iws) 
    return _trace(D,iws=iws)

def compute_regularization(walkers,eps_sq=None,iws=None):
    if eps_sq is None:
        return None 
    tr = compute_trace(walkers.D,iws=iws)
    R = 1./xp.sqrt(1.+eps_sq*tr)
    if iws is None:
        walkers.R = R
    else:
        walkers.R[iws] = R

def _update_walkers(C,d,u,w):
    if C is None:
        return C,None
    uC = xp.einsum('wxr,wxi->wri',u,C[w])
    C[w] += xp.einsum('wxr,wr,wri->wxi',u,d,uC)
    return C,uC

def _lowdin(C,uC,d,d2,w): 
    if C is None:
        return C
    r = d.shape[1]
    p = uC.transpose(0,2,1)
    if r==1:
        norm_sq = (p**2).sum(axis=1)
        p /= xp.sqrt(norm_sq)[:,None,:]
        lambda_ = d2*norm_sq 
    else:
        p,s = xp.linalg.qr(p,mode='reduced')
        lambda_ = xp.einsum('wrs,ws,wts->wrt',s,d2,s)
        lambda_,v = xp.linalg.eigh(lambda_)
        p = xp.einsum('wir,wrs->wis',p,v)
    delta = 1./xp.sqrt(lambda_+1.)-1.
    left = xp.einsum('wxi,wir->wxr',C[w],p)
    C[w] += xp.einsum('wxr,wr,wir->wxi',left,delta,p)
    return C

@plum.dispatch
def update_walkers(walkers:UHFWalkers,rotations,lowdin=False):
    for typ,rotations_ in rotations.items():
        w = rotations_['w']
        d = rotations_['d']
        d2 = rotations_['d2']
        u = rotations_['u']
        if typ=='h2ab':
            walkers.phia,uC = _update_walkers(walkers.phia,d[:,:1],u[:,:,:1],w)
            if lowdin:
                walkers.phia = _lowdin(walkers.phia,uC,d[:,:1],d2[:,:1],w)
            walkers.phib,uC = _update_walkers(walkers.phib,d[:,1:],u[:,:,1:],w)
            if lowdin:
                walkers.phib = _lowdin(walkers.phib,uC,d[:,1:],d2[:,1:],w)
        else:
            if typ[-1]=='a':
                walkers.phia,uC = _update_walkers(walkers.phia,d,u,w)
                if lowdin:
                    walkers.phia = _lowdin(walkers.phia,uC,d,d2,w)
            else:
                walkers.phib,uC = _update_walkers(walkers.phib,d,u,w)
                if lowdin:
                    walkers.phib = _lowdin(walkers.phib,uC,d,d2,w)

@plum.dispatch
def update_walkers(walkers:GHFWalkers,rotations,lowdin=False):
    nb = walkers.nbasis
    for typ,rotations_ in rotations.items():
        w = rotations_['w']
        d = rotations_['d']
        d2 = rotations_['d2']
        u = rotations_['u']
        if typ=='h2ab':
            walkers.phi[:,:nb],uCa = _update_walkers(walkers.phi[:,:nb],d[:,:1],u[:,:,:1],w)
            walkers.phi[:,nb:],uCb = _update_walkers(walkers.phi[:,nb:],d[:,1:],u[:,:,1:],w)
            if lowdin:
                uC = xp.concatenate([uCa,uCb],axis=1)
                walkers.phi = _lowdin(walkers.phi,uC,d,d2,w)
        else:
            if typ[-1]=='a':
                walkers.phi[:,:nb],uC = _update_walkers(walkers.phi[:,:nb],d,u,w)
            else:
                walkers.phi[:,nb:],uC = _update_walkers(walkers.phi[:,nb:],d,u,w)
            if lowdin:
                walkers.phi = _lowdin(walkers.phi,uC,d,d2,w)

def _lowdin_slow(C,thresh=1e-10):
    if C is None:
        return C,None
    ovlp = _ovlp(C) 
    w,v = xp.linalg.eigh(ovlp)
    w = 1./xp.sqrt(w)
    ovlp = xp.einsum('wij,wj,wkj->wik',v,w,v)
    C = xp.einsum('wxi,wij->wxj',C,ovlp)
    detR = xp.linalg.det(ovlp) 
    ovlp = _ovlp(C)
    assert xp.linalg.norm(ovlp-xp.eye(ovlp.shape[2])[None,:,:])<thresh
    return C,detR

@plum.dispatch
def update_walkers_slow(walkers:UHFWalkers,Us,lowdin=False):
    phi = [walkers.phia.real.copy(),None]
    if walkers.phib is not None:
        phi[1] = walkers.phib.real.copy()
    for w,U in enumerate(Us):
        for s,Ui in enumerate(U):
            if phi[s] is None:
                continue
            if Ui is None:
                continue
            phi[s][w] = xp.dot(U[s],phi[s][w])
    if lowdin:
        detR = xp.ones(walkers.nwalkers)
        for s in (0,1):
            phi[s],detR_ = _lowdin_slow(phi[s])
            if detR_ is not None:
                detR *= detR_
    return phi[0],phi[1],detR

@plum.dispatch
def update_walkers_slow(walkers:GHFWalkers,Us,lowdin=False):
    phi = walkers.phi.real.copy()
    nb = walkers.nbasis
    for w,U in enumerate(Us):
        if U[0] is not None:
            phi[w,:nb] = xp.dot(U[0],phi[w,:nb])
        if U[1] is not None:
            phi[w,nb:] = xp.dot(U[1],phi[w,nb:])
    if lowdin:
        phi,detR = _lowdin_slow(phi)
    return phi,detR

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

def _update_rdm1_ovlp_sd(D,b,d,u,w,s,thresh=1e4):
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
    M = xp.eye(r)[None,:,:] + d[:,:,None] * uDu
    detM = xp.linalg.det(M)
    if (detM[xp.fabs(detM)<1e-6]).size>0:
        print('detM=',detM)
        exit()
    b[w] *= detM 
    if xp.linalg.norm(b[w])>thresh or xp.count_nonzero(xp.isnan(b[w]))>0:
        print('before inverse')
        print('b=',to_host(b[w]))
        print('d=',to_host(d))
        print('M=',to_host(M))
        print('udu=',to_host(uDu))
        print('Dw nan=',xp.count_nonzero(xp.isnan(Dw)))
        exit()

    M = xp.linalg.inv(M)*d[:,None,:]
    if xp.linalg.norm(M)>thresh or xp.count_nonzero(xp.isnan(M))>0:
        print('after inverse')
        print('M=',to_host(M))
        exit()

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

def update_rdm1_and_ovlp(walkers,b,rotations,eps_sq=None):
    for typ,rotations_ in rotations.items():
        w = rotations_['w']
        d = rotations_['d']
        u = rotations_['u']
        if typ=='h2ab':
            walkers.D,b = _update_rdm1_ovlp_sd(walkers.D,b,d[:,:1],u[:,:,:1],w,0)
            walkers.D,b = _update_rdm1_ovlp_sd(walkers.D,b,d[:,1:],u[:,:,1:],w,1)
        else:
            if typ[-1]=='a':
                walkers.D,b = _update_rdm1_ovlp_sd(walkers.D,b,d,u,w,0)
            else:
                walkers.D,b = _update_rdm1_ovlp_sd(walkers.D,b,d,u,w,1)
    if eps_sq is None:
        return b
    b *= walkers.R
    compute_regularization(walkers,eps_sq=eps_sq)
    b /= walkers.R
    return b

