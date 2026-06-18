import numpy as np
import plum,h5py
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.trial_wavefunction.single_det import SingleDet 
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp

def _preprocess_trial(trial):
    trial.psi0 = None
    trial.psi0a = None
    trial.psi0b = None
    trial.G = None
    trial.Ghalf = None

@plum.dispatch
def preprocess_trial(trial:SingleDet)
    trial.psi = [trial.psi0a.real,None]
    if trial.nelec[1]>0:
        trial.psi[1] = trial.psi0b.real
    _preprocess_trial(trial)

@plum.dispatch
def preprocess_trial(trial:SingleDetGHF):
    trial.psi = trial.psi0.real
    _preprocess_trial(trial)

def _UB(B,chol_basis):
    if B is None:
        return None
    return xp.einsum('dxp,xi->dpi',chol_basis,B)

def _h1B(B,h1e):
    if B is None:
        return None
    return xp.dot(h1e,B)

def preprocess_hamiltonian(hamiltonian,trial):
    if isinstance(trial.psi,list):
        psi = trial.psi
    else: 
        nb = trial.nbasis
        psi = [trial.psi[:nb],trial.psi[nb:]]

    hamiltonian.UB = [_UB(Bi,hamiltonian.chol_basis) for Bi in psi]
    hamiltonian.h1B = [_h1B(Bi,hamiltonian.h1e) for Bi in psi]

    if hamiltonian.chol is None:
        return
    hamiltonian.LB = [_UB(B,hamiltonian.chol) for Bi in psi]

def _preprocess_walkers(walkers):
    walkers.buff_names += ['weight','phase']
    walkers.phase = xp.ones(walkers.nwalkers) 
    walkers.ovlp = xp.ones(walkers.nwalkers) 
    walkers.G = None
    walkers.R = None 
    walkers.sgn_ovlp = None
    walkers.eloc = None
    walkers.Ga = None
    walkers.Gb = None
    walkers.Ghalfa = None
    walkers.Ghalfb = None
    walkers.buff_size = round(walkers.set_buff_size_single_walker() / float(walkers.nwalkers))
    walkers.walker_buffer = xp.zeros(walkers.buff_size)

@plum.dispatch
def preprocess(walkers:UHFWalkers,trial,hamiltonian)
    preprocess_trial(trial)
    preprocess_hamiltonian(hamiltonian)

    walkers.phia = walkers.phia.real
    if walkers.ndown>0:
        walkers.phib = walkers.phib.real
    else:
        walkers.phib = None
    walkers.buff_names = ['phia','phib']
    compute_ovlp(walkers,trial)
    if walkers.S is None:
        walkers.buff_names += ['Sa','Sb']
    else:
        walkers.buff_names += ['S']
    _preprocess_walkers(walkers)

@plum.dispatch
def preprocess(walkers:GHFWalkers,trial,hamiltonian):
    preprocess_trial(trial)
    preprocess_hamiltonian(hamiltonian)

    walkers.phi = walkers.phi.real
    walkers.S = compute_ovlp(walkers,trial)
    walkers.buff_names = ['phi','S']
    _preprocess_walkers(walkers)

@plum.dispatch
def parse_phi(walkers:UHFWalkers):
    if walkers.phib is None:
        return walkers.phia
    return xp.concatenate([walkers.phia,walkers.phib],axis=2)

@plum.dispatch
def parse_phi(walkers:GHFWalkers):
    return walkers.phi

@plum.dispatch
def load_phi(walkers:UHFWalkers,phi):
    phi = xp.asarray(phi)
    walkers.phia = phi[:,:,:walkers.nup]
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

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDet):
    phi = [walkers.phia,walkers.phib]
    CB = [_ovlp(Ci,Bi) for Ci,Bi in zip(phi,trial.psi)] 
    walkers.Sa,walkers.Sb = [_inv(Si) for Si in CB]
    walkers.S = None

@plum.dispatch
def compute_ovlp(walkers:UHFWalkers,trial:SingleDetGHF):
    phi = [walkers.phia,walkers.phib]
    nb = trial.nbasis
    B = [trial.psi[:nb],trial.psi[nb:]]
    CB = [_ovlp(Ci,Bi) for Ci,Bi in zip(phi,B)] 
    if CB[1] is None:
        CB = CB[0]
    else:
        CB = xp.concatenate(CB,axis=1)
    walkers.S = xp.linalg.inv(CB)

@plum.dispatch
def compute_ovlp(walkers:GHFWalkers,trial:SingleDetGHF):
    CB = _ovlp(walkers.phi,trial.psi)
    walkers.S = xp.linalg.inv(CB)

def _SC(S,C):
    if C is None:
        return None
    return xp.einsum('wij,wxj->wix',S,C) 

@plum.dispatch
def compute_SC(walkers:UHFWalkers):
    if walkers.S is None:
        S = [walkers.Sa,walkers.Sb]
    else:
        nu = walkers.nup
        S = [walkers.S[:,:,:nu],walkers.S[:,:,nu:]]
    return [_SC(Si,Ci) for Si,Ci in zip(S,walkers.phi)]

@plum.dispatch
def compute_SC(walkers:GHFWalkers):
    nb = walkers.nbasis
    C = [walkers.phi[:,:nb],walkers.phi[:,nb:]]
    return [_SC(walkers.S,Ci) for Ci in C]

def multiply_h1(h1B,SC):
    if h1B is None:
        return None
    if SC is None:
        return None
    return xp.einsum('wix,xj->wij',SC,h1B)

def trace_h1(h1B,SC):
    if h1B is None:
        return 0 
    if SC is None:
        return 0
    return xp.einsum('wix,xi->w',SC,h1B)

def _1rdm_diag(B,SC,s):
    if B is None:
        return None
    if SC is None:
        return None
    ne = B.shape[1]
    if ne==SC.shape[1]:
        return xp.einsum('xi,wix->wx',B,SC)
    if s==0:
        return xp.einsum('xi,wix->wx',B,SC[:,:ne])
    else:
        return xp.einsum('xi,wix->wx',B,SC[:,-ne:])

@plum.dispatch
def compute_1rdm_diag(trial:SingleDet,SC):
    D = [_1rdm_diag(trial.psi[s],SC[s],s) for s in (0,1)] 
    return D[0],D[1],None,None

@plum.dispatch
def compute_1rdm_diag(trial:SingleDetGHF,SC):
    nb = trial.nbasis
    Daa = _1rdm_diag(trial.psi[:nb],SC[0],None) 
    Dab = _1rdm_diag(trial.psi[:nb],SC[1],None) 
    Dba = _1rdm_diag(trial.psi[nb:],SC[0],None) 
    Dbb = _1rdm_diag(trial.psi[nb:],SC[1],None) 
    return Daa,Dbb,Dab,Dba

def compute_E1(walkers,h1B):
    SC = compute_SC(walkers)
    E1 = [trace_h1(h1Bi,SCi) for h1Bi,SCi in zip(h1B,SC)]
    E1 = E1[0]+E1[1]
    return E1,SC

@plum.dispatch
def local_energy(hamiltonian:HubbardSOR,walkers,trial):
    E1,SC = compute_E1(walkers,hamiltonian.h1B)

    Daa,Dbb,Dab,Dba = compute_1rdm_diag(trial,SC)
    if Dbb is None:
        E2 = xp.zeros(Daa.shape[0])
        return E1+E2,E1,E2 
    E2 = (Daa*Dbb).sum(axis=1)
    if Dab is None:
        E2 *= hamiltonian.hubbard_U
        return E1+E2,E1,E2
    E2 -= (Dab*Dba).sum(axis=1)
    E2 *= hamiltonian.hubbard_U
    return E1+E2,E1,E2

def compute_chol_E2(SC,hamiltonian,cross_spin):
    LBa,LBb = hamiltonian.LB
    E2 = 0
    for i in range(LBa.shape[0]):
        LBai = LBa[i]
        LBbi = None if LBb is None else LBb[i] 

        E2 += (trace_h1(LBai,SC[0]) + trace_h1(LBbi,SC[1]))**2

        ta = multiply_h1(LBai,SC[0])
        tb = multiply_h1(LBbi,SC[1])
        E2 -= xp.einsum('wij,wji->w',ta,ta)
        if tb is None:
            continue
        E2 -= xp.einsum('wij,wji->w',tb,tb)
        if cross_spin:
            E2 -= 2*xp.einsum('wij,wji->w',ta,tb)
    return 0.5*E2

@plum.dispatch
def local_energy(hamiltonian:QCSOR,walkers:UHFWalkers,trial:SingleDet):
    E1,SC = compute_E1(walkers,hamiltonian.h1B)
    E2 = compute_chol_E2(SC,hamiltonian,False)
    return E1+E2,E1,E2

@plum.dispatch
def local_energy(hamiltonian:QCSOR,walkers,trial:SingleDetGHF):
    E1,SC = compute_E1(walkers,hamiltonian.h1B)
    E2 = compute_chol_E2(SC,hamiltonian,True)
    return E1+E2,E1,E2

#def _trace(D,iws=None):
#    if D is None:
#        return 0
#    Dw = D if iws is None else D[iws]
#    return xp.einsum('wxy->w',Dw**2)
#
#def compute_trace(D,iws=None):
#    if isinstance(D,list):
#        return  _trace(D[0],iws=iws)+_trace(D[1],iws=iws) 
#    return _trace(D,iws=iws)
#
#def compute_regularization(walkers,eps_sq=None,iws=None):
#    if eps_sq is None:
#        return None 
#    tr = compute_trace(walkers.D,iws=iws)
#    R = 1./xp.sqrt(1.+eps_sq*tr)
#    if iws is None:
#        walkers.R = R
#    else:
#        walkers.R[iws] = R

def _update_walkers(C,w,d,u):
    if C is None:
        return C,None
    uC = xp.einsum('wxr,wxi->wri',u,C[w])
    C[w] += xp.einsum('wxr,wr,wri->wxi',u,d,uC)
    return C,uC.transpose(0,2,1)

def _multiply_Cu(S,Cu,s):
    ne = Cu.shape[1]
    if S.shape[-1]==ne:
        return xp.einsum('wij,wjr->wir',S,Cu)
    if s==0:
       return xp.einsum('wij,wjr->wir',S[:,:,:ne],Cu)
    else:
       return xp.einsum('wij,wjr->wir',S[:,:,-ne:],Cu)

def _multiply_uB(S,uB,s):
    ne = uB.shape[-1]
    if S.shape[1]==ne:
        return xp.einsum('wri,wij->wrj',uB,S)
    if s==0:
        return xp.einsum('wri,wij->wrj',uB,S[:,:ne])
    else:
        return xp.einsum('wri,wij->wrj',uB,S[:,-ne:])

def _compute_ovlp_update(uDu,d,SCu,uBS,thresh=1e4):
    r = d.shape[-1]
    M = xp.eye(r)[None,:,:] + d[:,:,None] * uDu
    detM = xp.linalg.det(M)
    if (detM[xp.fabs(detM)<1e-6]).size>0:
        print('detM=',detM)
        exit()

    if xp.linalg.norm(detM)>thresh or xp.count_nonzero(xp.isnan(detM))>0:
        print('before inverse')
        print('M=',to_host(M))
        exit()

    M = xp.linalg.inv(M)*d[:,None,:]
    if xp.linalg.norm(M)>thresh or xp.count_nonzero(xp.isnan(M))>0:
        print('after inverse')
        print('M=',to_host(M))
        exit()

    S1 = xp.einsum('wir,wrs->wis',SCu,M)
    S1 = xp.einsum('wir,wrj->wij',S1,uBS)
    return detM,S1

def _update_ovlp_1(S,b,w,Cu,uB,d,s):
    if Cu is None:
        return S,b
    if uB is None:
        return S,b

    if isinstance(S,list):
        if S[s] is None:
            return S,b
        Sw = S[s][w]
    else:
        Sw = S[w]
    SCu = _multiply_Cu(Sw,Cu,s)
    uBS = _multiply_uB(Sw,uB,s)
    uDu = _multiply_uB(SCu,uB,s)
    detM,S1 = _compute_ovlp_update(uDu,d,SCu,uBS) 
    b[w] *= detM 
    if isinstance(S,list):
        S[s][w] -= S1
    else:
        S[w] -= S1
    return S,b

def _update_ovlp_2(S,b,w,Cu,uB,d):
    if isinstance(S,list):
        assert isinstance(Cu,list)
        assert isinstance(uB,list)
        S,b = _update_ovlp_1(S,b,w,Cu[0],uB[0],d[:,:1],0)
        S,b = _update_ovlp_1(S,b,w,Cu[1],uB[1],d[:,1:],1)
        return S,b
    Sw = S[w]
    nw,ne,_ = Sw.shape

    SCu = xp.zeros((nw,ne,2))
    SCu[:,:,:1] = _multiply_Cu(Sw,Cu[0],0) 
    if Cu[1] is not None:
        SCu[:,:,1:] = _multiply_Cu(Sw,Cu[1],1) 

    uBS = xp.zeros((nw,2,ne))
    uBS[:,:1] = _multiply_uB(Sw,uB[0],0)
    if uB[1] is not None:
        uBS[:,1:] = _multiply_uB(Sw,uB[1],1)

    uDu = xp.zeros((nw,2,2))
    uDu[:,:1] = _multiply_uB(SCu,uB[0],0)
    if uB[1] is not None:
        uDu[:,1:] = _multiply_uB(SCu,uB[1],1)

    detM,S1 = _compute_ovlp_update(uDu,d,SCu,uBS) 
    b[w] *= detM 
    S[w] -= S1
    return S,b

def _lowdin(C,w,Cu,d,d2): 
    if C is None:
        return C,None,None
    r = d.shape[1]
    p = Cu.copy()
    if r==1:
        norm_sq = (p**2).sum(axis=1)
        p /= xp.sqrt(norm_sq)[:,None,:]
        delta = d2*norm_sq 
    else:
        p,s = xp.linalg.qr(p,mode='reduced')
        delta = xp.einsum('wrs,ws,wts->wrt',s,d2,s)
        delta,v = xp.linalg.eigh(delta)
        p = xp.einsum('wir,wrs->wis',p,v)
    delta = xp.sqrt(delta+1.)
    left = xp.einsum('wxi,wir->wxr',C[w],p) * (1./delta-1.)[:,None,:] 
    C[w] += xp.einsum('wxr,wir->wxi',left,p)
    return C,p,delta-1.

def _lowdin_ovlp(S,w,p,delta,s):
    if p is None:
        return S
    if isinstance(S,list):
        Sw = S[s][w] 
    else:
        Sw = S[w]
    left = _multiply_Cu(Sw,p,s) * delta[:,None,:]
    S1 = xp.einsum('wir,wjr->wij',left,p)
    if isinstance(S,list):
        S[s][w] += S1
        return S
    ne = S1.shape[-1] 
    if S.shape[-1]==ne:
        S[w] += S1
        return S
    if s==0:
        S[w,:,:ne] += S1
    else:
        S[w,:,-ne:] += S1
    return S

@plum.dispatch
def update_walkers(walkers:UHFWalkers,rotations,b,lowdin=False,eps_sq=None):
    S = [walkers.Sa,walkers.Sb] if walkers.S is None else walkers.S
    for typ in rotations.typs:
        w,d,d2,u,uB = rotations.get_data(typ)
        if w is None:
            continue
        if typ=='h2ab':
            walkers.phia,Cua = _update_walkers(walkers.phia,w,d[:,:1],u[:,:,:1])
            walkers.phib,Cub = _update_walkers(walkers.phib,w,d[:,1:],u[:,:,1:])
            S,b = _update_ovlp_2(S,b,w,[Cua,Cub],uB,d)
            if lowdin:
                walkers.phia,pa,delta_a = _lowdin(walkers.phia,w,Cua,d[:,:1],d2[:,:1])
                S = _lowdin_ovlp(S,w,pa,delta_a,0)
                walkers.phib,pb,delta_b = _lowdin(walkers.phib,w,Cub,d[:,1:],d2[:,1:])
                S = _lowdin_ovlp(S,w,pb,delta_b,1)
            continue
        if typ[-1]=='a':
            walkers.phia,Cu = _update_walkers(walkers.phia,w,d,u)
            S,b = _update_ovlp_1(S,b,w,Cu,uB[0],d,0)
            if lowdin:
                walkers.phia,p,delta = _lowdin(walkers.phia,w,Cu,d,d2)
                S = _lowdin_ovlp(S,w,p,delta,0)
        else:
            walkers.phib,Cu = _update_walkers(walkers.phib,w,d,u)
            S,b = _update_ovlp_1(S,b,w,Cu,uB[1],d,1)
            if lowdin:
                walkers.phib,p,delta = _lowdin(walkers.phib,w,Cu,d,d2)
                S = _lowdin_ovlp(S,w,p,delta,1)
    if isinstance(S,list):
        walkers.Sa = S[0]
        walkers.Sb = S[1]
    else:
        walkers.S = S
    return b

@plum.dispatch
def update_walkers(walkers:GHFWalkers,rotations,b,lowdin=False,eps_sq=None):
    nb = walkers.nbasis
    for typ in rotations.typs:
        w,d,d2,u,uB = rotations.get_data(typ)
        if w is None:
            continue
        if typ=='h2ab':
            walkers.phi[:,:nb],Cua = _update_walkers(walkers.phi[:,:nb],w,d[:,:1],u[:,:,:1])
            walkers.phi[:,nb:],Cub = _update_walkers(walkers.phi[:,nb:],w,d[:,1:],u[:,:,1:])
            Cu = [Cua,Cub] 
            walkers.S,b =  _update_ovlp_2(walkers.S,b,w,Cu,uB,d)
            if lowdin:
                walkers.phi,p,delta = _lowdin(walkers.phi,w,xp.concatenate(Cu,axis=2),d,d2)
                walkers.S = _lowdin_ovlp(walkers.S,w,p,delta,None)
            continue
        if typ[-1]=='a':
            walkers.phi[:,:nb],Cu = _update_walkers(walkers.phi[:,:nb],w,d,u)
        else:
            walkers.phi[:,nb:],Cu = _update_walkers(walkers.phi[:,nb:],w,d,u)
        s = {'a':0,'b':1}[typ[-1]] 
        walkers.S,b = _update_ovlp_1(walkers.S,b,w,Cu,uB[s],d,s)
        if lowdin:
            walkers.phi,p,delta = _lowdin(walkers.phi,w,Cu,d,d2)
            walkers.S = _lowdin_ovlp(walkers.S,w,p,delta,None)
    return b

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
    for w,U in enumerate(Us):
        for s,Ui in enumerate(U):
            if walkers.phi[s] is None:
                continue
            if Ui is None:
                continue
            walkers.phi[s][w] = xp.dot(U[s],walkers.phi[s][w])
    detR = xp.ones(walkers.nwalkers)
    if lowdin:
        for s in (0,1):
            walkers.phi[s],detR_ = _lowdin_slow(walkers.phi[s])
            if detR_ is not None:
                detR *= detR_
    return detR

@plum.dispatch
def update_walkers_slow(walkers:GHFWalkers,Us,lowdin=False):
    nb = walkers.nbasis
    for w,U in enumerate(Us):
        if U[0] is not None:
            walkers.phi[w,:nb] = xp.dot(U[0],walkers.phi[w,:nb])
        if U[1] is not None:
            walkers.phi[w,nb:] = xp.dot(U[1],walkers.phi[w,nb:])
    detR = xp.ones(walkers.nwalkers)
    if lowdin:
        walkers.phi,detR = _lowdin_slow(walkers.phi)
    return detR


