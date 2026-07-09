import numpy as np
import plum,h5py
from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
from ipie.utils.backend import to_host#,qr,qr_mode
from ipie.utils.backend import arraylib as xp

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

def _compute_Cu(C,w,u):
    if C is None:
        return None
    return xp.einsum('wxi,wxr->wir',C[w],u)

def _update_walkers(C,w,d,u,Cu):
    if C is None:
        return None
    C[w] += xp.einsum('wxr,wir->wxi',u*d[:,None,:],Cu)
    return C

def _orthogonalise_Cu_1(Cu,thresh=1e-10):
    norm = xp.sqrt((Cu[:,:,0]**2).sum(axis=1))
    idx = xp.nonzero(norm>thresh)[0]
    r = norm[idx].reshape(idx.size,1,1)
    return idx,Cu[idx]/r,r

def _orthogonalise_Cu_2(Cu,thresh=1e-10):
    q,r = xp.linalg.qr(Cu,mode='reduced')
    if r.shape[1]==1:
        assert q.shape[-1]==1
        return q,r
    if xp.linalg.norm(r[:,1,1])<thresh:
        q = q[:,:,:1]
        r = r[:,:1]
    return q,r

def _compute_lowdin(Cu,d2,thresh=1e-10):
    if Cu is None:
        return None
    if d2.shape[1]==1:
        idx,q,r = _orthogonalise_Cu_1(Cu,thresh=thresh)
        delta = d2[idx]*(r[:,:,0]**2)
    else:
        idx = None
        q,r = _orthogonalise_Cu_2(Cu,thresh=thresh)
        delta = xp.einsum('wrs,ws,wts->wrt',r,d2,r)
        if delta.shape[-1]==1:
            delta = delta[:,:,0]
        else:
            delta,v = xp.linalg.eigh(delta)
            q = xp.einsum('wir,wrs->wis',q,v)
    if delta[delta+1.<thresh].size>0:
        print('delta=',delta)
        print('r=',r)
    delta = xp.sqrt(delta+1.)
    return delta,q,idx

def _update_walkers_lowdin(C,w,info): 
    if C is None:
        return None,w 
    delta,q,idx = info
    if idx is not None:
        w = w[idx]
    left = xp.einsum('wxi,wir->wxr',C[w],q) * (1./delta-1.)[:,None,:] 
    C[w] += xp.einsum('wxr,wir->wxi',left,q)
    return C,w

def _update_ovlp_lowdin(S,w,info,s):
    if info is None:
        return S
    delta,q,_ = info
    if isinstance(S,list):
        Sw = S[s][w] 
    else:
        Sw = S[w]
    left = _multiply_Cu(Sw,q,s) * (delta-1.)[:,None,:]
    S1 = xp.einsum('wir,wjr->wij',left,q)
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

def _compute_ovlp_update(uDu,d,uBS,SCu,thresh=1e4,b=None,update_S=True):
    M = xp.eye(d.shape[-1])[None,:,:] + d[:,:,None] * uDu 

    detM = None
    S1 = None
    if b is not None:
        detM = xp.linalg.det(M)
        if (detM[xp.fabs(detM)<1e-6]).size>0:
            print('detM=',detM)
            exit()
    if update_S:
        M = xp.linalg.inv(M)*d[:,None,:]
        S1 = xp.einsum('wir,wrs->wis',SCu,M)
        S1 = xp.einsum('wir,wrj->wij',S1,uBS)
    return detM,S1

def _update_ovlp_1(S,w,Cu,uB,d,s,b=None,update_S=True):
    if Cu is None:
        return S,b
    if uB is None:
        return S,b
    if isinstance(S,list):
        Sw = S[s][w]
    else:
        Sw = S[w]

    SCu = _multiply_Cu(Sw,Cu,s)
    uBS = _multiply_uB(Sw,uB,s)
    uDu = _multiply_uB(SCu,uB,s)
    detM,S1 = _compute_ovlp_update(uDu,d,uBS,SCu,b=b,update_S=update_S) 
    if b is not None:
        b[w] *= detM 
    if update_S:
        if isinstance(S,list):
            S[s][w] -= S1
        else:
            S[w] -= S1
    return S,b

def _update_ovlp_2(S,w,Cu,uB,d,b=None,update_S=True):
    if isinstance(S,list):
        assert isinstance(Cu,list)
        assert isinstance(uB,list)
        if not isinstance(d,list):
            d = [d[:,:1],d[:,1:]]
        S,b = _update_ovlp_1(S,w,Cu[0],uB[0],d[0],0,b=b,update_S=update_S)
        S,b = _update_ovlp_1(S,w,Cu[1],uB[1],d[1],1,b=b,update_S=update_S)
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

    detM,S1 = _compute_ovlp_update(uDu,d,uBS,SCu,b=b,update_S=update_S) 
    if b is not None:
        b[w] *= detM 
    if update_S:
        S[w] -= S1
    return S,b

def _check_nan(T,typ,txt):
    if xp.count_nonzero(xp.isnan(T))>0:
        print(f'{txt} {typ} contains Nan')
        return 1 
    return 0

def check_nan(walkers,txt):
    hasNan = 0 
    hasNan += _check_nan(walkers.phi,'phi',txt)
    if walkers.S is None:
        hasNan += _check_nan(walkers.Sa,'Sa',txt)
        if walkers.Sb is not None:
            hasNan += _check_nan(walkers.Sb,'Sb',txt)
    else:
        hasNan += _check_nan(walkers.S,'S',txt)
    if hasNan > 0:
        exit()

def check_orthonormal(phi,txt):
    if phi is None:
        return
    err =  xp.linalg.norm(xp.einsum('wxi,wxj->wij',phi,phi)-xp.eye(phi.shape[2])[None,:,:])
    if err>1e-10:
        print(f'{txt} orthonormal error=',err)
        exit()

def compute_Cu(phi,rotations):
    for typ in rotations.typs:
        w = rotations.get_itm(typ,'w')
        if w is None:
            continue
        u = rotations.get_itm(typ,'u')
        if typ=='h2ab':
            Cu = [_compute_Cu(phi[s],w,u[s]) for s in (0,1)] 
        else:
            s = {'a':0,'b':1}[typ[-1]]
            Cu = _compute_Cu(phi[s],w,u)
        rotations.add_itm(typ,'Cu',Cu)

def compute_lowdin(rotations,cat_h2ab):
    for typ in rotations.typs:
        w = rotations.get_itm(typ,'w')
        if w is None:
            continue
        Cu = rotations.get_itm(typ,'Cu')
        d2 = rotations.get_itm(typ,'d2')
        if typ=='h2ab':
            if cat_h2ab:
                info = _compute_lowdin(xp.concatenate(Cu,axis=2),d2)
            else:
                d2 = [d2[:,:1],d2[:,1:]]
                info = [_compute_lowdin(Cu[s],d2[s]) for s in (0,1)]
        else:
            info = _compute_lowdin(Cu,d2)
        rotations.add_itm(typ,'lowdin',info)

@plum.dispatch
def compute_intermediates(walkers:UHFWalkers,rotations,lowdin=True):
    check_nan(walkers,'pre update')
    nu,nd = walkers.nup,walkers.ndown
    if nd==0:
        phi = [walkers.phi,None]
    else:
        phi = [walkers.phi[:,:,:nu],walkers.phi[:,:,nu:]]
    #if lowdin:
    #    for phi_i in phi:
    #        check_orthonormal(phi_i,'pre update')

    compute_Cu(phi,rotations)
    if lowdin:
        compute_lowdin(rotations,cat_h2ab=False)

@plum.dispatch
def compute_intermediates(walkers:GHFWalkers,rotations,lowdin=True):
    check_nan(walkers,'pre update')
    nb = walkers.nbasis
    phi = [walkers.phi[:,:nb],walkers.phi[:,nb:]]
    compute_Cu(phi,rotations)
    if lowdin:
        compute_lowdin(rotations,cat_h2ab=True)

def update_ovlp_ratio(S,rotations,walkers_update=True,b=None):
    for typ in rotations.typs:
        w = rotations.get_itm(typ,'w')
        if w is None:
            continue
        Cu = rotations.get_itm(typ,'Cu')
        d = rotations.get_itm(typ,'d')
        uB = rotations.get_itm(typ,'uB')
        if typ=='h2ab':
            S,b = _update_ovlp_2(S,w,Cu,uB,d,b=b,update_S=walkers_update)
        else:
            s = {'a':0,'b':1}[typ[-1]]
            S,b = _update_ovlp_1(S,w,Cu,uB[s],d,s,b=b,update_S=walkers_update)
    return S,b

@plum.dispatch
def update_walkers(walkers:UHFWalkers,rotations,walkers_update=True,b=None,lowdin=True,eps_sq=None):

    S = [walkers.Sa,walkers.Sb] if walkers.S is None else walkers.S
    S,b = update_ovlp_ratio(S,rotations,walkers_update=walkers_update,b=b)
    if not walkers_update:
        return b

    nu,nd = walkers.nup,walkers.ndown
    if nd==0:
        phi = [walkers.phi,None]
    else:
        phi = [walkers.phi[:,:,:nu],walkers.phi[:,:,nu:]]

    for typ in rotations.typs:
        w = rotations.get_itm(typ,'w')
        if w is None:
            continue
        d = rotations.get_itm(typ,'d')
        u = rotations.get_itm(typ,'u')
        Cu = rotations.get_itm(typ,'Cu')
        if lowdin:
            info = rotations.get_itm(typ,'lowdin')
        if typ!='h2ab':
            s = {'a':0,'b':1}[typ[-1]]
            phi[s] = _update_walkers(phi[s],w,d,u,Cu)
            #_check_nan(phi[s][w],'phi',typ)
            if lowdin:
                phi[s],w = _update_walkers_lowdin(phi[s],w,info)
                #_check_nan(phi[s][w],'phi',typ+' lowdin')
                S = _update_ovlp_lowdin(S,w,info,s)
            continue
        d = [d[:,:1],d[:,1:]]
        for s in (0,1):
            phi[s] = _update_walkers(phi[s],w,d[s],u[s],Cu[s])
            #_check_nan(phi[s][w],'phi',typ)
            if lowdin:
                phi[s],w = _update_walkers_lowdin(phi[s],w,info[s])
                #_check_nan(phi[s][w],'phi',typ+' lowdin')
                S = _update_ovlp_lowdin(S,w,info[s],s)

    if isinstance(S,list):
        walkers.Sa = S[0]
        walkers.Sb = S[1]
    else:
        walkers.S = S

    #if lowdin: 
    #    for phi_i in phi:
    #        check_orthonormal(phi_i,'post update')
    nu,nd = walkers.nup,walkers.ndown
    walkers.phi[:,:,:nu] = phi[0]
    if nd>0: 
        walkers.phi[:,:,nu:] = phi[1]
    check_nan(walkers,'post update')
    return b

@plum.dispatch
def update_walkers(walkers:GHFWalkers,rotations,walkers_update=True,b=None,lowdin=True,eps_sq=None):
    walkers.S,b = update_ovlp_ratio(walkers.S,rotations,walkers_update=walkers_update,b=b)
    if not walkers_update:
        return b
    nb = walkers.nbasis
    phi = walkers.phi

    for typ in rotations.typs:
        w = rotations.get_itm(typ,'w')
        if w is None:
            continue
        d = rotations.get_itm(typ,'d')
        u = rotations.get_itm(typ,'u')
        Cu = rotations.get_itm(typ,'Cu')
        if lowdin:
            info = rotations.get_itm(typ,'lowdin')
        if typ=='h2ab':
            d = [d[:,:1],d[:,1:]]
            phi[:,:nb] = _update_walkers(phi[:,:nb],w,d[0],u[0],Cu[0]) 
            phi[:,nb:] = _update_walkers(phi[:,nb:],w,d[1],u[1],Cu[1]) 
        else:
            if typ[-1]=='a':
                phi[:,:nb] = _update_walkers(phi[:,:nb],w,d,u,Cu)
            else:
                phi[:,nb:] = _update_walkers(phi[:,nb:],w,d,u,Cu)

        if lowdin:
            phi,w = _update_walkers_lowdin(phi,w,info)
            walkers.S = _update_ovlp_lowdin(walkers.S,w,info,None)

    walkers.phi =  phi
    check_nan(walkers,'post update')
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



    #p = Cu.copy()
    #s = xp.zeros((p.shape[0],2,2))
    #s[:,0,0] = xp.sqrt((Cu[:,:,0]**2).sum(axis=1))
    #idx = xp.nonzero(norm<thresh)[0]
    #assert idx.size==0
    #p[:,:,0] /= s[:,0,0]

    #s[:,0,1] = xp.einsum('wi,wi->w',p[:,:,0],Cu[:,:,1])
    #p[:,:,1] -= Cu[:,:,1] - p0*s01[:,None]
    #s11 = xp.sqrt((p11**2).sum(axis=1))
    #if xp.linalg.norm(s11)<thresh:
    #    s = xp.zeros((w.size,2,1))
    #    s[:,0,0]

