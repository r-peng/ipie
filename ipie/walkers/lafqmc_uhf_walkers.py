import numpy as np
import plum,h5py
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.walkers.base_walkers import BaseWalkers 
from ipie.hamiltonians.sor_base import HubbardSOR,QCSOR
from ipie.utils.backend import to_host,cast_to_device
from ipie.utils.backend import arraylib as xp

def qr(phi,thresh=1e-3):
    Q,R = xp.linalg.qr(phi,mode='reduced')
    Rdiag = xp.einsum('wii->wi',R)

    Rabs = xp.fabs(Rdiag)
    assert Rabs[Rabs<thresh].size==0

    sign = xp.sign(Rdiag)
    Q *= sign[:,None,:]
    return Q

def update_phi(C,u,d):
    uC = xp.einsum('xwr,wxi->wri',u,C)
    duC = d[:,:,None]*uC
    C += xp.einsum('xwr,wri->wxi',u,duC)
    return C,uC

def compute_right2(uB,SCU,M1,d):
    right2 = xp.einsum('wri,wdip->wdrp',uB,SCU)
    M2 = xp.linalg.inv(M1) * d[:,None,:]
    return xp.einsum('wrs,wdsp->wdrp',M2,right2),M2

def compute_M3(uDu,M2,d):
    M3 = xp.eye(d.shape[1])[None,:,:]-xp.einsum('wrs,wst->wrt',M2,uDu)
    return M3 * d[:,None,:]

def compute_right3(uDu,M2,d,u2):
    M3 = compute_M3(uDu,M2,d)
    return xp.einsum('wrs,dpws->wdrp',M3,u2)

def update_SCU(SCU,SCu,right):
    SCU += xp.einsum('wri,wdrp->wdip',SCu,right)
    return SCU

def update_UDU(D,UB,SCu,right):
    left = xp.einsum('dpi,wri->wdpr',UB,SCu)
    D += xp.einsum('wdpr,wdrq->wdpq',left,right)
    return D

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
        self.nelec = nup+ndown
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
        self.measure_sign = False

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def get_phi(self):
        nu = self.nup
        return [self.phi[:,:,:nu],self.phi[:,:,nu:]]

    def set_phi(self,phi):
        nu = self.nup
        self.phi[:,:,:nu] = phi[0]
        self.phi[:,:,nu:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDet,set_attribute=True,set_buff=False):
        phi = self.get_phi()
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,trial.psi)] 
        S = [xp.linalg.inv(Si) for Si in CB]
        if set_attribute:
            self.Sa,self.Sb = S
        if set_buff:
            self.buff_names = ['Sa','Sb']
        return S
    
    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF,set_attribute=True,set_buff=False):
        phi = self.get_phi()
        B = [trial.psi[:self.nbasis],trial.psi[self.nbasis:]]
        CB = [xp.einsum('wxi,xj->wij',Ci,Bi) for Ci,Bi in zip(phi,B)] 
        CB = xp.concatenate(CB,axis=1)
        S = xp.linalg.inv(CB)
        if set_attribute:
            self.S = S
        if set_buff:
            self.buff_names = ['S']
        return S

    def compute_density(self,hamiltonian,trial,set_buff=True):
        S = self.compute_S(trial,set_attribute=False,set_buff=False)
        U = hamiltonian.chol_basis
        UC = xp.einsum('dxp,wxi->wdpi',U,self.phi)
        nu = self.nup
        UC = [UC[:,:,:,:nu],UC[:,:,:,nu:]]

        if isinstance(S,list):
            SCU = [xp.einsum('wij,wdpj->wdip',Si,UCi) for Si,UCi in zip(S,UC)]
            self.SCU = xp.concatenate(SCU,axis=2)
            UDU = [xp.einsum('dpi,wdiq->wdpq',UBi,SCUi) for UBi,SCUi in zip(trial.UB,SCU)]
            self.UDU = xp.concatenate(UDU,axis=2)
        else:
            nb = self.nbasis
            self.SCU = xp.zeros((self.nwalkers,hamiltonian.nchol,self.nelec,nb*2))
            self.SCU[:,:,:,:nb] = xp.einsum('wij,wdpj->wdip',S[:,:,:nu],UC[0])
            self.SCU[:,:,:,nb:] = xp.einsum('wij,wdpj->wdip',S[:,:,nu:],UC[1])
            self.UDU = xp.zeros((self.nwalkers,hamiltonian.nchol,nb*2,nb*2))
            self.UDU[:,:,:nb] = xp.einsum('dpi,wdiq->wdpq',trial.UB[0],self.SCU)
            self.UDU[:,:,nb:] = xp.einsum('dpi,wdiq->wdpq',trial.UB[1],self.SCU)
        if set_buff:
            self.buff_names = ['SCU','UDU']

    def build(self,hamiltonian,trial,importance):
        self.importance = importance
        self.has_E12 = False
        if importance:
            self.compute_density(hamiltonian,trial)
            self.E1 = xp.zeros(self.nwalkers) 
            self.E2 = xp.zeros(self.nwalkers) 
            self.buff_names += ['E1','E2']
        else:
            self.compute_S(trial,set_buff=True)
        self.buff_names += ['phi','weight','phase','unscaled_weight','hybrid_energy']

        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = np.zeros(self.buff_size, dtype=np.complex128)

    def get_SCU(self,chol_ix=None):
        nb = self.nbasis
        _,_,sh1,sh2 = self.SCU.shape
        assert sh1==self.nelec
        if sh2==nb:
            nu = self.nup
            SCU = [self.SCU[:,:,:nu],self.SCU[:,:,nu:]]
        elif sh2==nb*2:
            SCU = [self.SCU[:,:,:,:nb],self.SCU[:,:,:,nb:]]
        else:
            raise ValueError
        if chol_ix is not None:
            SCU = [SCUi[:,chol_ix] for SCUi in SCU]
        return SCU

    def set_SCU(self,SCU):
        nb = self.nbasis
        _,_,sh1,sh2 = self.SCU.shape
        assert sh1==self.nelec
        if sh2==nb:
            nu = self.nup
            if SCU[0] is not None:
                self.SCU[:,:,:nu] = SCU[0]
            if SCU[1] is not None:
                self.SCU[:,:,nu:] = SCU[1]
        elif sh2==nb*2:
            if SCU[0] is not None:
                self.SCU[:,:,:,:nb] = SCU[0]
            if SCU[1] is not None:
                self.SCU[:,:,:,nb:] = SCU[1]
        else:
            raise ValueError

    def get_UDU(self,chol_ix=None):
        nb = self.nbasis
        _,_,sh1,sh2 = self.UDU.shape
        assert sh1==self.nbasis*2
        D = dict()
        if sh2==nb:
            D[0,0] = self.UDU[:,:,:nb]
            D[1,1] = self.UDU[:,:,nb:]
        elif sh2==nb*2:
            D[0,0] = self.UDU[:,:,:nb,:nb]
            D[1,1] = self.UDU[:,:,nb:,nb:]
            D[0,1] = self.UDU[:,:,:nb,nb:]
            D[1,0] = self.UDU[:,:,nb:,:nb]
        else:
            raise ValueError
        if chol_ix is not None:
            D = {key:Di[:,chol_ix] for key,Di in D.items()}
        return D

    def set_UDU(self,D):
        nb = self.nbasis
        _,_,sh1,sh2 = self.UDU.shape
        assert sh1==self.nbasis*2
        if sh2==nb:
            if (0,0) in D:
                self.UDU[:,:,:nb] = D[0,0]
            if (1,1) in D:
                self.UDU[:,:,nb:] = D[1,1]
        elif sh2==nb*2:
            if (0,0) in D:
                self.UDU[:,:,:nb,:nb] = D[0,0] 
            if (1,1) in D:
                self.UDU[:,:,nb:,nb:] = D[1,1] 
            if (0,1) in D:
                self.UDU[:,:,:nb,nb:] = D[0,1] 
            if (1,0) in D:
                self.UDU[:,:,nb:,:nb] = D[1,0] 

    def get_uDu(self,key,p):
        chol_ix,spin = key
        D = self.get_UDU(chol_ix)
        if p.shape[1]==1:
            s = spin[0]
            uDu = D[s,s][:,p[:,0],p[:,0]]
            return uDu.reshape(self.nwalkers,p.shape[0],1,1)

        if spin!=(0,1):
            s = spin[0]
            D = {(s1,s2):D[s,s] for s1 in (0,1) for s2 in (0,1)}

        uDu = xp.zeros((self.nwalkers,p.shape[0],2,2))
        for s1 in (0,1):
            for s2 in (0,1):
                if (s1,s2) not in D:
                    continue
                uDu[:,:,s1,s2] = D[s1,s2][:,p[:,s1],p[:,s2]] 
        return uDu 

    def compute_M1(self,key,p,d):
        _,spin = key
        uDu = self.get_uDu(key,p) 

        M = d[None,:,:,None]*uDu + xp.eye(d.shape[1])[None,None,:,:]
        ovlp = xp.linalg.det(M)
        self.M[key] = uDu,M
        return ovlp.T

    def compute_ovlp_ratio(self,ham):
        ovlp = xp.zeros((ham.nterms,self.nwalkers))
        self.M = dict()
        keys = 'p','d','f','ix'
        for key,dat in ham.term_dict.items():
            p,d,f,ix = [dat[k] for k in keys]
            ovlp[ix] = self.compute_M1(key,p,d)/f[:,None]

        self.E1 = xp.dot(ham.a[ham.E1_ixs],ovlp[ham.E1_ixs])
        self.E2 = xp.dot(ham.a[ham.E2_ixs],ovlp[ham.E2_ixs])
        self.has_E12 = True
        return ovlp

    def update_walkers(self,hamiltonian,trial,b=None):
        self.itm = dict()
        self.has_E12 = False 
        for key,ixs in hamiltonian.samples.items():
            w,i = ixs['w'],ixs['i']
            p,d,f,u,u2 = hamiltonian.get_batch_ud(key,i)

            uC = self.update_phi(key,w,u,d)

            chol_ix,spin = key
            if b is None:
               if spin==(0,1):
                   self.update_density_2(key,w,i,p,d,u,u2,trial)
               else:
                   self.update_density_1(key,w,i,p,d,u,u2,trial)
            else:
               if spin==(0,1):
                   b = self.update_ovlp_2(key,w,p,d,uC,trial,b)
               else:
                   b = self.update_ovlp_1(key,w,p,d,uC,trial,b)
               #print(b[w],f)
               #b[w] /= f 
               #print(b[w])
        return b

    def update_phi(self,key,w,u,d):
        _,spin = key
        phi = self.get_phi()
        if spin==(0,1):
            d = [d[:,:1],d[:,1:]]
            u = [u[:,:,:1],u[:,:,1:]]
            uC = [None] * 2
            for s in (0,1):
                phi[s][w],uC[s] = update_phi(phi[s][w],u[s],d[s])
        else:
            s = spin[0]
            phi[s][w],uC = update_phi(phi[s][w],u,d)
        self.set_phi(phi)
        return uC

    @plum.dispatch
    def update_density_1(self,key,w,i,p,d,u,u2,trial:SingleDet):
        chol_ix,spin = key
        s = spin[0]

        SCU = self.get_SCU()[s]
        uDu,M1 = self.M[key]

        right2,M2 = compute_right2(trial.UB[s][chol_ix,p],SCU[w],M1[w,i],d)
        right3 = compute_right3(uDu[w,i],M2,d,u2)
        right = right3 - right2

        SCu = SCU[w[:,None],chol_ix,:,p]
        SCU[w] = update_SCU(SCU[w],SCu,right)

        D = self.get_UDU()[s,s]
        D[w] = update_UDU(D[w],trial.UB[s],SCu,right)
        self.set_SCU({s:SCU,1-s:None})
        self.set_UDU({(s,s):D})

    @plum.dispatch
    def update_density_2(self,key,w,i,p,d,u,u2,trial:SingleDet):
        chol_ix,_ = key
        p = [p[:,:1],p[:,1:]]
        d = [d[:,:1],d[:,1:]]
        u2 = [u2[:,:,:,:1],u2[:,:,:,1:]]

        SCU = self.get_SCU()
        D = self.get_UDU()

        uDu,M1 = self.M[key]
        uDu,M1 = uDu[w,i],M1[w,i]
        uDu = [uDu[:,:1,:1],uDu[:,1:,1:]]
        M1 = [M1[:,:1,:1],M1[:,1:,1:]]
        for s in (0,1):
            right2,M2 = compute_right2(trial.UB[s][chol_ix,p[s]],SCU[s][w],M1[s],d[s])
            right3 = compute_right3(uDu[s],M2,d[s],u2[s])
            right = right3 - right2

            SCu = SCU[s][w[:,None],chol_ix,:,p[s]]
            SCU[s][w] = update_SCU(SCU[s][w],SCu,right)
            D[s,s][w] = update_UDU(D[s,s][w],trial.UB[s],SCu,right)

        self.set_SCU(SCU)
        self.set_UDU(D)

    @plum.dispatch
    def update_density_1(self,key,w,i,p,d,u,u2,trial:SingleDetGHF):
        chol_ix,spin = key
        s = spin[0]
        nu = self.nup

        uB = trial.UB[s][chol_ix,p]
        uDu,M1 = self.M[key]

        right2,M2 = compute_right2(uB,self.SCU[w],M1[w,i],d) 
        right3 = compute_right3(uDu[w,i],M2,d,u2)
        nb = self.nbasis
        right = -right2
        if s==0:
            right[:,:,:,:nb] += right3
        else:
            right[:,:,:,nb:] += right3

        SCu = self.get_SCU(chol_ix)[s][w[:,None],:,p]
        self.SCU[w] = update_SCU(self.SCU[w],SCu,right)

        self.UDU[w,:,:nb] = update_UDU(self.UDU[w,:,:nb],trial.UB[0],SCu,right)
        self.UDU[w,:,nb:] = update_UDU(self.UDU[w,:,nb:],trial.UB[1],SCu,right)

    @plum.dispatch
    def update_density_2(self,key,w,i,p,d,u,u2,trial:SingleDetGHF):
        chol_ix,_ = key
        p = [p[:,:1],p[:,1:]]
        uB = xp.concatenate([trial.UB[s][chol_ix,p[s]] for s in (0,1)],axis=1)
        uDu,M1 = self.M[key]
        right2,M2 = compute_right2(uB,self.SCU[w],M1[w,i],d) 
        right = -right2

        M3 = compute_M3(uDu[w,i],M2,d)
        nb = self.nbasis
        right[:,:,:,:nb] += xp.einsum('wr,dpw->wdrp',M3[:,:,0],u2[:,:,:,0])
        right[:,:,:,nb:] += xp.einsum('wr,dpw->wdrp',M3[:,:,1],u2[:,:,:,1])

        SCU = self.get_SCU(chol_ix)
        SCu = [SCU[s][w[:,None],:,p[s]] for s in (0,1)]
        SCu = xp.concatenate(SCu,axis=1)
        self.SCU[w] = update_SCU(self.SCU[w],SCu,right)

        self.UDU[w,:,:nb] = update_UDU(self.UDU[w,:,:nb],trial.UB[0],SCu,right)
        self.UDU[w,:,nb:] = update_UDU(self.UDU[w,:,nb:],trial.UB[1],SCu,right)

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,d,uC,trial:SingleDet,b):
        chol_ix,spin = key
        s = spin[0]

        uB = trial.UB[s][chol_ix,p] 
        S = [self.Sa,self.Sb][s]
        uBS = xp.einsum('wri,wij->wrj',uB,S[w])
        SCu = xp.einsum('wij,wrj->wir',S[w],uC)

        M = xp.einsum('wri,wsi->wrs',uBS,uC)
        M = xp.eye(p.shape[1])[None,:,:] + d[:,:,None]*M
        b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,:]
        right = xp.einsum('wrs,wsj->wrj',M,uBS)
        S[w] -= xp.einsum('wir,wrj->wij',SCu,right)
        if s==0:
            self.Sa = S
        else:
            self.Sb = S
        return b 

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,d,uC,trial:SingleDetGHF,b):
        chol_ix,spin = key
        s = spin[0]
        uB = trial.UB[s][chol_ix,p] 
        uBS = xp.einsum('wri,wij->wrj',uB,self.S[w])
        if s==0:
            SCu = xp.einsum('wij,wrj->wir',self.S[w,:,:self.nup],uC)
        else:
            SCu = xp.einsum('wij,wrj->wir',self.S[w,:,self.nup:],uC)

        if s==0:
            M = xp.einsum('wri,wsi->wrs',uBS[:,:,:self.nup],uC)
        else:
            M = xp.einsum('wri,wsi->wrs',uBS[:,:,self.nup:],uC)
        M = xp.eye(p.shape[1])[None,:,:] + d[:,:,None]*M
        b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,:]
        right = xp.einsum('wrs,wsj->wrj',M,uBS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SCu,right)
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,d,uC,trial:SingleDet,b):
        chol_ix,_ = key

        p = [p[:,:1],p[:,1:]]
        uB = [trial.UB[s][chol_ix,p[s]] for s in (0,1)]
        S = [self.Sa,self.Sb]
        for s in (0,1):
            uBS = xp.einsum('wi,wij->wj',uB[s][:,0],S[s][w])
            SCu = xp.einsum('wij,wj->wi',S[s][w],uC[s][:,0])

            M = xp.einsum('wi,wi->w',uBS,uC[s][:,0])
            M = d[:,s]*M + 1.
            b[w] *= M

            M = 1./M * d[:,s]
            right = xp.einsum('w,wj->wj',M,uBS)
            S[s][w] -= xp.einsum('wi,wj->wij',SCu,right)

        self.Sa,self.Sb = S
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,d,uC,trial:SingleDetGHF,b):
        chol_ix,spin = key
        p = [p[:,:1],p[:,1:]]
        uB = xp.concatenate([trial.UB[s][chol_ix,p[s]] for s in (0,1)],axis=1)
        uBS = xp.einsum('wri,wij->wrj',uB,self.S[w])
        SCu = [None] * 2
        SCu[0] = xp.einsum('wij,wrj->wir',self.S[w,:,:self.nup],uC[0])
        SCu[1] = xp.einsum('wij,wrj->wir',self.S[w,:,self.nup:],uC[1])
        SCu = xp.concatenate(SCu,axis=2)

        M = [None] * 2 
        M[0] = xp.einsum('wri,wsi->wrs',uBS[:,:,:self.nup],uC[0])
        M[1] = xp.einsum('wri,wsi->wrs',uBS[:,:,self.nup:],uC[1])
        M = xp.concatenate(M,axis=2)
        M = xp.eye(2)[None,:,:] + d[:,:,None]*M
        b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,:]
        right = xp.einsum('wrs,wsj->wrj',M,uBS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SCu,right)
        return b 

    def reortho(self,trial):
        phi = self.get_phi()
        for s in (0,1):
            phi[s] = qr(phi[s])
        self.set_phi(phi)
        if 'S' in self.buff_names or 'Sa' in self.buff_names:
            self.compute_S(trial)

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
    
    def load(self,comm,dirname):
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
        self.phi = np.asarray(phi[start:stop])
        self.weight = np.asarray(weights[start:stop])
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

    def compute_SC(self,trial):
        phi = self.get_phi()
        S = self.S if 'S' in self.buff_names else [self.Sa,self.Sb]
        if isinstance(S,list):
            return [xp.einsum('wij,wxj->wix',Si,Ci) for Si,Ci in zip(S,phi)]
        else:
            nb = self.nbasis
            nu = self.nup
            Sa = xp.einsum('wij,wxj->wix',S[:,:,:nu],phi[0])
            Sb = xp.einsum('wij,wxj->wix',S[:,:,nu:],phi[1])
            return [Sa,Sb]

    def compute_E1(self,trial,SC):
        E1 = [xp.einsum('xi,wix->w',hBi,SCi) for hBi,SCi in zip(trial.hB,SC)]
        return E1[0]+E1[1]

    def compute_chol(self,trial,SC):
        tr = [xp.einsum('dxi,wix->wd',LBi,SCi) for LBi,SCi in zip(trial.LB,SC)]
        E2 = ((tr[0]+tr[1])**2).sum(axis=1)

        SCLB = [xp.einsum('wix,dxj->wdij',SCi,LBi) for SCi,LBi in zip(SC,trial.LB)]
        for s in (0,1):
            E2 -= xp.einsum('wdij,wdji->w',SCLB[s],SCLB[s])
        if SC[0].shape[1]==self.nelec and SC[1].shape[1]==self.nelec:
            E2 -= 2.*xp.einsum('wdij,wdji->w',SCLB[0],SCLB[1])
        return 0.5*E2

    def compute_1rdm_diag(self,trial,SC):
        psi = trial.get_psi()
        Daa = xp.einsum('xi,wix->wx',psi[0],SC[0])
        Dbb = xp.einsum('xi,wix->wx',psi[1],SC[1])

        Dab,Dba = None,None
        if psi[0].shape[1]==self.nelec:
            Dba = xp.einsum('xi,wix->wx',psi[1],SC[0])
            Dab = xp.einsum('xi,wix->wx',psi[0],SC[1])
        return Daa,Dbb,Dab,Dba

    def local_energy_fast(self,ham):
        if not self.has_E12:
            self.compute_ovlp_ratio(ham)
        E1 = ham.asum1 - self.E1*ham.denom
        E2 = ham.asum2 - self.E2*ham.denom
        return E1+E2+ham.const,E1,E2

    @plum.dispatch
    def local_energy(self,ham:HubbardSOR,trial):
        if self.importance:
            return self.local_energy_fast(ham)

        SC = self.compute_SC(trial)
        E1 = self.compute_E1(trial,SC)
        Daa,Dbb,Dab,Dba = self.compute_1rdm_diag(trial,SC)
        E2 = xp.einsum('wp,wp->w',Daa,Dbb)
        if Dab is not None:
            E2 -= xp.einsum('wp,wp->w',Dab,Dba)
        E2 *= ham.hubbard_U
        E = E1+E2
        if ham.compute_local_magnetization is not None:
            i,j = ham.compute_local_magnetization
            E1 = Daa[:,i]-Dbb[:,i]
            E2 = Daa[:,j]-Dbb[:,j]
            E1 = E1**2
            E2 = E2**2
        return E,E1,E2

    @plum.dispatch
    def local_energy(self,ham:QCSOR,trial):
        if self.importance:
            return self.local_energy_fast(ham)

        SC = self.compute_SC(trial)
        E1 = self.compute_E1(trial,SC)
        E2 = self.compute_chol(trial,SC)
        return E1+E2,E1,E2

    def _measure_sign(self,hamiltonian,trial):
        self.compute_density(hamiltonian,trial,set_buff=False)
        ovlp = self.compute_ovlp_ratio(hamiltonian)
        g = hamiltonian.a[:,None] * ovlp
        gsum = g.sum(axis=0)
        bsum = xp.fabs(g).sum(axis=0)

        b_plus = g.copy()
        xp.clip(b_plus, a_min=0.0, a_max=None, out=b_plus)  
        b_plus = b_plus.sum(axis=0)

        b_minus = g.copy()
        xp.clip(b_minus, a_min=None, a_max=0.0, out=b_minus)  
        b_minus = b_minus.sum(axis=0)
        b_minus *= -1

        err = xp.linalg.norm(b_plus+b_minus-bsum)
        if err>1e-10:
            print(err)
            exit()
        err = xp.linalg.norm(b_plus-b_minus-gsum)
        if err>1e-10: 
            print(err)
            exit()
        f = b_minus / bsum
        s = xp.fabs(gsum) / bsum
        return f,s
