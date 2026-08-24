import numpy as np
import plum
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.utils.backend import arraylib as xp
from ipie.walkers.lafqmc_uhf_walkers import (
        UHFWalkers,
        qr,
)

class GHFWalkers(UHFWalkers):

    def get_phi(self):
        phi = [self.phi[:,:self.nbasis],self.phi[:,self.nbasis:]]
        return phi

    def set_phi(self,phi):
        self.phi[:,:self.nbasis] = phi[0]
        self.phi[:,self.nbasis:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDet,**kwargs):
        raise NotImplementedError

    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF,set_attribute=True,set_buff=False):
        CB = xp.einsum('wxi,xj->wij',self.phi,trial.psi)
        S = xp.linalg.inv(CB)
        if set_attribute:
            self.S = S
        if set_buff:
            self.buff_names += ['S']
        return S

    @plum.dispatch
    def compute_density(self,hamiltonian,trial:SingleDet,set_buff=True):
        raise NotImplementedError

    @plum.dispatch
    def compute_density(self,hamiltonian,trial:SingleDetGHF,set_buff=True):
        S = self.compute_S(trial)

        nchol = hamiltonian.nchol 
        U = hamiltonian.chol_basis
        nb = self.nbasis
        UC = xp.zeros((self.nwalkers,nchol,nb*2,self.nelec))
        UC[:,:,:nb] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,:nb])
        UC[:,:,nb:] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,nb:])
        self.SCU = xp.einsum('wij,wdpj->wdip',S,UC)

        self.UDU = xp.zeros((self.nwalkers,hamiltonian.nchol,nb*2,nb*2))
        self.UDU[:,:,:nb] = xp.einsum('dpi,wdiq->wdpq',trial.UB[0],self.SCU)
        self.UDU[:,:,nb:] = xp.einsum('dpi,wdiq->wdpq',trial.UB[1],self.SCU)

        if set_buff:
            self.buff_names = ['SCU','UDU']

    def update_UBS_2(self,key,w,i,uC):
        M = self.M2[key][i,w]
        uC = xp.concatenate(uC,axis=1)
        MuC = xp.einsum('wri,wrs->wsi',uC,M)

        left = UBS_dot_Cu(self.UBS[w],MuC,None)
        uBS = self.uBS[key]
        uBS = [uBSi[i,w] for uBSi in uBS]
        uBS = xp.concatenate(uBS,axis=1)
        self.UBS[w] = update_UBS(self.UBS[w],left,uBS)

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,d,uC,trial:SingleDet,b):
        raise NotImplementedError

    @plum.dispatch
    def update_ovlp_1(self,key,w,p,d,uC,trial:SingleDetGHF,b):
        chol_ix,spin = key
        s = spin[0]

        uB = trial.UB[s][chol_ix,p] 
        uBS = xp.einsum('wri,wij->wrj',uB,self.S[w])
        SCu = xp.einsum('wij,wrj->wir',self.S[w],uC)

        M = xp.einsum('wri,wsi->wrs',uBS,uC)
        M = xp.eye(p.shape[1])[None,:,:] + d[:,:,None]*M
        b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,:]
        right = xp.einsum('wrs,wsj->wrj',M,uBS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SCu,right)
        return b 

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,d,uC,trial:SingleDet,b):
        raise NotImplementedError

    @plum.dispatch
    def update_ovlp_2(self,key,w,p,d,uC,trial:SingleDetGHF,b):
        chol_ix,spin = key
        p = [p[:,:1],p[:,1:]]
        uB = xp.concatenate([trial.UB[s][chol_ix,p[s]] for s in (0,1)],axis=1)
        uBS = xp.einsum('wri,wij->wrj',uB,self.S[w])
        uC = xp.concatenate(uC,axis=1)
        SCu = xp.einsum('wij,wrj->wir',self.S[w],uC)

        M = xp.einsum('wri,wsi->wrs',uBS,uC)
        M = xp.eye(2)[None,:,:] + d[:,:,None]*M
        b[w] *= xp.linalg.det(M)

        M = xp.linalg.inv(M) * d[:,None,:]
        right = xp.einsum('wrs,wsj->wrj',M,uBS)
        self.S[w] -= xp.einsum('wir,wrj->wij',SCu,right)
        return b 

    def reortho(self,trial):
        self.phi = qr(self.phi)
        if 'S' in self.buff_names:
            self.compute_S(trial)

    @plum.dispatch
    def compute_SC(self,trial:SingleDet):
        raise NotImplementedError

    @plum.dispatch
    def compute_SC(self,trial:SingleDetGHF):
        phi = self.get_phi()
        return [xp.einsum('wij,wxj->wix',self.S,Ci) for Ci in phi]

    def _load_phi(self,phi):
        self.phi = xp.asarray(phi)
