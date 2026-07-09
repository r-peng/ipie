import numpy as np
import plum
from ipie.trial_wavefunction.lafqmc_single_det_ghf import SingleDetGHF
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp
from ipie.walkers.lafqmc_uhf_walkers import (
        UHFWalkers,
        qr,
        UBS_dot_Cu,
        update_UBS,
        compute_lowdin,
        lowdin_phi,
        compute_1rdm_diag,
)

class GHFWalkers(UHFWalkers):

    def get_phi(self):
        phi = [self.phi[:,:self.nbasis],self.phi[:,self.nbasis:]]
        return phi

    def set_phi(self,phi):
        self.phi[:,:self.nbasis] = phi[0]
        self.phi[:,self.nbasis:] = phi[1]

    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF):
        CB = xp.einsum('wxi,xj->wij',self.phi,trial.psi)
        return xp.linalg.inv(CB)

    def compute_UC(self,hamiltonian):
        nchol = hamiltonian.nchol 
        U = hamiltonian.chol_basis
        nb = self.nbasis
        nelec = self.nup+self.ndown
        self.UC = xp.zeros((self.nwalkers,nchol,nb*2,nelec))
        self.UC[:,:,:nb] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,:nb])
        self.UC[:,:,nb:] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,nb:])

    def get_UC(self):
        UC = [self.UC[:,:,:self.nbasis],self.UC[:,:,self.nbasis:]]
        return UC

    def set_UC(self,UC):
        self.UC[:,:,:self.nbasis] = UC[0]
        self.UC[:,:,self.nbasis:] = UC[1]

    def update_UBS_2(self,key,w,i,uC):
        M = self.M2[key][i,w]
        uC = xp.concatenate(uC,axis=1)
        MuC = xp.einsum('wri,wrs->wsi',uC,M)

        left = UBS_dot_Cu(self.UBS[w],MuC,None)
        uBS = self.uBS[key]
        uBS = [uBSi[i,w] for uBSi in uBS]
        uBS = xp.concatenate(uBS,axis=1)
        self.UBS[w] = update_UBS(self.UBS[w],left,uBS)

    def compute_1rdm_diag(self,chol_idx):
        phi = self.get_phi()
        BS = [UBSi[:,chol_idx] for UBSi in self.get_UBS()]
        D = dict()
        for s1 in (0,1):
            for s2 in (0,1):
                D[s1,s2] = compute_1rdm_diag(BS[s1],phi[s2])
        return D

    def _compute_lowdin(self,key,d,uC):
        _,spin = key
        if spin==(0,1):
            uC = xp.concatenate(uC,axis=1)
        return compute_lowdin(d,uC) 

    def lowdin_phi(self,key,w,q,delta):
        _,spin = key
        self.phi[w] = lowdin_phi(self.phi[w],q,delta)

    def compute_bands_UBS(self,hamiltonian):
        nchol = hamiltonian.nchol 
        Sigma = hamiltonian.chol_bands
        nb = self.nbasis
        nelec = self.nup+self.ndown
        UBS = xp.zeros((self.nwalkers,nchol,nb*2,nelec))
        UBS[:,:,:nb] = Sigma[:,None,:,None]*self.UBS[:,:,:nb]
        UBS[:,:,nb:] = Sigma[:,None,:,None]*self.UBS[:,:,nb:]
        return UBS

    def reortho(self):
        self.phi,self.UC = qr(self.phi,self.UC)
