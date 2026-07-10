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
    def compute_S(self,trial:SingleDet):
        raise NotImplementedError

    @plum.dispatch
    def compute_S(self,trial:SingleDetGHF):
        CB = xp.einsum('wxi,xj->wij',self.phi,trial.psi)
        return xp.linalg.inv(CB)

    @plum.dispatch
    def build(self,hamiltonian,trial:SingleDetGHF):
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

        self.buff_names = ['phi','weight','phase','SCU','UDU']
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = xp.zeros(self.buff_size)

    def update_UBS_2(self,key,w,i,uC):
        M = self.M2[key][i,w]
        uC = xp.concatenate(uC,axis=1)
        MuC = xp.einsum('wri,wrs->wsi',uC,M)

        left = UBS_dot_Cu(self.UBS[w],MuC,None)
        uBS = self.uBS[key]
        uBS = [uBSi[i,w] for uBSi in uBS]
        uBS = xp.concatenate(uBS,axis=1)
        self.UBS[w] = update_UBS(self.UBS[w],left,uBS)

    def reortho(self):
        self.phi = qr(self.phi)
