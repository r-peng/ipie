import numpy as np
from ipie.utils.backend import to_host
from ipie.utils.backend import arraylib as xp
from ipie.walkers.lafqmc_uhf_walkers import (
        UHFWalkers,
        qr,
        det_ovlp,
)

class GHFWalkers(UHFWalkers):

    @plum.dispatch
    def build_trial(self,trial:SingleDetGHF,set_buffnames=False):
        CB = det_ovlp(self.phi,trial.psi)
        self.S = xp.linalg.inv(CB)
        if set_buffnames:
            self.buff_names += ['S']

    def build_hamiltonian(self,hamiltonian):
        nchol = hamiltonian.nchol 
        U = hamiltonian.chol_basis
        nb = self.nbasis
        nelec = self.nup+self.ndown
        self.UC = xp.zeros((self.nwalkers,nchol,nb*2,nelec))
        self.UC[:,:,:nb] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,:nb])
        self.UC[:,:,nb:] = xp.einsum('dxp,wxi->wdpi',U,self.phi[:,nb:])

    def compute_SC(self):
        C = [self.phi[:,:self.nbasis],self.phi[:,self.nbasis:]]
        return [_SC(self.S,Ci) for Ci in C]

    def reortho(self):
        self.phi = qr(self.phi)
