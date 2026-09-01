import numpy
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler

# class for UHF trial
class SingleDet(TrialWavefunctionBase):

    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")

        self.psi = [wavefunction[:, : self.nalpha],wavefunction[:, self.nalpha :]]
        self.handler = handler

    def get_psi(self):
        return self.psi 

    def compute_density(self,s,U=None,diag=True,backend='numpy'):
        if backend=='numpy':
            xp_ = numpy
        else:
            xp_ = xp

        psi = self.get_psi()[s]
        if U is not None:
            psi = xp_.dot(xp_.asarray(U).T,psi)
        S = xp_.dot(psi.T,psi)
        Sinv = xp_.linalg.inv(S)
        D = xp_.dot(psi,Sinv)
        if diag:
            D = xp_.einsum('pi,pi->p',D,psi)
        else:
            D = xp_.dot(D,psi.T)
        return D

    def build(self,hamiltonian,conjugate=False):
        psi = self.get_psi()
        U = hamiltonian.chol_basis
        self.UB = [xp.einsum('dxp,xi->dpi',U,Bi) for Bi in psi]
        if not conjugate:
            return
        self.hB = [xp.einsum('xy,yi->xi',hamiltonian.h1e,Bi) for Bi in psi]
        if hamiltonian.chol is not None:
            self.LB = [xp.einsum('dxy,yi->dxi',hamiltonian.chol,Bi) for Bi in psi]

    def calc_force_bias(self, hamiltonian, walkers, mpi_handler):
        pass

    def calc_greens_function(self, walkers): 
        pass

    def calc_overlap(self, walkers):
        pass

    def half_rotate(self, hamiltonian, comm):
        pass

    
