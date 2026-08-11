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

    def rotate_trial(self,U):
        self.psi = xp.dot(U.T,self.psi)

    def build(self,hamiltonian,conjugate=False):
        if hamiltonian.exact_1body:
            self.rotate_trial(hamiltonian.vk1)

        psi = self.get_psi()
        U = hamiltonian.chol_basis
        self.UB = [xp.einsum('dxp,xi->dpi',U,Bi) for Bi in psi]
        if not conjugate:
            return
        if hamiltonian.exact_1body:
            self.hB = [hamiltonian.ek1[:,None]*Bi for Bi in psi]
        else:
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

    
