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

    def build(self,hamiltonian,psi=None):
        if psi is None:
            psi = self.psi
        U = hamiltonian.chol_basis
        self.UB = [xp.einsum('dxp,xi->dpi',U,Bi) for Bi in psi]

    def calc_force_bias(self, hamiltonian, walkers, mpi_handler):
        pass

    def calc_greens_function(self, walkers): 
        pass

    def calc_overlap(self, walkers):
        pass

    def half_rotate(self, hamiltonian, comm):
        pass
