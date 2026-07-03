import numpy
import plum
from ipie.utils.backend import arraylib as xp
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.trial_wavefunction.lafqmc_single_det import (
        compute_UB,
        compute_h1B,
)

# class for GHF trial
class SingleDetGHF(TrialWavefunctionBase):
    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")

        self.psi = wavefunction
        self.handler = handler

    def build_hamiltonian(self,hamiltonian):
        psi = [self.psi[:self.nbasis],self.psi[self.nbasis:]]
        self.UB = [compute_UB(Bi,hamiltonian.chol_basis) for Bi in psi]
        self.h1B = [compute_h1B(Bi,hamiltonian.h1e) for Bi in psi]
