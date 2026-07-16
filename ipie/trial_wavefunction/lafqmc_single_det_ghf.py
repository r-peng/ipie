import numpy
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler

# class for GHF trial
class SingleDetGHF(SingleDet):

    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")

        self.psi = wavefunction
        self.handler = handler

    def get_psi(self):
        return [self.psi[:self.nbasis],self.psi[self.nbasis:]]

