import numpy
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet

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

    def compute_UB(self,hamiltonian):
        nchol = hamiltonian.nchol 
        U = hamiltonian.chol_basis
        nb = self.nbasis
        nelec = self.nalpha+self.nbeta
        UB = xp.zeros((nchol,nb*2,nelec))
        UB[:,:nb] = xp.einsum('dxp,xi->dpi',U,self.psi[:nb])
        UB[:,nb:] = xp.einsum('dxp,xi->dpi',U,self.psi[nb:])
        return UB

