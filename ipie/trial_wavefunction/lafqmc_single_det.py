import numpy
import plum
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp

def compute_UB(B,chol_basis):
    if B is None:
        return None
    return xp.einsum('dxp,xi->dpi',chol_basis,B)

def compute_h1B(B,h1e):
    if B is None:
        return None
    return xp.dot(h1e,B)

# class for UHF trial
class SingleDet(TrialWavefunctionBase):
    def __init__(self, wavefunction, num_elec, num_basis, handler=MPIHandler(), verbose=False):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 2
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")

        self.psi = [wavefunction[:, : self.nalpha],None]
        if self.nbeta>0:
            self.psi[1] = wavefunction[:, self.nalpha :]
        self.handler = handler

    def build_hamiltonian(self,hamiltonian):
        self.UB = [_UB(Bi,hamiltonian.chol_basis) for Bi in self.psi]
        self.h1B = [_h1B(Bi,hamiltonian.h1e) for Bi in self.psi]

    def get_uB(self,p,
