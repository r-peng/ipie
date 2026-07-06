import numpy
import plum
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp

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

    def build_hamiltonian(self,hamiltonian):
        U = hamiltonian.chol_basis
        h1e = hamiltonian.h1e
        self.UB = [xp.einsum('dxp,xi->dpi',U,Bi) for Bi in self.psi]
        self.h1B = [xp.dot(h1e,Bi) for Bi in self.psi]

    def get_uB(self,key,p):
        chol_idx,typ = key
        uB = [self.UB[0][chol_idx],self.UB[1][chol_idx]]

        if typ=='h2ab':
            p = [p[:,:1],p[:,1:]]
            return [xp.asarray([uB[s][pi] for pi in p[s]]) for s in (0,1)]
        else:
            s = {'a':0,'b':1}[typ[-1]]
            return xp.asarray(uB[s][pi] for pi in p])

