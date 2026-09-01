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
        nb = self.nbasis
        return [self.psi[:nb],self.psi[nb:]]

    def compute_density(self,s,U=None,diag=True,backend='numpy'):
        if backend=='numpy':
            xp_ = numpy
        else:
            xp_ = xp
        nb = self.nbasis

        if U is None:
            psi = self.psi
        else:
            psi = self.psi.copy()
            U = xp_.asarray(U)
            psi[:nb] = xp_.dot(U.T,psi[:nb])
            psi[nb:] = xp_.dot(U.T,psi[nb:])
        S = xp_.dot(psi.T,psi)
        Sinv = xp_.linalg.inv(S)
        D = xp_.dot(psi,Sinv)
        if diag:
            D = xp_.einsum('pi,pi->p',D,psi)
            if s==0:
                D = D[:nb]
            else:
                D = D[nb:]
            print(s,D)
        else:
            D = xp_.dot(D,psi.T)
            if s==0:
                D = D[:nb,:nb]
            else:
                D = D[nb:,nb:]
        return D
