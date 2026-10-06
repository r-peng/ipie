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

    def compute_density(self,s,X=None,diag=True,backend='numpy'):
        if backend=='numpy':
            xp_ = numpy
        else:
            xp_ = xp
        nb1 = self.nbasis

        if X is None:
            XB = self.psi
            nb2 = nb1
        else:
            X = xp_.asarray(X)
            nb2 = X.shape[1]
            XB = xp.zeros((nb2*2,self.psi.shape[1]))
            XB[:nb2] = xp_.dot(X.T,self.psi[:nb1])
            XB[nb2:] = xp_.dot(X.T,self.psi[nb1:])
        S = xp_.dot(XB.T,XB)
        Sinv = xp_.linalg.inv(S)
        D = xp_.dot(XB,Sinv)
        if diag:
            D = xp_.einsum('pi,pi->p',D,XB)
            if s==0:
                D = D[:nb2]
            else:
                D = D[nb2:]
            print(s,D)
        else:
            D = xp_.dot(D,XB.T)
            if s==0:
                D = D[:nb2,:nb2]
            else:
                D = D[nb2:,nb2:]
        return D
