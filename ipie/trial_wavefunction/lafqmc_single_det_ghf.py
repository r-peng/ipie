import numpy
from ipie.trial_wavefunction.lafqmc_single_det import SingleDet,compute_density
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
        self.density = None

    def get_psi(self):
        nb = self.nbasis
        return [self.psi[:nb],self.psi[nb:]]

    def compute_density(self,s,X=None,diag=True,backend='numpy'):
        if backend=='numpy':
            xp_ = numpy
        else:
            xp_ = xp
        nb1 = self.nbasis

        if self.density is None:
            self.density = compute_density(self.psi,xp_)

        if X is None:
            if diag:
                if s==0:
                    return xp_.diag(self.density)[:nb1]
                else:
                    return xp_.diag(self.density)[nb1:]
            else:
                if s==0:
                    return self.density[:nb1,:nb1]
                else:
                    return self.density[nb1:,nb1:]

        X = xp_.asarray(X)
        nb2 = X.shape[1]
        XD = xp.zeros((nb2*2,nb1*2))
        XD[:nb2] = xp_.dot(X.T,self.density[:nb1])
        XD[nb2:] = xp_.dot(X.T,self.density[nb1:])
        XDX = xp.zeros((nb2*2,nb2*2))
        XDX[:,:nb2] = xp_.dot(XD[:,:nb1],X)
        XDX[:,nb2:] = xp_.dot(XD[:,nb1:],X)
        if s==0:
            rho = XDX[:nb2,:nb2]
        else:
            rho = XDX[nb2:,nb2:]
        if diag:
            rho = xp_.diag(rho)
        return rho 
