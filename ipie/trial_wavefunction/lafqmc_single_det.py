import numpy
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler

def compute_density(B,xp_):
    S = xp_.dot(B.T,B)
    Sinv = xp_.linalg.inv(S)
    return xp_.dot(xp_.dot(B,Sinv),B.T)

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
        self.density = [None,None]

    def get_psi(self):
        return self.psi 

    def compute_density(self,s,X=None,diag=True,backend='numpy'):
        if backend=='numpy':
            xp_ = numpy
        else:
            xp_ = xp

        D = self.density[s]
        if D is None:
            B = self.get_psi()[s]
            D = compute_density(B,xp_)
            self.density[s] = D
        if X is None:
            rho = D
        else:
            X = xp_.asarray(X)
            D = xp_.dot(X.T,D)
            rho = xp_.dot(D,X)
        if diag:
            rho = xp_.diag(rho)
        return rho 

    def build(self,hamiltonian):
        psi = self.get_psi()
        self.Bh1 = [xp.einsum('xi,xy->iy',Bi,hamiltonian.h1) for Bi in psi]
        self.BX = dict()
        for key,sg in hamiltonian.size_groups.items():
            self.BX[key] = [xp.einsum('xi,dxp->dip',Bi,sg.basis) for Bi in psi]

    def get_Bv(self,size_key,bix,s,p):
        Bv = self.BX[size_key][s][bix][:,p]
        return Bv.transpose(1,0,2)

    def calc_force_bias(self, hamiltonian, walkers, mpi_handler):
        pass

    def calc_greens_function(self, walkers): 
        pass

    def calc_overlap(self, walkers):
        pass

    def half_rotate(self, hamiltonian, comm):
        pass

    
