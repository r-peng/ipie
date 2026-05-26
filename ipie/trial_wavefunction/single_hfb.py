
import time
from typing import Tuple, Union

import numpy
import plum

from ipie.utils.backend import arraylib as xp
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase

class SingleHFB(TrialWavefunctionBase):
    # `num_basis`is # of spin-less AOs.
    @plum.dispatch
    def __init__(
        self,
        occupied: numpy.ndarray,
        pairing: numpy.ndarray,
        num_basis: int,
        verbose: bool = False,
    ):
        assert len(occupied.shape) == 2
        assert len(pairing.shape) == 2
        assert occupied.shape[0] // 2 == num_basis
        super().__init__(None, (None,None), num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.SingleDetGHF.")
        self.B = occupied
        self.Z = pairing
        self.nocc = occupied.shape[1] 
        self._num_dets = 1
        self._max_num_dets = 1

    def build(self) -> None:
        pass

    def calc_force_bias(self,hamiltonian,walkers,mpi_handler=None):
        pass

    def calc_greens_function(self, walkers, build_full: bool = False):
        pass

    def calc_overlap(self, walkers):
        pass

    def half_rotate(self, hamiltonian, comm):
        pass
