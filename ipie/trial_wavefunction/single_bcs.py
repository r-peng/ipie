
import time
from typing import Tuple, Union

import numpy
import plum

from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.estimators.generic import cholesky_jk_ghf
from ipie.estimators.greens_function_single_det import greens_function_single_det_ghf
from ipie.estimators.utils import gab_mod
from ipie.propagation.overlap import calc_overlap_single_det_ghf
from ipie.propagation.force_bias import construct_force_bias_batch_single_det


class SingleBCS(TrialWavefunctionBase):
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
        self.psi = occupied
        self.pairing = pairing
        self._num_dets = 1
        self._max_num_dets = 1
