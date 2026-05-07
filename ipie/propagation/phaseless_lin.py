import math
import time

import numpy

from ipie.utils.pack_numba import unpack_VHS_batch

try:
    from ipie.utils.pack_numba_gpu import unpack_VHS_batch_gpu
except:
    pass

import plum

from ipie.config import config
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from typing import Union


class PhaselessLin(PhaselessBase):

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        pass

    def propagate_walkers(self, walkers, hamiltonian, trial, constraint_path=True):
        # 1.compute dressed gf
        synchronize()
        start_time = time.time()
        gf = hamiltonian.calc_gf(walkers,trial)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2.update walker
        keys,b = hamiltonian.sample_from_gf(gf)
        start_time = time.time()
        hamiltonian.update_walkers(keys,walkers)
        synchronize()
        self.timer.tgemm += time.time() - start_time

        # 3.update weight
        start_time = time.time()
        if constraint_path:
            nminus = len(b[b<0])
            if nminus>0:
                print('number of minus=',nminus)
            xp.clip(b,a_min=0.,a_max=None,out=b) 
        walkers.weight += xp.log(xp.fabs(b))
        # use .sgn_ovlp to store signs for now
        # doesn't seemed to be used for anything else
        walkers.sgn_ovlp *= xp.sign(b) 
        synchronize()
        self.timer.tupdate += time.time() - start_time
    def apply_VHS(self, walkers, hamiltonian, xshifted):
        pass
