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
from ipie.utils.backend import to_host
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from typing import Union

def sample_from_gf(gf):
    nkeys,nw = gf.shape
    sign = xp.sign(gf)
    #nminus = int(to_host(xp.sum(sign < -0.5)))
    #if nminus>0:
    #    print('number of minus=',nminus)
    #    #exit()

    p = xp.fabs(gf)
    b = p.sum(axis=0)
    p /= b.reshape(1,nw)
    cdf = xp.cumsum(p, axis=0)
    sample = xp.random.random(nw)
    kixs = xp.sum(cdf < sample.reshape(1,nw), axis=0).astype(xp.int64)
    kixs = xp.minimum(kixs, nkeys - 1)
    sign = sign[kixs, xp.arange(nw)]
    return kixs,b,sign

@plum.dispatch
def walkers_multiply_sign(walkers:UHFWalkers,minus_idx):
    walkers.phia[minus_idx,:,0] *= -1

@plum.dispatch
def walkers_multiply_sign(walkers:GHFWalkers,minus_idx):
    walkers.phi[minus_idx,:,0] *= -1

def update_weight_1(walkers,b,sign,constraint_path=True,thresh=1e-15):
    minus_idx = xp.nonzero(sign<-0.5)
    minus_val = -1
    zero_idx = xp.nonzero(b<thresh)
    if constraint_path:
        b[minus_idx] = thresh
        minus_val = 0
    walkers.weight += xp.log(b)
    # use .sgn_ovlp to store signs for now
    # doesn't seemed to be used for anything else
    walkers.sgn_ovlp[minus_idx] *= minus_val 
    walkers.sgn_ovlp[zero_idx] = 0

def update_weight_2(walkers,b,sign):
    walkers.weight += xp.log(b)
    minus_idx = xp.nonzero(sign<-0.5)
    walkers_multiply_sign(walkers,minus_idx)

def update_weight(walkers,b,sign,routine=2,constraint_path=True,thresh=1e-15):
    if routine==1:
        update_weight_1(walkers,b,sign,constraint_path=constraint_path,thresh=thresh)
    elif routine==2:
        update_weight_2(walkers,b,sign)
    else:
        raise ValueError

class PhaselessLin(PhaselessBase):

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        pass

    def propagate_walkers(self, walkers, hamiltonian, trial, update_weight_routine=2,constraint_path=True):
        # 1.compute dressed gf
        synchronize()
        start_time = time.time()
        gf = hamiltonian.calc_gf(walkers,trial)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2.update walker
        kixs,b,sign = sample_from_gf(gf)
        start_time = time.time()
        hamiltonian.update_walkers(kixs,walkers)
        synchronize()
        self.timer.tgemm += time.time() - start_time

        # 3.update weight
        start_time = time.time()
        update_weight(walkers,b,sign,routine=update_weight_routine,constraint_path=constraint_path)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def apply_VHS(self, walkers, hamiltonian, xshifted):
        pass
