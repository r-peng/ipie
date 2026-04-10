# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

"""Routines and classes for estimation of observables."""

from __future__ import print_function

import os
from typing import Tuple, Union

import h5py
import numpy

from ipie.config import config, MPI
from ipie.estimators.energy_custom import EnergyEstimator
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.handler import EstimatorHandler as EstimatorHandler_
from ipie.estimators.utils import H5EstimatorHelper
from ipie.utils.io import format_fixed_width_strings

# Some supported (non-custom) estimators
_predefined_estimators = {
    "energy": EnergyEstimator,
}


class EstimatorHandler(EstimatorHandler_):
    def __init__(
        self,
        comm,
        system,
        hamiltonian,
        trial,
        walker_state=None,
        verbose: bool = False,
        filename: Union[str, None] = None,
        block_size: int = 1,
        basename: str = "estimates",
        overwrite=True,
        observables: Tuple[str] = ("energy",),  # TODO: Use factory method!
        index: int = 0,
    ):
        if verbose:
            print("# Setting up estimator object.")
        if comm.rank == 0:
            self.basename = basename
            self.filename = filename
            self.index = 0
            if self.filename is None:
                self.filename = f"{self.basename}.{self.index}.h5"
                while os.path.isfile(self.filename) and not overwrite:
                    self.index = int(self.filename.split(".")[1])
                    self.index = self.index + 1
                    self.filename = f"{self.basename}.{self.index}.h5"
            if verbose:
                print(f"# Writing estimator data to {self.filename}")
        else:
            self.filename = None
        self.buffer_size = config.get_option("estimator_buffer_size")
        if walker_state is not None:
            self.num_walker_props = walker_state.size
            self.walker_header = walker_state.names
        else:
            self.num_walker_props = 0
            self.walker_header = ""
        self._estimators = {}
        self._shapes = []
        self._offsets = {}
        self.json_string = "{}"
        # TODO: Replace this, should be built outside
        for obs in observables:
            try:
                est = _predefined_estimators[obs](
                    system=system,
                    ham=hamiltonian,
                    trial=trial,
                )
                self[obs] = est
            except KeyError:
                raise RuntimeError(f"unknown observable: {obs}")
        if verbose:
            print("# Finished settting up estimator object.")

        self.log_average_weights = []
    def save_log_average_weights(self,dirname='.'):
        if dirname is None:
            return
        numpy.save(f'{dirname}/log_average_weights.npy',numpy.array(self.log_average_weights))
    def load_log_average_weights(self,dirname,rank):
        if rank>0:
            return
        self.log_average_weights = list(numpy.load(f'{dirname}/log_average_weights.npy'))
    def post_sr(self,comm,walker_factors,log_average_weight,dirname='.'):
        self.local_estimates[: walker_factors.size] = walker_factors.buffer
        comm.Reduce(self.local_estimates, self.global_estimates, op=MPI.SUM)
        if comm.rank > 0:
            self.zero()
            return
        self.log_average_weights.append(log_average_weight)
        self.save_log_average_weights(dirname)
        # Get walker data.
        offset = walker_factors.size
        #walker_factors.post_reduce_hook(self.global_estimates[:offset], 0)
        for k, e in self.items():
            start = offset + self.get_offset(k)
            end = start + int(self[k].size)
            est_data = self.global_estimates[start:end]
            e.process_sr_data(est_data,dirname)
        self.zero()
    def print_block_sr(self,comm,block,walker_factors,max_nprod,max_nsum):
        if comm.rank>0:
            return
        ntot = len(self.log_average_weights)
        nprod = min(max_nprod,ntot-1)
        nsum = min(ntot-nprod,max_nsum)
        shift = ntot - nsum
        print(f'ntot={ntot},nprod={nprod},nsum={nsum},shift={shift}')
        weight = numpy.zeros(nsum)
        for i in range(nsum):
            stop = i+shift
            start = stop-nprod
            weight[i] = sum(self.log_average_weights[start:stop])
        max_weight = numpy.amax(weight)
        weight = numpy.exp(weight-max_weight)

        output_string = " "
        vals = numpy.mean(weight),max_weight,0.
        output_string += walker_factors.to_text(numpy.array(vals))

        offset = self.num_walker_props
        for k,e in self.items():
            est_data = e.compute_estimator_sr(weight,comm.rank,shift)
            e.post_reduce_hook(est_data)
            est_string = e.data_to_text(est_data)
            e.to_ascii_file(est_string)
            if e.print_to_stdout:
                output_string += est_string

        self.output.push_to_chunk(self.global_estimates, f"data")
        self.output.increment()
        print(f"{block:>17d} " + output_string)
