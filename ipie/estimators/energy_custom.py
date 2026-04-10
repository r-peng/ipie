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
# Author: Fionn Malone <fmalone@google.com>
#

from typing import Union

import plum,h5py

from ipie.utils.backend import arraylib as xp
from ipie.estimators.energy import EnergyEstimator as EnergyEstimator_

class EnergyEstimator(EnergyEstimator_):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._sr_data = {
                'ENumer':[],
                'EDenom':[],
                'E1Body':[],
                'E2Body':[],
                }
    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        # Need to be able to dispatch here
        Etot,E1,E2,R0 = hamiltonian.local_energy(walkers,trial)
        # walker weights becomes 1 after sr
        assert xp.linalg.norm(walkers.weight)/xp.sqrt(walkers.nwalkers)<1e-6

        weight = walkers.sgn_ovlp * R0 
        self._data["ENumer"] = xp.sum(weight * Etot)
        self._data["EDenom"] = xp.sum(weight)
        self._data["E1Body"] = xp.sum(weight * E1)
        self._data["E2Body"] = xp.sum(weight * E2)
        return self.data
    def process_sr_data(self,data,dirname='.'):
        for key in self._sr_data: 
            ix = self._data_index[key]
            self._sr_data[key].append(data[ix])
        self.save_sr_data(dirname)
    def save_sr_data(self,dirname='./'): 
        if dirname is None:
            return
        f = h5py.File(f'{dirname}/energy.h5','w')
        for key,data in self._sr_data.items():
            f.create_dataset(key,data=xp.array(data))
        f.close()
    def load_sr_data(self,dirname,rank):
        if rank>0:
            return
        f = h5py.File(f'{dirname}/energy.h5','r')
        for key in self._sr_data:
            self._sr_data = list(f[key][:])
        f.close()
    def compute_estimator_sr(self,weight,rank,shift):
        if rank>0:
            return
        est_data = xp.zeros(len(self._data_index))
        for key,data in self._sr_data.items():
            ix = self._data_index[key]
            est_data[ix] = xp.dot(weight,xp.array(data[shift:]))
        return est_data
