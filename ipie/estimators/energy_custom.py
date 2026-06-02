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

import h5py

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
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
        Etot,E1,E2 = hamiltonian.local_energy(walkers)
        if walkers.R is None:
            weight = walkers.weight
        else:
            weight = walkers.weight * walkers.R 
            #print('R mean=',to_host(xp.mean(R)))

        self._data["ENumer"] = xp.sum(weight * Etot)
        self._data["EDenom"] = xp.sum(weight)
        self._data["E1Body"] = xp.sum(weight * E1)
        self._data["E2Body"] = xp.sum(weight * E2)
        return self.data
    def process_sr_data(self,data,dirname='.'):
        for key in self._sr_data: 
            ix = self._data_index[key]
            self._sr_data[key].append(data[ix])
    def save_sr_data(self,dirname='.'): 
        if dirname is None:
            return
        with h5py.File(f'{dirname}/energy.h5','w') as f:
            for key,data in self._sr_data.items():
                f.create_dataset(key,data=to_host(xp.asarray(data)))
    def load_sr_data(self,dirname,rank):
        if rank>0:
            return
        with h5py.File(f'{dirname}/energy.h5','r') as f:
            for key in self._sr_data:
                self._sr_data[key] = list(f[key][:])
    def compute_estimator_sr(self,weight,rank,shift,ntot=None):
        if rank>0:
            return
        est_data = xp.zeros(len(self._data_index))
        for key,data in self._sr_data.items():
            ix = self._data_index[key]
            dat = data[shift:] if ntot is None else data[shift:ntot]
            est_data[ix] = xp.dot(weight,xp.asarray(dat))
        return est_data
