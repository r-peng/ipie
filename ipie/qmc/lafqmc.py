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
#          Joonho Lee <linusjoonho@gmail.com>
#          Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

"""Driver to perform LAFQMC calculation"""

import abc
import json
import time
import uuid
import math
from typing import Dict, Optional, Tuple
from functools import partial
import numpy

from ipie.config import config
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.handler_custom import EstimatorHandler
from ipie.qmc.options import QMCParams
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.backend import synchronize
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.pop_controller_custom import PopController
from ipie.qmc.afqmc import AFQMCBase
from ipie.walkers.walkers_utils import (
        preprocess,
        save_walkers,
        load_walkers,
        compute_intermediates,
        update_walkers,
        orthogonalise,
)
   
class LAFQMC(AFQMCBase):
    """AFQMC driver for zero temperature open ended random walk.

    Parameters
    ----------
    system :
        System class. TODO Remove this?
    hamiltonian :
        Hamiltonian describing the system.
    trial :
        Trial wavefunction
    walkers :
        Walkers used for open ended random walk.
    propagator :
        Class describing how to propagate walkers.
    params :
        Parameters of simulation. See QMCParams for description.
    verbose : bool
        How much information to print.

    Attributes
    ----------
    _parallel_rng_seed : int
        Seed deduced from params.rng_seed which is generally different on each
            MPI process.
    """

    @staticmethod
    # TODO: wavefunction type, trial type, hamiltonian type
    def build(
        hamiltonian,
        trial_wavefunction,
        walkers,
        seed: Optional[int] = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        stabilize_freq=5,
        eq_stabilize_freq=2,
        pop_control_method="stochastic_reconfiguration",
        pop_control_freq=5,
        eq_pop_control_freq=2,
        eq_num_steps_per_block=None,
        num_eq_blocks: int = 0,
        ene_bound_const: float = 2.0,
        fb_bound: float = 1.0,
        correlated_samp: bool = False,
        reference_run: bool = False,
        walkermap_filepath: Optional[str] = None,
        verbose=True,
        mpi_handler=None,
    ) -> "AFQMC":
        """Factory method to build AFQMC driver from hamiltonian and trial wavefunction.

        Parameters
        ----------
        num_elec: tuple(int, int)
            Number of alpha and beta electrons.
        hamiltonian :
            Hamiltonian describing the system.
        trial_wavefunction:
            Trial wavefunction
        num_walkers : int
            Number of walkers per MPI process used in the simulation. The TOTAL
                number of walkers is num_walkers * number of processes.
        num_steps_per_block : int
            Number of Monte Carlo steps before estimators are evaluatied.
                Default 25.
        num_blocks : int
            Number of blocks to perform. Total number of steps = num_blocks *
                num_steps_per_block.
        timestep : float
            Imaginary timestep. Default 0.005.
        stabilize_freq : int
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 5.
        eq_stabilize_freq : int
            Frequency at which to perform QR factorization of walkers during equilibration (in units
                of steps.) Default 2.
        pop_control_freq : int
            Frequency at which to perform population control (in units of
                steps.) Default 5.
        eq_pop_control_freq : int
            Frequency at which to perform population control during equilibration (in units of
                steps.) Default 2.
        eq_timestep : float
            Imaginary timestep to use during equilibration. Default None (use same as timestep).
        eq_num_steps_per_block : int
            Number of Monte Carlo steps before estimators are evaluatied during equilibration. Default None (use same as num_steps_per_block).
        num_eq_blocks : int
            Number of blocks to perform during equilibration. Total number of steps = num_eq_blocks * eq_num_steps_per_block. Default 50.
        ene_bound_const : float
            Constant to determine local energy bound.
        fb_bound : float
            Constant to determine force bias bound.
        correlated_samp : bool
            Whether to use correlated sampling for population control. Default False.
        reference_run : bool
            Whether this is a reference run (i.e. generating the reference population control decisions for the sample runs in correlated sampling). Default False.
        walkermap_filepath : str
            Filepath to write walkermap (the population control decisions) to. If None, do not write walkermap. Default None, only write walkermap if this is a reference run.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        if mpi_handler is None:
            mpi_handler = MPIHandler()
            comm = mpi_handler.comm
        else:
            comm = mpi_handler.comm
        num_walkers = walkers.nwalkers
        params = QMCParams(
            num_walkers=num_walkers,
            total_num_walkers=num_walkers * comm.size,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            timestep=None,
            num_stblz=stabilize_freq,
            pop_control_method=pop_control_method,
            num_eq_stblz=eq_stabilize_freq,
            pop_control_freq=pop_control_freq,
            eq_pop_control_freq=eq_pop_control_freq,
            rng_seed=seed,
            eq_timestep=None,
            eq_num_steps_per_block=eq_num_steps_per_block,
            num_eq_blocks=num_eq_blocks,
            fb_bound=fb_bound,
            ene_bound_const=ene_bound_const,
            correlated_samp=correlated_samp,
            reference_run=reference_run,
            walkermap_filepath=walkermap_filepath,
        )
        return LAFQMC(
            None,
            hamiltonian,
            trial_wavefunction,
            walkers,
            None,
            mpi_handler,
            params,
            None,
            verbose=(verbose and comm.rank == 0),
        )

    def copy_to_gpu(self):
        comm = self.mpi_handler.comm
        if config.get_option("use_gpu"):
            ngpus = xp.cuda.runtime.getDeviceCount()
            _ = xp.cuda.runtime.getDeviceProperties(0)
            # xp.cuda.runtime.setDevice(self.shared_comm.rank % 4)
            xp.cuda.runtime.setDevice(comm.rank % ngpus)
            if comm.rank == 0:
                if ngpus > comm.size:
                    print(
                        f"# There are unused GPUs ({comm.size} MPI tasks but {ngpus} GPUs). "
                        " Check if this is really what you wanted."
                    )
            self.trial.cast_to_cupy(self.verbose and comm.rank == 0)
            self.walkers.cast_to_cupy(self.verbose and comm.rank == 0)

    def setup_estimators(
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None, start_step=0, load_dirname=None,
    ):
        self.accumulators = WalkerAccumulator(
            ["Weight", "WeightFactor", "HybridEnergy"], self.params.num_steps_per_block
        )
        comm = self.mpi_handler.comm
        self.estimators = EstimatorHandler(
            self.mpi_handler.comm,
            None,
            self.hamiltonian,
            self.trial,
            walker_state=self.accumulators,
            verbose=(comm.rank == 0 and self.verbose),
            filename=filename,
        )
        if additional_estimators is not None:
            for k, v in additional_estimators.items():
                self.estimators[k] = v
        ## TODO: Move this to estimator and log uuid etc in serialization
        #json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
        #json_string = to_json(self)
        #self.estimators.json_string = json_string

        self.estimators.initialize(comm)
        # Calculate estimates for initial distribution of walkers.
        self.accumulators.update(self.walkers)
        if start_step==0:
            self.estimators.compute_estimators(self.system, self.hamiltonian, self.trial, self.walkers)
            self.estimators.print_block(comm, 0, self.accumulators)
        else:
            self.estimators.load(load_dirname,comm.rank)
            self.estimate_energy(comm,start_step)
        self.accumulators.zero()

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose=True,
        discard_weights_aftereq=False,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
        constraint_path=True,
        minibatch_size=1,
        lowdin=False,
        eps_sq=None,
        max_nprod=20,
        max_nsum=500,
        dirname='.',
        start_step=0,
        load_dirname=None,
    ):
        """Perform AFQMC simulation on state object using open-ended random walk.

        Parameters
        ----------
        walkers : :class:`ipie.walker.Walkers` object
            Initial wavefunction / distribution of walkers. Default None.
        estimator_filename : str
            File to write estimates to.
        additional_estimators : dict
            Dictionary of additional estimators to evaluate.
        """
        comm = self.mpi_handler.comm
        # parsing propagation parameters.
        num_eqlb_steps = self.params.num_eq_blocks * self.params.eq_num_steps_per_block
        total_steps = self.params.num_steps_per_block * self.params.num_blocks + num_eqlb_steps
        self.eps_sq = eps_sq
        self.max_nprod = max_nprod
        self.max_nsum = max_nsum
        if comm.rank==0:
            print('num_eqlb_steps=',num_eqlb_steps)
            print('num_eq_stblz=',self.params.num_eq_stblz)
            print('num_stblz=',self.params.num_stblz)
            print('eq_pop_control_freq=',self.params.eq_pop_control_freq)
            print('pop_control_freq=',self.params.pop_control_freq)

        self.setup_timers()
        tzero_setup = time.time()
        if walkers is not None:
            self.walkers = walkers
        if start_step>0:
            load_walkers(self.walkers,comm,load_dirname)
        self.setup_timers()
        eshift = 0.0
        self.walkers.orthogonalise()

        self.pcontrol_eq = PopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            pop_control_method=self.params.pop_control_method,
            verbose=self.verbose,
        )

        self.pcontrol = PopController(
            self.params.num_walkers,
            self.params.num_steps_per_block,
            self.mpi_handler,
            pop_control_method=self.params.pop_control_method,
            verbose=self.verbose,
            correlated_samp=self.params.correlated_samp,
            reference_run=self.params.reference_run,
            walkermap_filepath=self.params.walkermap_filepath,
        )

        self.get_env_info()
        # self.distribute_hamiltonian()
        self.copy_to_gpu()

        start = time.time()
        iprint = 1 if comm.rank==0 else 0
        preprocess(self.walkers,self.trial,self.hamiltonian,iprint=iprint)
        if comm.rank==0:
            print('preprocess time=',time.time()-start)

        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators,start_step=start_step,load_dirname=load_dirname)

        synchronize()
        self.tsetup += time.time() - tzero_setup

        for step in range(1+start_step, total_steps + 1+start_step):
            synchronize()
            start_step = time.time()
            if step <= num_eqlb_steps:
                if step % self.params.num_eq_stblz == 0:
                    start = time.time()
                    #self.walkers.orthogonalise()
                    orthogonalise(self.walkers,self.trial)
                    synchronize()
                    self.tortho += time.time() - start
            else:
                if step % self.params.num_stblz == 0:
                    start = time.time()
                    #self.walkers.orthogonalise()
                    orthogonalise(self.walkers,self.trial)
                    synchronize()
                    self.tortho += time.time() - start

            start = time.time()
            self.propagate_walkers(lowdin=lowdin,constraint_path=constraint_path,minibatch_size=minibatch_size)
            self.tprop_update += time.time() - start 

            #start_clip = time.time()
            #if step > 1 and step <= num_eqlb_steps:
            #    wbound = self.pcontrol_eq.total_weight * 0.10
            #    xp.nan_to_num(self.walkers.weight, copy=False)
            #    xp.clip(
            #        self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
            #    )  # in-place clipping
            #elif step > num_eqlb_steps and step > 1:
            #    wbound = self.pcontrol.total_weight * 0.10
            #    xp.nan_to_num(self.walkers.weight, copy=False)
            #    xp.clip(
            #        self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
            #    )  # in-place clipping

            #synchronize()
            #self.tprop_clip += time.time() - start_clip

            start_barrier = time.time()
            if step % self.params.pop_control_freq == 0:
                comm.Barrier()
            self.tprop_barrier += time.time() - start_barrier

            self.tprop += time.time() - start
            if step <= num_eqlb_steps:
                if step % self.params.eq_pop_control_freq == 0:
                    start = time.time()
                    self.pop_ctr(comm,self.pcontrol_eq)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol_eq.timer.send_time
                    self.tpopc_recv = self.pcontrol_eq.timer.recv_time
                    self.tpopc_comm = self.pcontrol_eq.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol_eq.timer.non_communication_time
            else:
                if step % self.params.pop_control_freq == 0:
                    start = time.time()
                    self.pop_ctr(comm,self.pcontrol)
                    synchronize()
                    orthogonalise(self.walkers,self.trial)
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol.timer.send_time
                    self.tpopc_recv = self.pcontrol.timer.recv_time
                    self.tpopc_comm = self.pcontrol.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

            # accumulate weight, hybrid energy etc. across block
            start = time.time()
            self.accumulators.update(self.walkers)
            orthogonalise(self.walkers,self.trial)
            synchronize()
            self.testim += time.time() - start  # we dump this time into estimator

            # calculate estimators
            start = time.time()
            if step > num_eqlb_steps:
                if step % self.params.num_steps_per_block == 0:
                    block = (step - num_eqlb_steps) // self.params.num_steps_per_block
                    self.estimate_energy(comm,block)
                    self.accumulators.zero()
                    self.save(comm,dirname)
                    self.print_stats(comm,self.pcontrol)
            else:
                if step % self.params.eq_num_steps_per_block == 0:
                    block = step // self.params.eq_num_steps_per_block
                    self.estimate_energy(comm,block)
                    self.accumulators.zero()
                    self.save(comm,dirname)
                    self.print_stats(comm,self.pcontrol_eq)
            synchronize()
            self.testim += time.time() - start

            ## restart write features disabled
            #if self.walkers.write_restart:
            #    if self.walkers.write_freq is not None:
            #        if step % self.walkers.write_freq == 0:
            #            self.walkers.write_walkers_batch(comm)
            #    else:
            #        assert self.walkers.write_time is not None
            #        if step == self.walkers.write_time:
            #            self.walkers.write_walkers_batch(comm)

            #if step < num_eqlb_steps:
            #    eshift = self.accumulators.eshift
            #else:
            #    eshift += self.accumulators.eshift - eshift
            #synchronize()
            #self.tstep += time.time() - start_step

    def propagate_walkers_simple(self,lowdin=True):
        # sample rotations 
        kixs = xp.random.choice(self.hamiltonian.nterms,size=self.walkers.nwalkers,replace=True,p=self.hamiltonian.prob)
        rotations = self.hamiltonian.parse_sampled_rotations(to_host(kixs))

        # update walkers 
        b = self.hamiltonian.a_over_q[kixs].copy()
        try:
            compute_intermediates(self.walkers,rotations,lowdin=lowdin)
        except ValueError:
            orthogonalise(self.walkers,self.trial)
            compute_intermediates(self.walkers,rotations,lowdin=lowdin)
        b = update_walkers(self.walkers,rotations,b=b,lowdin=lowdin)
        return kixs,b
    
    def propagate_walkers_minibatch(self,K,lowdin=True):
        nw = self.walkers.nwalkers
        # compute probability 
        minibatch_kixs = xp.random.choice(self.hamiltonian.nterms,size=(K,nw),replace=True,p=self.hamiltonian.prob)
        minibatch_kixs_host = to_host(minibatch_kixs)
    
        b = xp.ones((K,nw))
        rotations = [None] * K
        for i in range(K):
            kixs = minibatch_kixs[i]
            b[i] = self.hamiltonian.a_over_q[kixs].copy()
            kixs = minibatch_kixs_host[i]
            rotations[i] = self.hamiltonian.parse_sampled_rotations(kixs)
    
        try:
            for i in range(K):
                compute_intermediates(self.walkers,rotations[i],lowdin=lowdin)
        except ValueError:
            orthogonalise(self.walkers,self.trial)
            for i in range(K):
                compute_intermediates(self.walkers,rotations[i],lowdin=lowdin)

        for i in range(K):
            b[i] = update_walkers(self.walkers,rotations[i],walkers_update=False,b=b[i])
        
        # sample rotations 
        p = xp.fabs(b)
        B = p.sum(axis=0)
        p /= B[None,:] 
        kixs = [None] * nw
        for i in range(nw):
            k = xp.random.choice(K,size=1,p=p[:,i])[0]
            B[i] *= xp.sign(b[k,i])
    
            kixs[i] = minibatch_kixs[k,i]
        kixs = xp.asarray(kixs)
        rotations = self.hamiltonian.parse_sampled_rotations(to_host(kixs))

        # update walkers
        B /= K
        compute_intermediates(self.walkers,rotations,lowdin=lowdin)
        update_walkers(self.walkers,rotations,lowdin=lowdin)
        return kixs,B

    def propagate_walkers(self, lowdin=True, constraint_path=True, minibatch_size=1):
        if minibatch_size==1:
            kixs,b = self.propagate_walkers_simple(lowdin=lowdin)
        elif minibatch_size<self.hamiltonian.nterms:
            kixs,b = self.propagate_walkers_minibatch(minibatch_size,lowdin=lowdin)
        else:
            raise NotImplementedError
        synchronize()
    
        # 4.update weight
        bminus = xp.nonzero(b<0.)[0] 
        nminus = bminus.size
        if nminus>0: 
            kixs = kixs[bminus]
            kixs = to_host(kixs)
            for kix in kixs:
                term = self.hamiltonian.terms[kix]
                print(f'term.chol_idx={term.chol_idx},d={term.d}')

        if constraint_path:
            xp.clip(b, a_min=0.0, a_max=None, out=b)  
        self.walkers.weight *= b 
        #print(f'term_label={kixs[0]},weight multiplier={b[0]},accumulated weight={self.walkers.weight[0]},ovlp={1./self.walkers.Sa[0,0,0]}')
        synchronize()

    def pop_ctr(self,comm,pcontrol,pre_estimate=True):
        if self.params.pop_control_method=='stochastic_reconfiguration':
            if pre_estimate: 
                self.estimators.compute_estimators(self.system, self.hamiltonian, self.trial, self.walkers)
        log_average_weight = pcontrol.pop_control(self.walkers, comm)
        if self.params.pop_control_method!='stochastic_reconfiguration':
            return
        if not pre_estimate:
            self.estimators.compute_estimators(self.system, self.hamiltonian, self.trial, self.walkers)
        self.estimators.post_sr(comm,self.accumulators,log_average_weight)

    def estimate_energy(self,comm,block):
        if self.params.pop_control_method=='stochastic_reconfiguration':
            self.estimators.print_block_sr(comm,block,self.accumulators,self.max_nprod,self.max_nsum)
        else:
            self.estimators.compute_estimators(self.system, self.hamiltonian, self.trial, self.walkers)
            self.estimators.print_block(comm, block, self.accumulators)

    def save(self,comm,dirname='.'):
        if dirname is None:
            return
        save_walkers(self.walkers,comm,dirname)
        if self.params.pop_control_method!='stochastic_reconfiguration':
            return
        if comm.rank>0:
            return
        self.estimators.save(dirname)

    def print_stats(self,comm,pcontrol):
        if comm.rank>0:
            return
        ttot = self.tortho + self.tprop_update + self.tprop_barrier + self.tpopc + self.testim
        print(f'total={ttot}, orth={self.tortho}, prop={self.tprop_update}, barrier={self.tprop_barrier}, pop ctr={self.tpopc}, estimate={self.testim}')
        if self.params.pop_control_method!='stochastic_reconfiguration':
            return
        Neff = pcontrol.Neff
        Neff = max(Neff),min(Neff)
        pcontrol.Neff = []
        Ndistinct = pcontrol.Ndistinct
        Ndistinct = max(Ndistinct),min(Ndistinct)
        pcontrol.Ndistinct = []
        w = self.estimators.log_average_weights
        w = max(w),min(w)
        w = float(numpy.exp(w[0])),float(numpy.exp(w[1]))
        print(f'wmean={w},Neff={Neff},Ndistinct={Ndistinct}')
