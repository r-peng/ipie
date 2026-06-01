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


from ipie.config import config
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.handler_custom import EstimatorHandler
from ipie.hamiltonians.sor_base import save_walkers
from ipie.qmc.options import QMCParams
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.backend import synchronize
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import WalkerAccumulator
from ipie.walkers.pop_controller_custom import PopController
from ipie.qmc.afqmc import AFQMCBase

#def update_weight(walkers,b,constraint_path=True,thresh=1e-15):
#    minus_idx = xp.nonzero(b<-thresh)
#    minus_val = -1
#    b_abs = xp.fabs(b)
#    zero_idx = xp.nonzero(b_abs<thresh)
#    if constraint_path:
#        b[minus_idx] = thresh
#        minus_val = 0
#    walkers.weight += xp.log(b_abs)
#    # use .sgn_ovlp to store signs for now
#    # doesn't seemed to be used for anything else
#    walkers.sgn_ovlp[minus_idx] *= minus_val 
#    walkers.sgn_ovlp[zero_idx] = 0

def propagate_walkers(walkers, hamiltonian, trial, constraint_path=True):
    # 1.compute gf
    gf = hamiltonian.bare_gf

    # 2.sample from gf
    sign = xp.sign(gf)
    p = xp.fabs(gf)
    p /= p.sum()
    kixs = xp.random.choice(gf.size,size=walkers.nwalkers,replace=True,p=p)
    sign = sign[kixs] 

    # 3.update walker
    #start_time = time.time()
    ovlp_old = walkers.ovlp
    #keys = [hamiltonian.keys[int(kix)] for kix in to_host(kixs)]
    hamiltonian.update_walkers(to_host(kixs),walkers)
    hamiltonian.compute_guiding_fxn(walkers,trial)
    synchronize()
    #self.timer.tgemm += time.time() - start_time

    # 4.update weight
    #start_time = time.time()
    b = sign*walkers.ovlp/ovlp_old
    if constraint_path:
        xp.clip(b, a_min=0.0, a_max=None, out=b)  # in-place clipping (cosine projection)
    walkers.weight *= b 
    synchronize()
    #self.timer.tupdate += time.time() - start_time

    
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
        num_elec: Tuple[int, int],
        hamiltonian,
        trial_wavefunction,
        walkers=None,
        num_walkers: int = 100,
        seed: Optional[int] = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq=5,
        eq_stabilize_freq=2,
        pop_control_method="stochastic_reconfiguration",
        pop_control_freq=5,
        eq_pop_control_freq=2,
        eq_timestep=None,
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
        params = QMCParams(
            num_walkers=num_walkers,
            total_num_walkers=num_walkers * comm.size,
            num_blocks=num_blocks,
            num_steps_per_block=num_steps_per_block,
            timestep=timestep,
            num_stblz=stabilize_freq,
            pop_control_method=pop_control_method,
            num_eq_stblz=eq_stabilize_freq,
            pop_control_freq=pop_control_freq,
            eq_pop_control_freq=eq_pop_control_freq,
            rng_seed=seed,
            eq_timestep=eq_timestep,
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

    @staticmethod
    # TODO: wavefunction type, trial type, hamiltonian type
    def build_from_hdf5(
        num_elec: Tuple[int, int],
        ham_file,
        wfn_file,
        num_walkers: int = 100,
        seed: Optional[int] = None,
        num_steps_per_block: int = 25,
        num_blocks: int = 100,
        timestep: float = 0.005,
        stabilize_freq=5,
        pop_control_freq=5,
        num_dets_chunk=1,
        num_dets_for_trial_props=100,
        pack_cholesky=True,
        verbose=True,
    ) -> "AFQMC":
        """Factory method to build AFQMC driver from hamiltonian and trial wavefunction.

        Parameters
        ----------
        num_elec: tuple(int, int)
            Number of alpha and beta electrons.
        ham_file : str
            Path to Hamiltonian describing the system.
        wfn_file : str
            Path to Trial wavefunction
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
        stabilize_freq : float
            Frequency at which to perform QR factorization of walkers (in units
                of steps.) Default 25.
        pop_control_freq : int
            Frequency at which to perform population control (in units of
                steps.) Default 25.
        num_det_chunks : int
            Size of chunks of determinants to process during batching. Default=1 (no batching).
        num_dets_for_trial_props: int
            Number of determinants to use to evaluate trial wavefunction properties.
        pack_cholesky : bool
            Use symmetry to reduce memory consumption of integrals. Default True.
        verbose : bool
            Log verbosity. Default True i.e. print information to stdout.
        """
        mpi_handler = MPIHandler()
        _verbose = verbose and mpi_handler.comm.rank == 0
        ham = get_hamiltonian(
            ham_file, mpi_handler.scomm, verbose=_verbose, pack_chol=pack_cholesky
        )
        trial = get_trial_wavefunction(
            num_elec,
            ham.nbasis,
            wfn_file,
            ndet_chunks=num_dets_chunk,
            ndets_props=num_dets_for_trial_props,
            verbose=_verbose,
        )
        trial.half_rotate(ham, mpi_handler.scomm)
        return LAFQMC.build(
            trial.nelec,
            ham,
            trial,
            num_walkers=num_walkers,
            seed=seed,
            num_steps_per_block=num_steps_per_block,
            num_blocks=num_blocks,
            timestep=timestep,
            stabilize_freq=stabilize_freq,
            pop_control_freq=pop_control_freq,
            verbose=verbose,
            mpi_handler=mpi_handler,
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
        self, filename, additional_estimators: Optional[Dict[str, EstimatorBase]] = None
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
        self.estimators.compute_estimators(self.system, self.hamiltonian, self.trial, self.walkers)
        self.accumulators.update(self.walkers)
        self.estimators.print_block(comm, 0, self.accumulators)
        self.accumulators.zero()

    def post_pop_ctr(self,comm,average_weight):
        if self.params.pop_control_method!='stochastic_reconfiguration':
            return
        self.estimators.compute_estimators(
            self.system, self.hamiltonian, self.trial, self.walkers
        )
        self.estimators.post_sr(comm,self.accumulators,average_weight)

    def estimate_energy(self,comm,block,max_nprod,max_nsum):
        if self.params.pop_control_method=='stochastic_reconfiguration':
            self.estimators.print_block_sr(comm,block,self.accumulators,max_nprod,max_nsum)
        else:
            self.estimators.compute_estimators(
                self.system, self.hamiltonian, self.trial, self.walkers
            )
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

    def run(
        self,
        walkers=None,
        estimator_filename=None,
        verbose=True,
        discard_weights_aftereq=False,
        additional_estimators: Optional[Dict[str, EstimatorBase]] = None,
        constraint_path=True,
        max_nprod=20,
        max_nsum=500,
        dirname='.',
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
        self.setup_timers()
        tzero_setup = time.time()
        if walkers is not None:
            self.walkers = walkers
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
        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators)

        # TODO: This magic value of 2 is pretty much never controlled on input.
        # Moreover I'm not convinced having a two stage shift update actually
        # matters at all.
        # num_eqlb_steps = 2.0 / self.params.timestep
        num_eqlb_steps = self.params.num_eq_blocks * self.params.eq_num_steps_per_block

        total_steps = self.params.num_steps_per_block * self.params.num_blocks + num_eqlb_steps

        synchronize()
        comm = self.mpi_handler.comm
        self.tsetup += time.time() - tzero_setup

        t0 = time.time()
        for step in range(1, total_steps + 1):
            synchronize()
            start_step = time.time()
            if step <= num_eqlb_steps:
                if step % self.params.num_eq_stblz == 0:
                    start = time.time()
                    self.walkers.orthogonalise()
                    synchronize()
                    self.tortho += time.time() - start
            else:
                if step % self.params.num_stblz == 0:
                    start = time.time()
                    self.walkers.orthogonalise()
                    synchronize()
                    self.tortho += time.time() - start
            start = time.time()
            if step <= num_eqlb_steps:
                propagate_walkers(
                    self.walkers, self.hamiltonian, self.trial, constraint_path=constraint_path 
                )
                #self.tprop_fbias = self.eq_propagator.timer.tfbias
                #self.tprop_ovlp = self.eq_propagator.timer.tovlp
                #self.tprop_update = self.eq_propagator.timer.tupdate
                #self.tprop_gf = self.eq_propagator.timer.tgf
                #self.tprop_vhs = self.eq_propagator.timer.tvhs
                #self.tprop_gemm = self.eq_propagator.timer.tgemm
            else:
                if discard_weights_aftereq:
                    if step == num_eqlb_steps + 1:
                        self.walkers.weight.fill(1.0)
                propagate_walkers(
                    self.walkers, self.hamiltonian, self.trial, constraint_path=constraint_path 
                )
                #self.tprop_fbias = self.propagator.timer.tfbias
                #self.tprop_ovlp = self.propagator.timer.tovlp
                #self.tprop_update = self.propagator.timer.tupdate
                #self.tprop_gf = self.propagator.timer.tgf
                #self.tprop_vhs = self.propagator.timer.tvhs
                #self.tprop_gemm = self.propagator.timer.tgemm

            start_clip = time.time()
            if step > 1 and step <= num_eqlb_steps:
                wbound = self.pcontrol_eq.total_weight * 0.10
                xp.nan_to_num(self.walkers.weight, copy=False)
                xp.clip(
                    self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
                )  # in-place clipping
            elif step > num_eqlb_steps and step > 1:
                wbound = self.pcontrol.total_weight * 0.10
                xp.nan_to_num(self.walkers.weight, copy=False)
                xp.clip(
                    self.walkers.weight, a_min=-wbound, a_max=wbound, out=self.walkers.weight
                )  # in-place clipping

            synchronize()
            self.tprop_clip += time.time() - start_clip

            start_barrier = time.time()
            if step % self.params.pop_control_freq == 0:
                comm.Barrier()
            self.tprop_barrier += time.time() - start_barrier

            self.tprop += time.time() - start
            if step <= num_eqlb_steps:
                if step % self.params.eq_pop_control_freq == 0:
                    start = time.time()
                    average_weight = self.pcontrol_eq.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol_eq.timer.send_time
                    self.tpopc_recv = self.pcontrol_eq.timer.recv_time
                    self.tpopc_comm = self.pcontrol_eq.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol_eq.timer.non_communication_time
            else:
                if step % self.params.pop_control_freq == 0:
                    start = time.time()
                    average_weight = self.pcontrol.pop_control(self.walkers, comm)
                    synchronize()
                    self.tpopc += time.time() - start
                    self.tpopc_send = self.pcontrol.timer.send_time
                    self.tpopc_recv = self.pcontrol.timer.recv_time
                    self.tpopc_comm = self.pcontrol.timer.communication_time
                    self.tpopc_non_comm = self.pcontrol.timer.non_communication_time

            # accumulate weight, hybrid energy etc. across block
            start = time.time()
            self.accumulators.update(self.walkers)
            synchronize()
            self.testim += time.time() - start  # we dump this time into estimator

            # post poppulation control accumulation
            if step <= num_eqlb_steps:
                if step % self.params.eq_pop_control_freq == 0:
                    self.post_pop_ctr(comm,average_weight)
            else:
                if step % self.params.pop_control_freq == 0:
                    self.post_pop_ctr(comm,average_weight)

            # calculate estimators
            start = time.time()
            if step > num_eqlb_steps:
                if step % self.params.num_steps_per_block == 0:
                    block = (step - num_eqlb_steps) // self.params.num_steps_per_block
                    self.estimate_energy(comm,block,max_nprod,max_nsum)
                    self.accumulators.zero()
                    self.save(comm,dirname)
                    if comm.rank==0:
                        print('time=',time.time()-t0)
            else:
                if step % self.params.eq_num_steps_per_block == 0:
                    block = step // self.params.eq_num_steps_per_block
                    self.estimate_energy(comm,block,max_nprod,max_nsum)
                    self.accumulators.zero()
                    self.save(comm,dirname)
                    if comm.rank==0:
                        print('time=',time.time()-t0)
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
