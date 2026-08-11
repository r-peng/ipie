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

"""Driver to perform LAFQMC calculation with Poisson count propagation."""

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
#from ipie.walkers.pop_controller_custom import PopController
from ipie.walkers.pop_controller import PopController
from ipie.qmc.afqmc import AFQMCBase
   
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
        pop_control_method="pair_branch",
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
            #self.estimators.load(load_dirname,comm.rank)
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
        importance_sample=False,
        prop_interval=5,
        poisson_rate=1.0,
        poisson_energy_shift=0.0,
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
        poisson_rate : float
            Poisson process rate r. Bare sampling uses r = 1 by default.
        poisson_energy_shift : float
            Trial energy shift E_T used in the final Poisson weight factor.
        """
        comm = self.mpi_handler.comm
        if importance_sample:
            raise NotImplementedError("Poisson LAFQMC is only implemented for bare sampling.")
        # parsing propagation parameters.
        num_eqlb_steps = self.params.num_eq_blocks * self.params.eq_num_steps_per_block
        num_prod_steps = self.params.num_steps_per_block * self.params.num_blocks
        total_steps = num_prod_steps + num_eqlb_steps
        if self.params.pop_control_freq <= 0:
            raise ValueError("pop_control_freq must be positive.")
        if self.params.eq_pop_control_freq <= 0:
            raise ValueError("eq_pop_control_freq must be positive.")
        poisson_rate = float(poisson_rate)
        poisson_energy_shift = float(poisson_energy_shift)
        if poisson_rate <= 0.0:
            raise ValueError("poisson_rate must be positive.")
        self.eps_sq = eps_sq
        self.max_nprod = max_nprod
        self.max_nsum = max_nsum
        if comm.rank==0:
            print('num_eqlb_steps=',num_eqlb_steps)
            print('num_eq_stblz=',self.params.num_eq_stblz)
            print('num_stblz=',self.params.num_stblz)
            print('eq_pop_control_freq=',self.params.eq_pop_control_freq)
            print('pop_control_freq=',self.params.pop_control_freq)
            print('poisson_rate=',poisson_rate)
            print('poisson_energy_shift=',poisson_energy_shift)

        self.setup_timers()
        tzero_setup = time.time()
        if walkers is not None:
            self.walkers = walkers
        if start_step>0:
            self.walkers.load(comm,load_dirname)
        self.setup_timers()
        eshift = 0.0
        self.walkers.reortho(None)

        self.pcontrol_eq = PopController(
            self.params.num_walkers,
            self.params.eq_num_steps_per_block,
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
        self.trial.build(self.hamiltonian,conjugate=(not importance_sample))
        self.walkers.build(self.hamiltonian,self.trial,importance=importance_sample)
        if comm.rank==0:
            print('preprocess time=',time.time()-start)

        self.setup_estimators(estimator_filename, additional_estimators=additional_estimators,start_step=start_step,load_dirname=load_dirname)

        synchronize()
        self.tsetup += time.time() - tzero_setup

        for step in range(1, total_steps + 1):
            synchronize()
            #start_step = time.time()
            if step <= num_eqlb_steps:
                if step % self.params.num_eq_stblz == 0:
                    start = time.time()
                    self.walkers.reortho(self.trial)
                    synchronize()
                    self.tortho += time.time() - start
            else:
                if step % self.params.num_stblz == 0:
                    start = time.time()
                    self.walkers.reortho(self.trial)
                    synchronize()
                    self.tortho += time.time() - start

            start = time.time()
            self.propagate_walkers(
                prop_interval,
                constraint_path=constraint_path,
                poisson_rate=poisson_rate,
                poisson_energy_shift=poisson_energy_shift,
            )
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

            # calculate estimators
            start = time.time()
            if step > num_eqlb_steps:
                if step % self.params.num_steps_per_block == 0:
                    block = (step - num_eqlb_steps) // self.params.num_steps_per_block
                    #print(start_step)
                    self.estimate_energy(comm,block+start_step)
                    self.accumulators.zero()
                    self.save(comm,dirname)
                    self.print_stats(comm,self.pcontrol)
            else:
                if step % self.params.eq_num_steps_per_block == 0:
                    block = step // self.params.eq_num_steps_per_block
                    self.estimate_energy(comm,block+start_step)
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

    def poisson_tau_from_pop_control_freq(self, pop_control_freq):
        lambda_tot = float(to_host(self.hamiltonian.Lambda[-1]))
        if lambda_tot <= 0.0:
            raise ValueError("hamiltonian.Lambda[-1] must be positive.")
        return float(pop_control_freq) / lambda_tot

    def parse_walker_samples(self, ixs, walkers=None):
        if walkers is None:
            self.hamiltonian.parse_samples(to_host(ixs))
            return

        ixs = numpy.atleast_1d(to_host(ixs))
        walkers = numpy.atleast_1d(to_host(walkers))
        if ixs.size != walkers.size:
            raise ValueError("Number of samples must match number of walker indices.")

        w_dict = dict()
        i_dict = dict()
        for walker, ix in zip(walkers, ixs):
            key, i = self.hamiltonian.ix2key[int(ix)]
            if key not in w_dict:
                w_dict[key] = []
            w_dict[key].append(int(walker))
            if key not in i_dict:
                i_dict[key] = []
            i_dict[key].append(i)

        self.hamiltonian.samples = dict()
        for key in w_dict:
            w = xp.asarray(w_dict[key], dtype=xp.int64)
            i = xp.asarray(i_dict[key], dtype=xp.int64)
            self.hamiltonian.samples[key] = {'w':w,'i':i}

    def propagate_walkers(
        self,
        prop_interval,
        constraint_path=True,
        poisson_rate=1.0,
        poisson_energy_shift=0.0,
    ):
        tau = self.poisson_tau_from_pop_control_freq(prop_interval)
        b,ixs = self.propagate_walkers_bare_poisson(
            tau,
            rate=poisson_rate,
            energy_shift=poisson_energy_shift,
        )
        synchronize()

        #bminus = xp.nonzero(b<0.)[0] 
        #nminus = bminus.size
        #if nminus>0: 
        #    print('nminus=',nminus)
        #    ixs = to_host(ixs)
        #    ixs = [ixs[i] for i in bminus]
        #    for ix in ixs:
        #        chol_ix,spin,p,d = self.hamiltonian.get_term(ix)
        #        print(f'chol_ix={chol_ix},spin={spin},p={p},d={d}')

        if constraint_path:
            xp.clip(b, a_min=0.0, a_max=None, out=b)  
        self.walkers.weight *= b 
        #print(f'term_label={kixs[0]},weight multiplier={b[0]},accumulated weight={self.walkers.weight[0]},ovlp={1./self.walkers.Sa[0,0,0]}')
        synchronize()

    def propagate_walkers_bare(self):
        nterms = self.hamiltonian.nterms
        nw = self.walkers.nwalkers

        p = xp.fabs(self.hamiltonian.a)
        p /= p.sum()
        b = self.hamiltonian.a / p 
        ixs = xp.random.choice(nterms,size=nw,p=p,replace=True)
        b = b[ixs]

        self.hamiltonian.parse_samples(to_host(ixs))
        b = self.walkers.update_walkers(self.hamiltonian,self.trial,b=b)
        return b,ixs

    def propagate_walkers_bare_poisson(self, tau, rate=1.0, energy_shift=0.0):
        nterms = self.hamiltonian.nterms
        nw = self.walkers.nwalkers
        tau = float(tau)
        rate = float(rate)
        energy_shift = float(energy_shift)
        if tau < 0.0:
            raise ValueError("tau must be non-negative.")
        if rate <= 0.0:
            raise ValueError("rate must be positive.")

        p = self.hamiltonian.a.copy()
        if bool(to_host(xp.any(p < 0.0))):
            raise ValueError("Bare Poisson sampling requires non-negative hamiltonian.a.")
        p /= p.sum()
        event_weight = self.hamiltonian.a / (rate * p)

        lambda_tot = float(to_host(self.hamiltonian.Lambda[-1]))
        mean_count = rate * lambda_tot * tau
        counts = xp.random.poisson(mean_count, size=nw)
        max_count = int(to_host(counts).max()) if nw > 0 else 0

        b = xp.ones(nw, dtype=self.hamiltonian.a.dtype)
        for event in range(max_count):
            active = xp.nonzero(counts > event)[0]
            if active.size == 0:
                continue
            ixs = xp.random.choice(nterms, size=active.size, p=p, replace=True)
            b[active] = b[active] * event_weight[ixs]
            self.hamiltonian.parse_samples(to_host(ixs), to_host(active))
            b = self.walkers.update_walkers(self.hamiltonian,self.trial,b=b)

        end_factor = math.exp(tau * ((rate - 1.0) * lambda_tot + energy_shift))
        #print('max count=',max_count,'end factor=',end_factor)
        b *= end_factor
        return b,counts

    def propagate_walkers_importance(self):
        # sample rotations 
        nterms = self.hamiltonian.nterms
        nw = self.walkers.nwalkers

        ovlp = self.walkers.compute_ovlp_ratio(self.hamiltonian)
        g = ovlp * self.hamiltonian.a[:,None]

        gp = g.copy()
        gp = xp.clip(gp,a_min=0.,a_max=None,out=gp)
        bp = gp.sum(axis=0)
        gm = g.copy()
        gm = xp.clip(gm,a_min=None,a_max=0.,out=gm)
        bm = -gm.sum(axis=0)
        gsum = g.sum(axis=0)
        assert xp.linalg.norm(bp-bm-gsum)<1e-10

        ixs = xp.nonzero(bm)[0]
        if ixs.size>0:
            print('bm,bp=',bm[ixs],bp[ixs])

        p = xp.fabs(g)
        sign = g/p
        b = p.sum(axis=0)
        assert xp.linalg.norm(bp+bm-b)<1e-10
        p = p/b[None,:] 

        ixs = xp.asarray([xp.random.choice(nterms,size=1,p=p[:,i])[0] for i in range(nw)])
        #print(ixs)
        self.hamiltonian.parse_samples(to_host(ixs))
        self.walkers.update_walkers(self.hamiltonian,self.trial)
    
        # 4.update weight
        b *= sign[ixs,xp.arange(nw)]
        return b,ixs

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
        self.walkers.save(comm,dirname)
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
