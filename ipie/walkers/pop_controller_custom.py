import time

import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp
#from ipie.walkers.pop_controller import stochastic_reconfiguration
from ipie.walkers.pop_controller import (
        get_buffer,
        set_buffer,
        minimize_communication,
)
from ipie.walkers.pop_controller import PopController as PopController_

class PopController(PopController_):
    def pop_control(self,walkers,comm):
        if self.method!='stochastic_reconfiguration':
            #raise NotImplementedError
            super().pop_control(walkers,comm)
            self.pop_control_counter += 1
            return None
        #self.timer.start_time()
        walkers.phase = xp.sign(walkers.weight)
        walkers.weight = xp.fabs(walkers.weight)
        w = walkers.weight
        walkers.unscaled_weight = walkers.weight
        average_weight,Neff,Ndistinct = stochastic_reconfiguration(walkers, comm, self.timer, pop_control_counter=self.pop_control_counter)
        walkers.weight = walkers.phase
        if self.pop_control_counter == 0:
            self.Neff = []
            self.Ndistinct = []
        self.Neff.append(Neff)
        self.Ndistinct.append(Ndistinct)
        self.pop_control_counter += 1
        return numpy.log(average_weight)

def stochastic_reconfiguration(
    walkers,
    comm,
    timer,
    pop_control_counter=0,
    store_walkermap=False,
    read_walkermap=False,
    walkermap_file=None,
):
    timer.start_time()
    nwalkers = walkers.nwalkers
    local_weight = walkers.weight.get() if hasattr(walkers.weight, "get") else walkers.weight
    global_weight = None
    if comm.rank == 0:
        global_weight = numpy.zeros((comm.size, nwalkers), dtype=local_weight.dtype)
    timer.add_non_communication()

    timer.start_time()
    comm.Gather(local_weight, global_weight, root=0)
    timer.add_communication()

    # perform sr on the root
    timer.start_time()
    new_average_weight = None
    if comm.rank == 0:
        cumulative_weights = numpy.cumsum(abs(global_weight))
        total_weight = cumulative_weights[-1]
        new_average_weight = total_weight / nwalkers / comm.size
        if not read_walkermap:
            zeta = numpy.random.rand()
            new_indices = numpy.zeros(comm.size * nwalkers, dtype=numpy.int64)
            for i in range(comm.size * nwalkers):
                z = (i + zeta) / nwalkers / comm.size
                new_indices[i] = numpy.searchsorted(cumulative_weights, z * total_weight)
            reordered_indices, _ = minimize_communication(new_indices)
            if store_walkermap:
                assert walkermap_file is not None, "Must provide filename to store the walker map."
                with h5py.File(walkermap_file, "a") as f:
                    name = f"walker_map_{pop_control_counter}"
                    if name in f:
                        f[name][...] = reordered_indices
                    else:
                        f.create_dataset(name, data=reordered_indices)
        else:
            assert walkermap_file is not None, "Must provide filename to read the walker map."
            with h5py.File(walkermap_file, "r") as f:
                reordered_indices = f[f"walker_map_{pop_control_counter}"][:]
    timer.add_non_communication()

    timer.start_time()
    glob_inf = None
    Neff = None
    Ndistinct = None
    if comm.rank == 0:
        ntot = nwalkers*comm.size
        Neff = float(total_weight**2/(global_weight**2).sum()/ntot)
        Ndistinct = numpy.unique(reordered_indices).size/ntot

        glob_indices = numpy.arange(comm.size * nwalkers, dtype=numpy.int64)
        mask = reordered_indices != glob_indices
        sendidx = reordered_indices[mask]
        destidx = glob_indices[mask]
        glob_inf = numpy.column_stack((sendidx, destidx))

    timer.add_non_communication()
    timer.start_time()
    glob_inf = comm.bcast(glob_inf, root=0)
    new_average_weight = comm.bcast(new_average_weight, root=0)
    timer.add_communication()

    timer.start_time()
    local_sends = [glob_inf[(glob_inf[:, 0] // nwalkers == i)] for i in range(comm.size)]
    local_recvs = [glob_inf[(glob_inf[:, 1] // nwalkers == i)] for i in range(comm.size)]
    num_local_sends = numpy.array([len(s) for s in local_sends])
    cumsum_local_sends = numpy.cumsum(num_local_sends) - num_local_sends
    num_local_recvs = numpy.array([len(r) for r in local_recvs])
    cumsum_local_recvs = numpy.cumsum(num_local_recvs) - num_local_recvs

    buflis = {}
    local_send = local_sends[comm.rank]
    local_send_loc_idx = local_send[:, 0] % nwalkers
    local_recv = local_recvs[comm.rank]
    for i in range(nwalkers):
        if i in local_send_loc_idx:
            buflis[i] = get_buffer(walkers, i)
    timer.add_non_communication()
    comm.barrier()
    send_reqs = []
    for isend, (src_idx, dest_idx) in enumerate(local_send):
        src_loc = src_idx % nwalkers
        dest_rk = dest_idx // nwalkers
        tag = isend + cumsum_local_sends[comm.rank]

        buf = buflis[src_loc]
        req = comm.Issend(buf, dest=int(dest_rk), tag=int(tag))
        send_reqs.append(req)

    # Post all nonblocking recvs, saving a Status for each to inspect later
    walker_len = get_buffer(walkers, 0).shape[0]
    recv_reqs = []
    for irecv, (src_idx, dest_idx) in enumerate(local_recv):
        iw = dest_idx % nwalkers
        src_rank = src_idx // nwalkers
        tag_recv = irecv + cumsum_local_recvs[comm.rank]

        recv_buf = numpy.empty(walker_len, dtype=numpy.complex128)
        status = MPI.Status()
        req = comm.Irecv(recv_buf, source=int(src_rank), tag=int(tag_recv))
        recv_reqs.append((iw, recv_buf, status, req))

    # Wait on recvs and inspect their Status
    for iw, buf, status, req in recv_reqs:
        req.Wait(status)
        set_buffer(walkers, iw, buf)

    # 4) Wait on sends
    MPI.Request.Waitall(send_reqs)

    comm.Barrier()

    timer.start_time()
    #walkers.weight[:] = new_average_weight
    timer.add_non_communication()
    return new_average_weight,Neff,Ndistinct
