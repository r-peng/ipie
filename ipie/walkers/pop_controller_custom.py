import time

import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp
from ipie.walkers.pop_controller import (
        #get_buffer,
        #set_buffer,
        #minimize_communication,
        #comb,
        #pair_branch,
        stochastic_reconfiguration,
)
from ipie.walkers.pop_controller import PopController as PopController_

class PopController(PopController_):
    def pop_control(self,walkers,comm):
        if self.method!='stochastic_reconfiguration':
            super().pop_control(walkers,comm)
            return None
        #self.timer.start_time()
        walkers.phase = xp.sign(walkers.weight)
        walkers.weight = xp.fabs(walkers.weight)

        global_max = xp.amax(walkers.weight) 
        global_max = comm.reduce(global_max, op=MPI.MAX, root=0)
        global_max = comm.bcast(global_max, root=0)
        #global_max = 0.

        walkers.weight /= global_max
        walkers.unscaled_weight = walkers.weight
        stochastic_reconfiguration(walkers, comm, self.timer, self.pop_control_counter)
        average_weight = walkers.weight[0]
        walkers.weight = walkers.phase
        #print('global_max,ave',global_max,average_weight)
        return xp.log(average_weight * global_max)

