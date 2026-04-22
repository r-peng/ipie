import time

import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp
from ipie.walkers.pop_controller import (
        get_buffer,
        set_buffer,
        minimize_communication,
        comb,
        pair_branch,
        stochastic_reconfiguration,
)
from ipie.walkers.pop_controller import PopController as PopController_

class PopController(PopController_):
    def pop_control(self,walkers,comm):
        if self.method!='stochastic_reconfiguration':
            raise NotImplementedError
        self.timer.start_time()
        global_max = xp.amax(walkers.weight) 
        global_max = comm.reduce(global_max, op=MPI.MAX, root=0)
        global_max = comm.bcast(global_max, root=0)
        #global_max = 0.

        walkers.weight = xp.exp(walkers.weight-global_max)
        walkers.unscaled_weight = walkers.weight
        if self.method=='stochastic_reconfiguration':
            stochastic_reconfiguration(walkers, comm, self.timer, self.pop_control_counter)
            average_weight = walkers.weight[0]
            walkers.weight = 0. 
            average_weight = xp.log(average_weight)
            #print('global_max,log_ave',global_max,average_weight)
            return average_weight + global_max
        else:
            raise NotImplementedError

