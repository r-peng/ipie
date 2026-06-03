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
        walkers.unscaled_weight = walkers.weight
        stochastic_reconfiguration(walkers, comm, self.timer, self.pop_control_counter)
        average_weight = walkers.weight[0]
        walkers.weight = walkers.phase
        return xp.log(average_weight)

