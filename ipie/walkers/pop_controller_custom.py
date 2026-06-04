import time

import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp
from ipie.walkers.pop_controller import stochastic_reconfiguration
from ipie.walkers.pop_controller import PopController as PopController_

class PopController(PopController_):
    def pop_control(self,walkers,comm):
        if self.method!='stochastic_reconfiguration':
            super().pop_control(walkers,comm)
            self.pop_control_counter += 1
            return None
        #self.timer.start_time()
        walkers.phase = xp.sign(walkers.weight)
        walkers.weight = xp.fabs(walkers.weight)
        w = walkers.weight
        #print(f'wmean={xp.mean(w)},max={xp.amax(w)/w.sum()},Neff={w.sum()**2/(w**2).sum()}')
        walkers.unscaled_weight = walkers.weight
        stochastic_reconfiguration(walkers, comm, timer=self.timer, pop_control_counter=self.pop_control_counter)
        average_weight = walkers.weight[0]
        walkers.weight = walkers.phase
        self.pop_control_counter += 1
        return xp.log(average_weight)

