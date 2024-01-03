import gymnasium
import numpy as np
from collections import deque
from Environments.Evolution_Paths.CMA_ES_EP import CMAES_EP


class CMA_ES_EP(gymnasium.Env):

    def __init__(self, objective_funcs, x_start, sigma):
        super(CMA_ES_EP, self).__init__()
        self.cma_es = None
        self.objective_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.curr_index = 0

        self.h = 40
        self.curr_sigma = sigma
        self.curr_ps = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_ps = deque(np.zeros(self.h), maxlen=self.h)