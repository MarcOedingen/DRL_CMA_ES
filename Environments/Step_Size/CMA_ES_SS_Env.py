import gymnasium
import numpy as np
from collections import deque
from Environments.Step_Size.CMA_ES_SS import CMAES

def _norm(x):
    return np.sqrt(np.sum(np.square(x)))


class CMA_ES_SS(gymnasium.Env):
    def __init__(self, objetive_funcs, sigma):
        super(CMA_ES_SS, self).__init__()
        self.cma_es = None
        self.objetive_funcs = objetive_funcs
        self.sigma = sigma
        self.curr_index = 0

        self.h = 40
        self.curr_sigma = sigma
        self.curr_ps = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_sigmas = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(low=1e-5, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=0, high=5, shape=(2 + 2 * self.h,))

    def step(self, action):
        self.curr_sigma = action[0]
        self.cma_es.sigma = self.curr_sigma

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objetive_funcs[self.curr_index](x) for x in X]
        new_ps = self.cma_es.tell(X, fit)

        # Calculate reward (Turning maximization into minimization)
        reward = -np.mean(fit)

        # Check if the algorithm should stop
        terminated = self.curr_index >= len(self.objetive_funcs) - 1
        truncated = bool(self.cma_es.stop())

        # Update history
        # insert just the reward if nothing is in the history, otherwise insert the difference between the current
        # reward and the last reward
        if len(self.hist_fit_vals) == 0:
            self.hist_fit_vals.append(reward)
        else:
            self.hist_fit_vals.append(abs(reward - self.hist_fit_vals[-1]))
        self.hist_sigmas.append(self.curr_sigma)

        new_state = np.concatenate([np.array([self.curr_sigma]), np.array([_norm(new_ps) / self.cma_es.params.chiN - 1]), np.array(self.hist_fit_vals), np.array(self.hist_sigmas)], dtype=np.float32)

        # Update current ps
        self.curr_ps = new_ps

        # Update current index
        self.curr_index = (self.curr_index + 1) % len(self.objetive_funcs)

        return new_state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        self.curr_sigma = self.sigma
        self.curr_ps = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_sigmas = deque(np.zeros(self.h), maxlen=self.h)
        self.cma_es = CMAES(np.random.uniform(low=-5, high=5, size=self.objetive_funcs[self.curr_index].dimension),
                            self.sigma)
        return np.concatenate(
            [np.array([self.curr_sigma]), np.array([self.curr_ps]), list(self.hist_fit_vals), list(self.hist_sigmas)]), {}

    def render(self):
        pass


