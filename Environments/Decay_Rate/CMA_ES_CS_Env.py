import gymnasium
import numpy as np
from collections import deque
from Environments.Decay_Rate.CMA_ES_CS import CMAES_CS


class CMA_ES_CS(gymnasium.Env):
    def __init__(self, objective_funcs, x_start, sigma):
        super(CMA_ES_CS, self).__init__()
        self.cma_es = None
        self.objetive_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.curr_index = 0

        self.h = 40
        self.curr_sigma = sigma
        self.curr_ps = 0
        self.curr_cs = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cs = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=1e-10, high=1, shape=(1,), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + 2 * self.h,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = np.power(10, 28)

    def step(self, action):
        self.curr_cs = action[0]
        self.cma_es.params.cs = self.curr_cs

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objetive_funcs[self.curr_index](x) for x in X]
        self.curr_ps, self.curr_sigma = self.cma_es.tell(X, fit)

        # Calculate reward (Turn minimization into maximization)
        reward = -np.min(fit)
        reward = np.clip(reward, -self._f_limit, self._f_limit)

        # Check if the algorithm should stop
        # Terminated if all functions have been evaluated
        terminated = self.stop = self.curr_index >= len(self.objetive_funcs)
        # Truncated if the current function has been evaluated
        truncated = bool(self.cma_es.stop())

        # Update history
        if self.iteration > 0:
            difference = np.clip(
                np.abs((reward - self.hist_fit_vals[len(self.hist_fit_vals) - 1])),
                -self._f_limit,
                self._f_limit,
            )
            self.hist_fit_vals.append(difference / reward)
            self.hist_cs.append(self.curr_cs)

        new_state = np.concatenate(
            [
                np.array([self.curr_sigma]),
                np.array([self.curr_cs]),
                np.array([np.linalg.norm(self.curr_ps) / self.cma_es.params.chiN - 1]),
                np.array(self.hist_fit_vals),
                np.array(self.hist_cs),
            ]
        )

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objetive_funcs):
            self.stop = True

        # Update iteration
        self.iteration += 1

        return new_state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None, verbose=1):
        if verbose > 0 and self.curr_index < len(self.objetive_funcs):
            print(
                f"Training on function {self.objetive_funcs[self.curr_index % len(self.objetive_funcs)].function} | {(self.curr_index / len(self.objetive_funcs)) * 100}% done"
            )
        self.curr_sigma = self.sigma
        self.curr_ps = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cs = deque(np.zeros(self.h), maxlen=self.h)
        x_start = (
            np.zeros(
                self.objetive_funcs[
                    self.curr_index % len(self.objetive_funcs)
                ].dimension
            )
            if self.x_start == 0
            else np.random.uniform(
                low=-5,
                high=5,
                size=self.objetive_funcs[
                    self.curr_index % len(self.objetive_funcs)
                ].dimension,
            )
        )
        self.cma_es = CMAES_CS(x_start, self.curr_sigma)
        self.curr_cs = self.cma_es.params.cs
        self.iteration = 0
        return (
            np.concatenate(
                [
                    np.array([self.curr_sigma]),
                    np.array([self.curr_cs]),
                    np.array([self.curr_ps]),
                    np.array(self.hist_fit_vals),
                    np.array(self.hist_cs),
                ]
            ),
            {},
        )

    def render(self):
        pass
