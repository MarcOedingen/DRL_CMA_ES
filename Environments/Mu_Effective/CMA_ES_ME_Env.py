import gymnasium
import numpy as np
from collections import deque
from Environments.Mu_Effective.CMA_ES_ME import CMAES_ME


class CMA_ES_ME(gymnasium.Env):
    def __init__(self, objective_funcs, x_start, sigma):
        super(CMA_ES_ME, self).__init__()
        self.cma_es = None
        self.objetive_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.curr_index = 0

        self.h = 40
        self.curr_mueff = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_mueff = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=2, high=5, shape=(1,), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + 2 * self.h,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = np.power(10, 28)

        self._last_achieved = 0

    def step(self, action):
        new_mueff = action[0]
        self.cma_es.params.mueff = new_mueff

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objetive_funcs[self.curr_index](x) for x in X]
        self.cma_es.tell(X, fit)

        self._last_achieved = np.min(fit)

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
            self.hist_mueff.append(self.curr_mueff)

        new_state = np.concatenate(
            [
                np.array([new_mueff]),
                np.array([np.power(np.sum(self.cma_es.params.weights), 2)]),
                np.array([np.sum(np.power(self.cma_es.params.weights, 2))]),
                np.array(self.hist_fit_vals),
                np.array(self.hist_mueff),
            ]
        )

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objetive_funcs):
            self.stop = True

        # Update iteration
        self.iteration += 1

        # Update variables
        self.curr_mueff = new_mueff

        return new_state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None, verbose=1):
        if verbose > 0 and self.curr_index < len(self.objetive_funcs):
            print(
                f"{(self.curr_index / len(self.objetive_funcs) * 100):6.2f}% of training completed"
                f" | {self.objetive_funcs[self.curr_index % len(self.objetive_funcs) - 1].best_value():30.10f} optimum"
                f" | {self._last_achieved:30.10f} achieved"
                f" | {np.abs(self.objetive_funcs[self.curr_index % len(self.objetive_funcs) - 1].best_value() - self._last_achieved):30.18f} difference"
            )
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_damps = deque(np.zeros(self.h), maxlen=self.h)
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
        self.cma_es = CMAES_ME(x_start, self.sigma)
        self.curr_mueff = self.cma_es.params.mueff
        self.iteration = 0
        return (
            np.concatenate(
                [
                    np.array([self.curr_mueff]),
                    np.array([np.power(np.sum(self.cma_es.params.weights), 2)]),
                    np.array([np.sum(np.power(self.cma_es.params.weights, 2))]),
                    np.array(self.hist_fit_vals),
                    np.array(self.hist_damps),
                ]
            ),
            {},
        )

    def render(self):
        pass
