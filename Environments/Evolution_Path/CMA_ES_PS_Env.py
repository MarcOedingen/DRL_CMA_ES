import gymnasium
import numpy as np
from collections import deque
from Environments.Evolution_Path.CMA_ES_PS import CMAES_PS


class CMA_ES_PS(gymnasium.Env):
    def __init__(self, objective_funcs, x_start, sigma):
        super(CMA_ES_PS, self).__init__()
        self.cma_es = None
        self.objetive_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.curr_index = 0

        self.h = 40
        self.curr_ps = np.zeros(40)
        self.hist_fit_vals = deque(np.zeros(40), maxlen=self.h)
        self.hist_ps = deque([np.zeros(40) for _ in range(40)], maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=-5, high=5, shape=(40,), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1 + 40 * 43,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = np.power(10, 28)

        self._last_achieved = 0

    def step(self, action):
        new_ps = action[: self.objetive_funcs[self.curr_index].dimension]

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objetive_funcs[self.curr_index](x) for x in X]
        intermediate_ps, _, x_old, arx = self.cma_es.tell1(X, fit)

        # Update variables
        self.cma_es.ps = new_ps

        self.cma_es.tell2(arx=arx, x_old=x_old, N=self.objetive_funcs[self.curr_index].dimension)

        self._last_achieved = np.min(fit)

        # Calculate reward (Turn minimization into maximization)
        reward = -np.log(np.abs(np.min(fit) - self.objetive_funcs[self.curr_index].best_value()))
        reward = np.clip(reward, -self._f_limit, self._f_limit)

        # Check if the algorithm should stop
        # Terminated if all functions have been evaluated
        terminated = self.stop = self.curr_index >= len(self.objetive_funcs)
        # Truncated if the current function has been evaluated
        truncated = bool(self.cma_es.stop())

        # Update history
        pad_size = 40 - self.objetive_funcs[self.curr_index].dimension
        if self.iteration > 0:
            difference = np.clip(
                np.log(np.abs((self._last_achieved - self.hist_fit_vals[len(self.hist_fit_vals) - 1]))),
                -self._f_limit,
                self._f_limit,
            )
            self.hist_fit_vals.append(difference / self._last_achieved)
        self.hist_ps.append(np.pad(self.curr_ps, (0, pad_size), "constant"))

        new_state = np.concatenate(
            [
                np.array([self.objetive_funcs[self.curr_index].dimension]),
                np.pad(intermediate_ps, (0, pad_size), "constant") if pad_size > 0 else intermediate_ps,
                np.pad(new_ps, (0, pad_size), "constant") if pad_size > 0 else new_ps,
                np.array(self.hist_fit_vals),
                np.array(self.hist_ps).flatten(),
            ]
        )

        new_state = np.clip(new_state, -self._f_limit, self._f_limit)

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objetive_funcs):
            self.stop = True

        # Update iteration
        self.iteration += 1

        # Update current ps
        self.curr_ps = new_ps

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
        self.hist_ps = deque([np.zeros(40) for _ in range(40)], maxlen=self.h)
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
        self.cma_es = CMAES_PS(x_start, self.sigma)
        self.iteration = 0
        self.curr_ps = np.zeros(self.objetive_funcs[self.curr_index % len(self.objetive_funcs)].dimension)
        return (
            np.concatenate(
                [
                    np.array([self.objetive_funcs[self.curr_index % len(self.objetive_funcs)].dimension]),
                    np.zeros(40 * 43),
                ]
            ),
            {},
        )

    def render(self):
        pass
