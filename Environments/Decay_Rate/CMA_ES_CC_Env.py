import g_utils
import gymnasium
import numpy as np
from collections import deque
from Environments.Decay_Rate.CMA_ES_CC import CMAES_CC


class CMA_ES_CC(gymnasium.Env):
    def __init__(self, objective_funcs, x_start, sigma, reward_type):
        super(CMA_ES_CC, self).__init__()
        self.cma_es = None
        self.objective_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.reward_type = reward_type
        self.curr_index = 0

        self.h = 40
        self.curr_cc = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cc = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=1e-3, high=1, shape=(1,), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 + 2 * self.h,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = 4.6*np.power(10, 18)
        self._f_targets = []

        self.last_achieved = 0

    def step(self, action):
        new_cc = action[0]
        self.cma_es.params.cc = new_cc

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objective_funcs[self.curr_index](x) for x in X]
        pc = self.cma_es.tell(X, fit)

        self.last_achieved = np.min(fit)
        reward = g_utils.calc_reward(
            optimum=self.objective_funcs[self.curr_index].best_value(),
            min_eval=self.last_achieved,
            reward_type=self.reward_type,
            reward_targets=self._f_targets,
        )
        reward = np.clip(reward, -self._f_limit, self._f_limit)

        # Check if the algorithm should stop
        # Terminated if all functions have been evaluated
        terminated = self.stop = self.curr_index >= len(self.objective_funcs)
        # Truncated if the current function has been evaluated
        truncated = bool(self.cma_es.stop())

        # Update history
        if self.iteration > 0:
            difference = np.clip(
                np.log(
                    np.abs(
                        (
                            self.last_achieved
                            - self.hist_fit_vals[len(self.hist_fit_vals) - 1]
                        )
                    )
                ),
                -self._f_limit,
                self._f_limit,
            )
            self.hist_fit_vals.append(difference)
            self.hist_cc.append(self.curr_cc)

        new_state = np.concatenate(
            [
                np.array([new_cc]),
                np.array([self.objective_funcs[self.curr_index].dimension]),
                np.array([np.linalg.norm(pc)]),
                np.array(self.hist_fit_vals),
                np.array(self.hist_cc),
            ]
        )

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objective_funcs):
            self.stop = True

        # Update iteration
        self.iteration += 1

        # Update variables
        self.curr_cc = new_cc

        return new_state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None, verbose=1):
        if verbose > 0 and len(self.objective_funcs) > self.curr_index > 0:
            print(
                f"{(self.curr_index / len(self.objective_funcs) * 100):6.2f}% of training completed"
                f" | {self.objective_funcs[self.curr_index % len(self.objective_funcs) - 1].best_value():30.10f} optimum"
                f" | {self.last_achieved:30.10f} achieved"
                f" | {np.abs(self.objective_funcs[self.curr_index % len(self.objective_funcs) - 1].best_value() - self.last_achieved):30.18f} difference"
            )
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cc = deque(np.zeros(self.h), maxlen=self.h)
        self._f_targets = g_utils.set_reward_targets(
            self.objective_funcs[
                self.curr_index % len(self.objective_funcs)
            ].best_value()
        )
        x_start = (
            np.zeros(
                self.objective_funcs[
                    self.curr_index % len(self.objective_funcs)
                ].dimension
            )
            if self.x_start == 0
            else np.random.uniform(
                low=-5,
                high=5,
                size=self.objective_funcs[
                    self.curr_index % len(self.objective_funcs)
                ].dimension,
            )
        )
        self.cma_es = CMAES_CC(x_start, self.sigma)
        self.curr_cc = self.cma_es.params.cc
        self.iteration = 0
        return (
            np.concatenate(
                [
                    np.array([self.curr_cc]),
                    np.array([len(x_start)]),
                    np.array([0]),
                    np.array(self.hist_fit_vals),
                    np.array(self.hist_cc),
                ]
            ),
            {},
        )

    def render(self):
        pass
