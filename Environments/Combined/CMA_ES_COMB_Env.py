import g_utils
import gymnasium
import numpy as np
from collections import deque
from Environments.Combined.CMA_ES_COMB import CMAES_COMB


class CMA_ES_COMB(gymnasium.Env):
    def __init__(self, objective_funcs, x_start, sigma, reward_type):
        super(CMA_ES_COMB, self).__init__()
        self.cma_es = None
        self.objective_funcs = objective_funcs
        self.x_start = x_start
        self.sigma = sigma
        self.reward_type = reward_type
        self.curr_index = 0

        self.h = 40
        self.curr_sigma = sigma
        self.curr_c1, self.curr_cc, self.curr_chiN, self.curr_cmu, self.curr_cs, self.curr_mueff, self.curr_h_sigma = [0] * 7
        """self.hist_c1, self.hist_cc, self.hist_ChiN, self.hist_cmu, self.hist_cs, self.hist_fit_vals, self.hist_h_sigma, self.hist_mueff, self.hist_sigma = [deque(np.zeros(self.h), maxlen=self.h)] * 9"""
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=np.array([1e-4, 1e-3, 1, 1e-4, 1e-10, 2, 5e-2, 0]), high=np.array([2e-1, 1, 8, 1e-1, 1, 5, 10, 1]), dtype=np.float64
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(15 + self.h,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = np.power(10, 28)
        self._f_targets = []

        self.last_achieved = 0

    def step(self, action):
        new_c1, new_cc, new_ChiN, new_cmu, new_cs, new_mueff, new_sigma, new_h_sig = action

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objective_funcs[self.curr_index](x) for x in X]
        expert_h_sigma, x_old, arx = self.cma_es.tell(X, fit)
        self.cma_es.h_sig = new_h_sig
        expert_sigma, pc, ps, count_eval = self.cma_es.tell2(x_old, arx)

        self.last_achieved = np.min(fit)

        # Calculate reward (Turn minimization into maximization)
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
            """self.hist_c1.append(self.curr_c1)
            self.hist_cc.append(self.curr_cc)
            self.hist_ChiN.append(self.curr_chiN)
            self.hist_cmu.append(self.curr_cmu)
            self.hist_cs.append(self.curr_cs)
            self.hist_fit_vals.append(difference)
            self.hist_h_sigma.append(self.curr_h_sigma)
            self.hist_mueff.append(self.curr_mueff)
            self.hist_sigma.append(self.curr_sigma)"""

        new_state = np.concatenate(
            [
                np.array([new_c1]),
                np.array([new_cc]),
                np.array([new_ChiN]),
                np.array([new_cmu]),
                np.array([new_cs]),
                np.array([new_mueff]),
                np.array([new_sigma]),
                np.array([self.objective_funcs[self.curr_index % len(self.objective_funcs)].dimension]),
                np.array([np.linalg.norm(pc)]),
                np.array([np.linalg.norm(ps)]),
                np.array([np.power(np.sum(self.cma_es.params.weights), 2)]),
                np.array([np.sum(np.power(self.cma_es.params.weights, 2))]),
                np.array([count_eval]),
                np.array([new_h_sig]),
                np.array([np.linalg.norm(ps) / new_ChiN - 1]),
                np.array(self.hist_fit_vals)
            ]
        )

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objective_funcs):
            self.stop = True

        # Update iteration
        self.iteration += 1

        # Update variables
        self.curr_sigma = new_sigma
        self.curr_h_sigma = new_h_sig
        self.curr_c1 = new_c1
        self.curr_cc = new_cc
        self.curr_chiN = new_ChiN
        self.curr_cmu = new_cmu
        self.curr_cs = new_cs
        self.curr_mueff = new_mueff

        self.cma_es.sigma = new_sigma
        self.cma_es.params.c1 = new_c1
        self.cma_es.params.cc = new_cc
        self.cma_es.params.chiN = new_ChiN
        self.cma_es.params.cmu = new_cmu
        self.cma_es.params.cs = new_cs
        self.cma_es.params.mueff = new_mueff

        return new_state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None, verbose=1):
        if verbose > 0 and len(self.objective_funcs) > self.curr_index > 0:
            print(
                f"{(self.curr_index / len(self.objective_funcs) * 100):6.2f}% of training completed"
                f" | {self.objective_funcs[self.curr_index % len(self.objective_funcs) - 1].best_value():30.10f} optimum"
                f" | {self.last_achieved:30.10f} achieved"
                f" | {np.abs(self.objective_funcs[self.curr_index % len(self.objective_funcs) - 1].best_value() - self.last_achieved):30.18f} difference"
            )
        """self.hist_c1, self.hist_cc, self.hist_ChiN, self.hist_cmu, self.hist_cs, self.hist_fit_vals, self.hist_h_sigma, self.hist_mueff, self.hist_sigma = [deque(np.zeros(self.h),maxlen=self.h)] * 9"""
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
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
        self.cma_es = CMAES_COMB(x_start, self.sigma)
        self.curr_sigma = self.sigma
        self.curr_h_sigma = self.cma_es.h_sig
        self.curr_c1 = self.cma_es.params.c1
        self.curr_cc = self.cma_es.params.cc
        self.curr_chiN = self.cma_es.params.chiN
        self.curr_cmu = self.cma_es.params.cmu
        self.curr_cs = self.cma_es.params.cs
        self.curr_mueff = self.cma_es.params.mueff
        self.iteration = 0
        return (
            np.concatenate(
                [
                    np.array([self.curr_c1]),
                    np.array([self.curr_cc]),
                    np.array([self.curr_chiN]),
                    np.array([self.curr_cmu]),
                    np.array([self.curr_cs]),
                    np.array([self.curr_mueff]),
                    np.array([self.curr_sigma]),
                    np.array([self.objective_funcs[self.curr_index % len(self.objective_funcs)].dimension]),
                    np.array([np.linalg.norm(self.cma_es.pc)]),
                    np.array([np.linalg.norm(self.cma_es.ps)]),
                    np.array([np.power(np.sum(self.cma_es.params.weights), 2)]),
                    np.array([np.sum(np.power(self.cma_es.params.weights, 2))]),
                    np.zeros(3),
                    np.array(self.hist_fit_vals)
                ]
            ),
            {},
        )

    def render(self):
        pass
