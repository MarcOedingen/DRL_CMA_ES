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
        self.curr_ChiN = 0
        self.curr_cc = 0
        self.curr_cs = 0
        self.curr_c1 = 0
        self.curr_cmu = 0
        self.curr_mueff = 0
        self.curr_sigma = sigma
        self.curr_h_sig = 0
        self.hist_fit_vals = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_ChiN = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cc = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cs = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_c1 = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cmu = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_mueff = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_sigma = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_h_sig = deque(np.zeros(self.h), maxlen=self.h)

        self.action_space = gymnasium.spaces.Box(
            low=np.array([1, 1e-3, 1e-10, 1e-4, 1e-4, 2, 5e-2, 0]), high=np.array([8, 1, 1, 0.2, 0.1, 5, 10, 1]), dtype=np.float64
        )

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9 + 9 * self.h,), dtype=np.float64
        )

        self.iteration = 0
        self.stop = False
        self._f_limit = np.power(10, 28)
        self._f_targets = []

        self.last_achieved = 0

    def step(self, action):
        new_ChiN, new_cc, new_cs, new_c1, new_cmu, new_mueff, new_sigma, new_h_sig = action[0], action[1], action[2], action[3], action[4], action[5], action[6], action[7]
        self.cma_es.params.chiN = new_ChiN
        self.cma_es.params.cc = new_cc
        self.cma_es.params.cs = new_cs
        self.cma_es.params.c1 = new_c1
        self.cma_es.params.cmu = new_cmu
        self.cma_es.params.mueff = new_mueff
        self.cma_es.h_sig = 1 if new_h_sig > 0.5 else 0

        # Run one iteration of CMA-ES
        X = self.cma_es.ask()
        fit = [self.objective_funcs[self.curr_index](x) for x in X]
        _, old_x, arx = self.cma_es.tell(X, fit)
        _, ps, pc = self.cma_es.tell2(old_x, arx)

        self.last_achieved = np.min(fit)

        reward = g_utils.calc_reward(
            optimum=self.objective_funcs[self.curr_index].best_value(),
            min_eval=self.last_achieved,
            reward_type=self.reward_type,
            reward_targets=self._f_targets,
        )

        terminated = self.stop = self.curr_index >= len(self.objective_funcs)
        truncated = bool(self.cma_es.stop())

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
            self.hist_ChiN.append(self.curr_ChiN)
            self.hist_cc.append(self.curr_cc)
            self.hist_cs.append(self.curr_cs)
            self.hist_c1.append(self.curr_c1)
            self.hist_cmu.append(self.curr_cmu)
            self.hist_mueff.append(self.curr_mueff)
            self.hist_sigma.append(self.curr_sigma)
            self.hist_h_sig.append(self.curr_h_sig)

        new_state = np.concatenate(
            [
                np.array([new_ChiN]),
                np.array([new_cc]),
                np.array([new_cs]),
                np.array([new_c1]),
                np.array([new_cmu]),
                np.array([new_mueff]),
                np.array([new_sigma]),
                np.array([new_h_sig]),
                np.array([self.objective_funcs[self.curr_index % len(self.objective_funcs)].dimension]),
                np.array(self.hist_fit_vals),
                np.array(self.hist_ChiN),
                np.array(self.hist_cc),
                np.array(self.hist_cs),
                np.array(self.hist_c1),
                np.array(self.hist_cmu),
                np.array(self.hist_mueff),
                np.array(self.hist_sigma),
                np.array(self.hist_h_sig)
            ]
        )

        if truncated:
            self.curr_index += 1

        if self.curr_index >= len(self.objective_funcs):
            self.stop = True

        self.iteration += 1

        self.curr_ChiN = new_ChiN
        self.curr_cc = new_cc
        self.curr_cs = new_cs
        self.curr_c1 = new_c1
        self.curr_cmu = new_cmu
        self.curr_mueff = new_mueff
        self.curr_sigma = new_sigma
        self.curr_h_sig = new_h_sig

        self.cma_es.sigma = new_sigma

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
        self.hist_ChiN = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cc = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cs = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_c1 = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_cmu = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_mueff = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_sigma = deque(np.zeros(self.h), maxlen=self.h)
        self.hist_h_sig = deque(np.zeros(self.h), maxlen=self.h)
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
        self.curr_ChiN = self.cma_es.params.chiN
        self.curr_cc = self.cma_es.params.cc
        self.curr_cs = self.cma_es.params.cs
        self.curr_c1 = self.cma_es.params.c1
        self.curr_cmu = self.cma_es.params.cmu
        self.curr_mueff = self.cma_es.params.mueff
        self.curr_sigma = self.sigma
        self.curr_h_sig = 0
        self.iteration = 0
        return (
            np.concatenate(
                [
                    np.array([self.curr_ChiN]),
                    np.array([self.curr_cc]),
                    np.array([self.curr_cs]),
                    np.array([self.curr_c1]),
                    np.array([self.curr_cmu]),
                    np.array([self.curr_mueff]),
                    np.array([self.curr_sigma]),
                    np.array([self.curr_h_sig]),
                    np.array([self.objective_funcs[self.curr_index % len(self.objective_funcs)].dimension]),
                    np.array(self.hist_fit_vals),
                    np.array(self.hist_ChiN),
                    np.array(self.hist_cc),
                    np.array(self.hist_cs),
                    np.array(self.hist_c1),
                    np.array(self.hist_cmu),
                    np.array(self.hist_mueff),
                    np.array(self.hist_sigma),
                    np.array(self.hist_h_sig)
                ]
            ),
            {},
        )

    def render(self):
        pass
