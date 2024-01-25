import os
import numpy as np
from tqdm import tqdm
from collections import deque
from Parameters.CMA_ES_Parameters import CMAESParameters


def run_CMAES_COMB(objective_fct, x_start, sigma, h=40, f_limit=np.power(10, 28)):
    es = CMAES_COMB(x_start, sigma)
    start_state = np.array([es.params.c1, es.params.cc, es.params.chiN, es.params.cmu, es.params.cs, es.params.mueff,
                            sigma, objective_fct.dimension, np.linalg.norm(es.pc), np.linalg.norm(es.ps),
                            np.power(np.sum(es.params.weights), 2), np.sum(np.power(es.params.weights, 2))])
    observations, actions, dones = (
        [np.hstack((start_state, np.zeros(int(9 * h) + 3)))],
        [],
        [],
    )
    hist_c1, hist_cc, hist_ChiN, hist_cmu, hist_cs, hist_fit_vals, hist_h_sigma, hist_mueff, hist_sigmas = [deque(np.zeros(h), maxlen=h)] * 9
    iteration = 0
    curr_h_sig, curr_sigma = es.h_sig, sigma
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        new_sigma, new_h_sig, pc, ps, count_eval = es.tell(X, fit)
        es.sigma = new_sigma
        f_best = np.min(fit)
        if iteration > 0:
            difference = np.clip(
                np.log(np.abs((f_best - hist_fit_vals[len(hist_fit_vals) - 1]))),
                -f_limit,
                f_limit,
            )
            hist_c1.append(es.params.c1)
            hist_cc.append(es.params.cc)
            hist_ChiN.append(es.params.chiN)
            hist_cmu.append(es.params.cmu)
            hist_cs.append(es.params.cs)
            hist_fit_vals.append(difference)
            hist_h_sigma.append(curr_h_sig)
            hist_mueff.append(es.params.mueff)
            hist_sigmas.append(curr_sigma)
        observations.append(
            np.concatenate(
                [
                    np.array([es.params.c1]),
                    np.array([es.params.cc]),
                    np.array([es.params.chiN]),
                    np.array([es.params.cmu]),
                    np.array([es.params.cs]),
                    np.array([es.params.mueff]),
                    np.array([new_sigma]),
                    np.array([objective_fct.dimension]),
                    np.array([np.linalg.norm(pc)]),
                    np.array([np.linalg.norm(ps)]),
                    np.array([np.power(np.sum(es.params.weights), 2)]),
                    np.array([np.sum(np.power(es.params.weights, 2))]),
                    np.zeros(3),
                    np.array(hist_c1),
                    np.array(hist_cc),
                    np.array(hist_ChiN),
                    np.array(hist_cmu),
                    np.array(hist_cs),
                    np.array(hist_fit_vals),
                    np.array(hist_h_sigma),
                    np.array(hist_mueff),
                    np.array(hist_sigmas)
                ]
            )
        )
        actions.append([es.params.c1, es.params.cc, es.params.chiN, es.params.cmu, es.params.cs, es.params.mueff,
                        new_sigma, new_h_sig])
        curr_sigma = new_sigma
        curr_h_sig = new_h_sig
        dones.append(False)
        iteration += 1
    dones[-1] = True
    return np.array(observations), np.array(actions), np.array(dones)


def collect_expert_samples(
    dimension, instance, split, p_class, x_start, sigma, bbob_functions
):
    p_class = p_class if split == "classes" else -1
    if os.path.isfile(
        f"Environments/Combined/Samples/CMA_ES_COMB_Samples_{dimension}D_{instance}I_{p_class}C.npz"
    ):
        print("Loading expert samples...")
        data = np.load(
            f"Environments/Combined/Samples/CMA_ES_COMB_Samples_{dimension}D_{instance}I_{p_class}C.npz"
        )
        return data
    print("Collecting expert samples...")
    observations, actions, dones = [], [], []
    for function in tqdm(bbob_functions):
        _x_start = (
            np.zeros(function.dimension)
            if x_start == 0
            else np.random.uniform(-5, 5, function.dimension)
        )
        obs, act, done = run_CMAES_COMB(
            objective_fct=function,
            x_start=_x_start,
            sigma=sigma,
        )
        observations.extend(obs)
        actions.extend(act)
        dones.extend(done)
    np.savez(
        f"Environments/Combined/Samples/CMA_ES_COMB_Samples_{dimension}D_{instance}I_{p_class}C.npz",
        observations=observations,
        actions=actions,
        dones=dones,
    )
    return np.load(
        f"Environments/Combined/Samples/CMA_ES_COMB_Samples_{dimension}D_{instance}I_{p_class}C.npz"
    )

class CMAES_COMB:
    def __init__(self, x_start, sigma):
        N = len(x_start)
        self.params = CMAESParameters(N)
        self.max_f_evals = 1e3 * N**2

        # initializing dynamic state variables
        self.x_mean = x_start
        self.sigma = sigma
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.C = np.eye(N)
        self.inv_sqrt_C = np.eye(N)
        self.condition_number = 1
        self.updated_eval = 0
        self.count_eval = 0
        self.fit_vals = np.zeros(N)
        self.h_sig = 0

    def ask(self):
        self._update_Eigensystem()
        return self.x_mean + np.dot(
            self.sigma
            * np.sqrt(self.D)
            * np.random.randn(self.params.lam, len(self.D)),
            self.B.T,
        )

    def tell(self, arx, fit_vals):
        self.count_eval += len(fit_vals)
        N = len(self.x_mean)
        x_old = self.x_mean

        arx = arx[np.argsort(fit_vals)]
        self.fit_vals = np.sort(fit_vals)

        self.x_mean = np.sum(
            arx[0 : self.params.mu] * self.params.weights[: self.params.mu, None],
            axis=0,
        )

        # Update evolution paths
        self.ps = (1 - self.params.cs) * self.ps + np.sqrt(
            self.params.cs * (2 - self.params.cs) * self.params.mueff
        ) * np.dot(self.inv_sqrt_C, (self.x_mean - x_old)) / self.sigma
        expert_h_sig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.params.cs) ** (2 * self.count_eval / self.params.lam)
        ) / self.params.chiN < 1.4 + 2 / (N + 1)
        self.pc = (1 - self.params.cc) * self.pc + self.h_sig * np.sqrt(
            self.params.cc * (2 - self.params.cc) * self.params.mueff
        ) * (self.x_mean - x_old) / self.sigma

        # Adapt covariance matrix C
        ar_temp = (arx[0 : self.params.mu] - x_old) / self.sigma
        self.C = (
            (1 - self.params.c1 - self.params.cmu) * self.C
            + self.params.c1
            * (
                np.outer(self.pc, self.pc)
                + (1 - self.h_sig) * self.params.cc * (2 - self.params.cc) * self.C
            )
            + self.params.cmu * ar_temp.T.dot(np.diag(self.params.weights)).dot(ar_temp)
        )

        # Adapt step-size sigma
        expert_sigma = self.sigma * np.exp(
            (self.params.cs / self.params.damps)
            * (np.linalg.norm(self.ps) / self.params.chiN - 1)
        )

        return expert_sigma, expert_h_sig, self.pc, self.ps, self.count_eval

    def stop(self):
        res = {}
        if self.count_eval <= 0:
            return res
        if self.count_eval >= self.max_f_evals:
            res["maxfevals"] = self.max_f_evals
        if self.condition_number > 1e14:
            res["condition"] = self.condition_number
        if len(self.fit_vals) > 1 and self.fit_vals[-1] - self.fit_vals[0] < 1e-12:
            res["tolfun"] = 1e-12
        if self.sigma * np.sqrt(max(self.D)) < 1e-11:
            res["tolx"] = 1e-11
        return res

    def _update_Eigensystem(self):
        if self.count_eval >= self.updated_eval + self.params.lazy_gap_evals:
            return
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        self.D, self.B = np.linalg.eigh(self.C)
        self.condition_number = max(self.D) / min(self.D)
        self.inv_sqrt_C = np.dot(
            np.dot(self.B, np.diag(1.0 / np.sqrt(self.D))), self.B.T
        )
        self.updated_eval = self.count_eval