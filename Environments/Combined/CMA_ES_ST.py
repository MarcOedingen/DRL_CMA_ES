import os
import numpy as np
from tqdm import tqdm
from collections import deque
from Parameters.CMA_ES_Parameters import CMAESParameters


def run_CMAES_ST(objective_fct, x_start, sigma, h=40, f_limit=4.6 * np.power(10, 18)):
    es = CMAES_ST(x_start, sigma)
    start_state = np.array(
        [
            es.params.chiN,
            es.params.damps,
            es.params.cs,
            es.params.cc,
            es.params.c1,
            es.params.cmu,
            es.params.mueff,
            objective_fct.dimension,
        ]
    )
    observations, actions, dones = [np.hstack((start_state, np.zeros(8 * 40)))], [], []
    hist_fit_vals = deque(np.zeros(h), maxlen=h)
    hist_chiN = deque(np.zeros(h), maxlen=h)
    hist_damps = deque(np.zeros(h), maxlen=h)
    hist_cs = deque(np.zeros(h), maxlen=h)
    hist_cc = deque(np.zeros(h), maxlen=h)
    hist_c1 = deque(np.zeros(h), maxlen=h)
    hist_cmu = deque(np.zeros(h), maxlen=h)
    hist_mueff = deque(np.zeros(h), maxlen=h)
    iteration = 0
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        sigma, ps, pc = es.tell(X, fit)
        f_best = np.min(fit)
        if iteration > 0:
            difference = np.clip(
                np.log(np.abs((f_best - hist_fit_vals[len(hist_fit_vals) - 1]))),
                -f_limit,
                f_limit,
            )
            hist_fit_vals.append(difference)
            hist_chiN.append(es.params.chiN)
            hist_damps.append(es.params.damps)
            hist_cs.append(es.params.cs)
            hist_cc.append(es.params.cc)
            hist_c1.append(es.params.c1)
            hist_cmu.append(es.params.cmu)
            hist_mueff.append(es.params.mueff)
        observations.append(
            np.concatenate(
                [
                    np.array([es.params.chiN]),
                    np.array([es.params.damps]),
                    np.array([es.params.cs]),
                    np.array([es.params.cc]),
                    np.array([es.params.c1]),
                    np.array([es.params.cmu]),
                    np.array([es.params.mueff]),
                    np.array([objective_fct.dimension]),
                    np.array(hist_fit_vals),
                    np.array(hist_chiN),
                    np.array(hist_damps),
                    np.array(hist_cs),
                    np.array(hist_cc),
                    np.array(hist_c1),
                    np.array(hist_cmu),
                    np.array(hist_mueff),
                ]
            )
        )
        actions.append(
            [
                es.params.chiN,
                es.params.damps,
                es.params.cs,
                es.params.cc,
                es.params.c1,
                es.params.cmu,
                es.params.mueff,
            ]
        )
        dones.append(False)
        iteration += 1
    dones[-1] = True
    return np.array(observations), np.array(actions), np.array(dones)


def collect_expert_samples(
    dimension, instance, split, p_class, x_start, sigma, bbob_functions
):
    p_class = p_class if split == "classes" else -1
    if os.path.isfile(
        f"Environments/Combined/Samples/CMA_ES_ST_Samples_{dimension}D_{instance}I_{p_class}C.npz"
    ):
        data = np.load(
            f"Environments/Combined/Samples/CMA_ES_ST_Samples_{dimension}D_{instance}I_{p_class}C.npz"
        )
        return data
    observations, actions, dones = [], [], []
    for function in tqdm(bbob_functions):
        _x_start = (
            np.zeros(function.dimension)
            if x_start == 0
            else np.random.uniform(-5, 5, function.dimension)
        )
        obs, act, done = run_CMAES_ST(
            objective_fct=function,
            x_start=_x_start,
            sigma=sigma,
        )
        observations.extend(obs)
        actions.extend(act)
        dones.extend(done)
    np.savez(
        f"Environments/Combined/Samples/CMA_ES_ST_Samples_{dimension}D_{instance}I_{p_class}C.npz",
        observations=observations,
        actions=actions,
        dones=dones,
    )
    return np.load(
        f"Environments/Combined/Samples/CMA_ES_ST_Samples_{dimension}D_{instance}I_{p_class}C.npz"
    )


class CMAES_ST:
    def __init__(self, x_start, sigma):
        N = len(x_start)
        self.params = CMAESParameters(N)
        self.max_f_evals = 1e3 * N**2

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

    def ask(self):
        self._update_Eigensystem()
        return np.random.multivariate_normal(
            self.x_mean, (self.sigma**2) * self.C, self.params.lam
        )

    def tell(self, arx, fit_vals):
        self.count_eval += len(fit_vals)
        N = len(self.x_mean)
        x_old = self.x_mean

        arx = arx[np.argsort(fit_vals)]
        self.fit_vals = np.sort(fit_vals)
        self.fit_vals = np.clip(self.fit_vals, -np.power(10, 28), np.power(10, 28))

        self.x_mean = np.sum(
            arx[0 : self.params.mu] * self.params.weights[: self.params.mu, None],
            axis=0,
        )

        # Update evolution paths
        self.ps = (1 - self.params.cs) * self.ps + np.sqrt(
            self.params.cs * (2 - self.params.cs) * self.params.mueff
        ) * np.dot(self.inv_sqrt_C, (self.x_mean - x_old)) / self.sigma
        h_sig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.params.cs) ** (2 * self.count_eval / self.params.lam)
        ) / self.params.chiN < 1.4 + 2 / (N + 1)
        self.pc = (1 - self.params.cc) * self.pc + h_sig * np.sqrt(
            self.params.cc * (2 - self.params.cc) * self.params.mueff
        ) * (self.x_mean - x_old) / self.sigma

        # Adapt covariance matrix C
        ar_temp = (arx[0 : self.params.mu] - x_old) / self.sigma
        self.C = (
            (1 - self.params.c1 - self.params.cmu) * self.C
            + self.params.c1
            * (
                np.outer(self.pc, self.pc)
                + (1 - h_sig) * self.params.cc * (2 - self.params.cc) * self.C
            )
            + self.params.cmu * ar_temp.T.dot(np.diag(self.params.weights)).dot(ar_temp)
        )

        # Adapt step-size sigma
        self.sigma = self.sigma * np.exp(
            (self.params.cs / self.params.damps)
            * (np.linalg.norm(self.ps) / self.params.chiN - 1)
        )

        return self.sigma, self.ps, self.pc

    def stop(self):
        return (self.count_eval > 0) and (
            self.count_eval >= self.max_f_evals
            or self.condition_number > 1e14
            or len(self.fit_vals) > 1
            and self.fit_vals[-1] - self.fit_vals[0] < 1e-12
            or self.sigma * np.sqrt(max(self.D)) < 1e-11
        )

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
