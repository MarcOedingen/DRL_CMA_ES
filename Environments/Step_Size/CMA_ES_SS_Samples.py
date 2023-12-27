import numpy as np
from collections import deque
from cocoex.function import BenchmarkFunction
from Parameters.CMA_ES_Parameters import CMAESParameters


def runCMAES(objective_fct, x_start, sigma, h=40, f_limit=np.power(10, 28)):
    es = CMAES_SS_Samples(x_start, sigma)
    observations, actions, dones = [np.hstack((np.array(sigma), np.zeros(81)))], [], []
    hist_fit_vals = deque(np.zeros(h), maxlen=h)
    hist_sigmas = deque(np.zeros(h), maxlen=h)
    iteration = 0
    cur_sigma = sigma
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        ps, new_sigma = es.tell(X, fit)
        reward = np.clip(-np.mean(fit), -f_limit, f_limit)
        if iteration > 0:
            difference = np.clip(np.abs((reward - hist_fit_vals[len(hist_fit_vals) - 1])), -f_limit, f_limit) / reward
            hist_fit_vals.append(difference)
            hist_sigmas.append(cur_sigma)
        observations.append(np.concatenate(
            [np.array([new_sigma]), np.array([np.linalg.norm(ps) / es.params.chiN - 1]), np.array(hist_fit_vals),
             np.array(hist_sigmas)]))
        actions.append(new_sigma)
        dones.append(False)
        cur_sigma = new_sigma
        iteration += 1
    dones[-1] = True
    return np.array(observations), np.array(actions), np.array(dones)


class CMAES_SS_Samples:
    def __init__(self, x_start, sigma):
        N = len(x_start)
        self.params = CMAESParameters(N)
        self.max_f_evals = 1e3 * N ** 2

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
        par = self.params
        x_old = self.x_mean

        arx = arx[np.argsort(fit_vals)]
        self.fit_vals = np.sort(fit_vals)

        self.x_mean = np.sum(arx[0:par.mu] * par.weights[:par.mu, None], axis=0)

        # Update evolution paths
        self.ps = (1 - par.cs) * self.ps + np.sqrt(
            par.cs * (2 - par.cs) * par.mueff
        ) * np.dot(self.inv_sqrt_C, (self.x_mean - x_old)) / self.sigma
        h_sig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - par.cs) ** (2 * self.count_eval / par.lam)
        ) / par.chiN < 1.4 + 2 / (N + 1)
        self.pc = (1 - par.cc) * self.pc + h_sig * np.sqrt(
            par.cc * (2 - par.cc) * par.mueff
        ) * (self.x_mean - x_old) / self.sigma

        # Adapt covariance matrix C
        ar_temp = (arx[0: par.mu] - x_old) / self.sigma
        self.C = (
                (1 - par.c1 - par.cmu) * self.C
                + par.c1
                * (np.outer(self.pc, self.pc) + (1 - h_sig) * par.cc * (2 - par.cc) * self.C)
                + par.cmu * ar_temp.T.dot(np.diag(par.weights)).dot(ar_temp)
        )

        # Adapt step-size sigma
        self.sigma = self.sigma * np.exp(
            (par.cs / par.damps) * (np.linalg.norm(self.ps) / par.chiN - 1)
        )

        return self.ps, self.sigma

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


def run(dimensions, x_start, sigma, instance):
    observations, actions, dones = [], [], []
    for i in range(1, 25):
        function = BenchmarkFunction("bbob", i, dimensions, instance)
        obs, acts, dns = runCMAES(objective_fct=function, x_start=x_start, sigma=sigma)
        observations.extend(obs)
        actions.extend(acts)
        dones.extend(dns)
    # Cache the observations and actions as npz file for later use, i.e. imitation learning
    np.savez(f"Environments/Step_Size/CMA_ES_SS_Samples_{dimensions}D.npz", observations=observations, actions=actions, dones=dones)

    test = np.load(f"Environments/Step_Size/CMA_ES_SS_Samples_{dimensions}D.npz")
    print(test["observations"].shape)
    print(test["actions"].shape)


run(2, np.zeros(2), 0.5, 1)
