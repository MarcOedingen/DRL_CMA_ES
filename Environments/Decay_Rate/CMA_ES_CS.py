import numpy as np
from Parameters.CMA_ES_Parameters import CMAESParameters


class CMAES_CS:
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
