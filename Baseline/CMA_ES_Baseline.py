import time
import numpy as np
from cocoex.function import BenchmarkFunction

def fmin(objective_fct, xstart, sigma):
    es = CMAES(xstart, sigma)
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        es.tell(X, fit)
    return es.xmean, es


class CMAESParameters(object):
    def __init__(self, N):
        self.dimension = N
        self.chiN = N**0.5 * (1 - 1.0 / (4 * N) + 1.0 / (21 * N**2))

        # Strategy parameter setting: Selection
        self.lam = 4 + int(3 * np.log(N))
        self.mu = int(self.lam / 2)
        _weights = np.log(self.lam / 2 + 1) - np.log(np.arange(1, self.mu + 1))
        w_sum = np.sum(_weights[: self.mu])
        self.weights = _weights / w_sum
        self.mueff = np.power(np.sum(self.weights[: self.mu]), 2) / np.sum(np.power(self.weights[: self.mu], 2))

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / (N + 2) ** 2)
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu) ** -1 / N**2


class CMAES:
    def __init__(self, xstart, sigma):
        N = len(xstart)
        self.params = CMAESParameters(N)
        self.maxfevals = 1e3 * N**2

        # initializing dynamic state variables
        self.xmean = xstart
        self.sigma = sigma
        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.B = np.eye(N)
        self.D = np.ones(N)
        self.C = np.eye(N)
        self.inv_sqrt_C = np.eye(N)
        self.condition_number = 1
        self.updated_eval = 0
        self.counteval = 0
        self.fitvals = np.zeros(N)

    def ask(self):
        if self.counteval > self.updated_eval + self.params.lazy_gap_evals:
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            self.D, self.B = np.linalg.eigh(self.C)
            self.condition_number = max(self.D) / min(self.D)
            self.inv_sqrt_C = np.dot(
                np.dot(self.B, np.diag(1.0 / np.sqrt(self.D))), self.B.T
            )
            self.updated_eval = self.counteval
        return self.xmean + np.dot(
            self.sigma
            * np.sqrt(self.D)
            * np.random.randn(self.params.lam, len(self.D)),
            self.B.T,
        )

    def tell(self, arx, fitvals):
        self.counteval += len(fitvals)
        N = len(self.xmean)
        par = self.params
        xold = self.xmean

        arx = arx[np.argsort(fitvals)]
        self.fitvals = np.sort(fitvals)

        self.xmean = np.sum(arx[0 : par.mu] * par.weights[: par.mu, None], axis=0)

        # Update evolution paths
        self.ps = (1 - par.cs) * self.ps + np.sqrt(
            par.cs * (2 - par.cs) * par.mueff
        ) * np.dot(self.inv_sqrt_C, (self.xmean - xold)) / self.sigma
        hsig = np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - par.cs) ** (2 * self.counteval / par.lam)
        ) / par.chiN < 1.4 + 2 / (N + 1)
        self.pc = (1 - par.cc) * self.pc + hsig * np.sqrt(
            par.cc * (2 - par.cc) * par.mueff
        ) * (self.xmean - xold) / self.sigma

        # Adapt covariance matrix C
        artemp = (arx[0 : par.mu] - xold) / self.sigma
        self.C = (
            (1 - par.c1 - par.cmu) * self.C
            + par.c1
            * (np.outer(self.pc, self.pc) + (1 - hsig) * par.cc * (2 - par.cc) * self.C)
            + par.cmu * artemp.T.dot(np.diag(par.weights)).dot(artemp)
        )

        # Adapt step-size sigma
        self.sigma = self.sigma * np.exp(
            (par.cs / par.damps) * (np.linalg.norm(self.ps) / par.chiN - 1)
        )

    def stop(self):
        res = {}
        if self.counteval <= 0:
            return res
        if self.counteval >= self.maxfevals:
            res["maxfevals"] = self.maxfevals
        if self.condition_number > 1e14:
            res["condition"] = self.condition_number
        if len(self.fitvals) > 1 and self.fitvals[-1] - self.fitvals[0] < 1e-12:
            res["tolfun"] = 1e-12
        if self.sigma * np.sqrt(max(self.D)) < 1e-11:
            res["tolx"] = 1e-11
        return res


if __name__ == "__main__":
    dim = 10
    resutls_cma_es = []
    start = time.perf_counter()
    for i in range(1, 25):
        function = BenchmarkFunction("bbob", i, dim, 1)
        x_min = fmin(function, dim * [0], 0.5)[0]
        print(
            f"Difference between actual minimum and found minimum: {abs(function(x_min) - function.best_value())}"
        )
        resutls_cma_es.append(abs(function(x_min) - function.best_value()))
    end = time.perf_counter() - start
    print(f"Time: {end}")
    print(sum(resutls_cma_es) / len(resutls_cma_es))
