import g_utils
import numpy as np
from tqdm import tqdm
from cocoex.function import BenchmarkFunction
from Parameters.CMA_ES_Parameters import CMAESParameters


def runCMAES(objective_fct, x_start, sigma):
    es = CMAES(x_start, sigma)
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        es.tell(X, fit)
    return es.x_mean, es


class CMAES:
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

        self.x_mean = np.sum(arx[0 : par.mu] * par.weights[: par.mu, None], axis=0)

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
        ar_temp = (arx[0 : par.mu] - x_old) / self.sigma
        self.C = (
            (1 - par.c1 - par.cmu) * self.C
            + par.c1
            * (
                np.outer(self.pc, self.pc)
                + (1 - h_sig) * par.cc * (2 - par.cc) * self.C
            )
            + par.cmu * ar_temp.T.dot(np.diag(par.weights)).dot(ar_temp)
        )

        # Adapt step-size sigma
        self.sigma = self.sigma * np.exp(
            (par.cs / par.damps) * (np.linalg.norm(self.ps) / par.chiN - 1)
        )

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


def run(dimension, x_start, sigma, instance):
    print("---------------Running CMA-ES baseline---------------")
    results_CMA_ES = []
    func_dimensions = (
        np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    )
    func_instances = (
        np.repeat(instance, 24)
        if instance > 0
        else np.random.randint(1, int(1e3) + 1, 24)
    )

    for i in tqdm(range(1, 25)):
        function = BenchmarkFunction(
            "bbob", i, func_dimensions[i - 1], func_instances[i - 1]
        )
        _x_start = (
            np.zeros(func_dimensions[i - 1])
            if x_start == 0
            else np.random.uniform(-5, 5, func_dimensions[i - 1])
        )
        x_min = runCMAES(objective_fct=function, x_start=_x_start, sigma=sigma)[0]
        results_CMA_ES.append(abs(function(x_min) - function.best_value()))

    g_utils.print_pretty_table(
        func_dimensions=func_dimensions,
        func_instances=func_instances,
        results=results_CMA_ES,
    )
    print(f"Mean Difference: {np.mean(results_CMA_ES)} +/- {np.std(results_CMA_ES)}")
