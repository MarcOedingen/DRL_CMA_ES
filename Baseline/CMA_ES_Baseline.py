import g_utils
import numpy as np
from tqdm import tqdm
from Parameters.CMA_ES_Parameters import CMAESParameters


def runCMAES(objective_fct, x_start, sigma):
    es = CMAES(x_start, sigma)
    while not es.stop():
        X = es.ask()
        fit = [objective_fct(x) for x in X]
        es.tell(X, fit)
    return es.fit_vals[0], es


class CMAES:
    def __init__(self, x_start, sigma, parameters={}):
        N = len(x_start)
        self.params = CMAESParameters(N)
        # If params is not empty set the parameters
        if parameters:
            self.params.set_params(params=parameters)
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
        return np.random.multivariate_normal(
            self.x_mean, (self.sigma**2) * self.C, self.params.lam
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


def run(dimension, x_start, sigma, instance, split, p_class, test_repeats, seed):
    print("---------------Running CMA-ES baseline---------------")
    p_class = p_class if split == "classes" else -1
    _, functions = g_utils.split_train_test(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        test_repeats=test_repeats,
        random_state=seed,
    )

    results = []
    groups = {}
    for index, test_func in enumerate(functions):
        key = test_func.id
        if key not in groups:
            groups[key] = []
        groups[key].append(index)

    for key, indices in tqdm(groups.items()):
        grp_rewards = np.zeros(len(indices))
        reward_index = 0
        for index in indices:
            test_func = functions[index]
            _x_start = (
                np.random.uniform(
                    low=-5,
                    high=5,
                    size=test_func.dimension,
                )
                if x_start == -1
                else np.zeros(test_func.dimension)
            )
            f_min = runCMAES(objective_fct=test_func, x_start=_x_start, sigma=sigma)[0]
            grp_rewards[reward_index] = np.abs(test_func.best_value() - f_min)
            reward_index += 1
        results.append(
            {
                "id": key,
                "stats": np.array(
                    [
                        np.mean(grp_rewards),
                        np.std(grp_rewards),
                        np.max(grp_rewards),
                        np.min(grp_rewards),
                    ]
                ),
            }
        )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"baseline_{dimension}D_{instance}I_{p_class}C"
    )
