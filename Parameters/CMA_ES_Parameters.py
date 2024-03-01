import numpy as np


class CMAESParameters:
    def __init__(self, N, lam=None):
        self.dimension = N
        self.chiN = N**0.5 * (1 - 1.0 / (4 * N) + 1.0 / (21 * N**2))

        # Strategy parameter setting: Selection
        self.lam = 4 + int(3 * np.log(N)) if lam is None else lam
        self.mu = int(self.lam / 2)
        _weights = np.log(self.lam / 2 + 1) - np.log(np.arange(1, self.mu + 1))
        w_sum = np.sum(_weights[: self.mu])
        self.weights = _weights / w_sum
        self.mueff = np.power(np.sum(self.weights[: self.mu]), 2) / np.sum(
            np.power(self.weights[: self.mu], 2)
        )

        # Strategy parameter setting: Adaptation
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / (N + 2) ** 2
        )
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu) ** -1 / N**2

    def to_dict(self):
        return {
            "chiN": self.chiN,
            "mueff": self.mueff,
            "cc": self.cc,
            "cs": self.cs,
            "c1": self.c1,
            "cmu": self.cmu,
            "damps": self.damps,
            "lam": self.lam,
            "mu": self.mu,
            "lazy_gap_evals": self.lazy_gap_evals,
            "dimension": self.dimension,
            "weights": self.weights,
        }

    def set_params(self, params):
        if "chiN" in params.keys():
            self.chiN = params["chiN"]
        if "mueff" in params.keys():
            self.mueff = params["mueff"]
        if "cc" in params.keys():
            self.cc = params["cc"]
        if "cs" in params.keys():
            self.cs = params["cs"]
        if "c1" in params.keys():
            self.c1 = params["c1"]
        if "cmu" in params.keys():
            self.cmu = params["cmu"]
        if "damps" in params.keys():
            self.damps = params["damps"]
        if "lam" in params.keys():
            self.lam = params["lam"]
        if "mu" in params.keys():
            self.mu = params["mu"]
        if "lazy_gap_evals" in params.keys():
            self.lazy_gap_evals = params["lazy_gap_evals"]
        if "dimension" in params.keys():
            self.dimension = params["dimension"]
        if "weights" in params.keys():
            self.weights = params["weights"]