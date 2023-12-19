import numpy as np


class CMAESParameters:
    def __init__(self, N):
        self.dimension = N
        self.chiN = N ** 0.5 * (1 - 1.0 / (4 * N) + 1.0 / (21 * N ** 2))

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
        self.lazy_gap_evals = 0.5 * N * self.lam * (self.c1 + self.cmu) ** -1 / N ** 2
