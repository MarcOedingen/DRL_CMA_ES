import cma
import numpy as np
from cocoex.function import BenchmarkFunction


def run(dimension, x_start, sigma, instance):
    print("---------------Starting CMA-ES Optimized---------------")
    results_CMA_ES = []
    x_start = (
        np.zeros(dimension)
        if x_start == "zero"
        else np.random.uniform(low=-5, high=5, size=dimension)
    )
    for i in range(1, 25):
        function = BenchmarkFunction("bbob", i, dimension, instance)
        es = cma.CMAEvolutionStrategy(x_start, sigma, {"verbose": -9})
        es.optimize(function)
        results_CMA_ES.append(abs(function(es.best.x) - function.best_value()))
        print(
            f"Difference between actual minimum and found minimum: {abs(function(es.best.x) - function.best_value())}"
        )
    print(f"Mean: {np.mean(results_CMA_ES)} +/- {np.std(results_CMA_ES)}")
