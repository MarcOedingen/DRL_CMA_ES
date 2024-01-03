import cma
import g_utils
import numpy as np
from tqdm import tqdm
from cocoex.function import BenchmarkFunction


def run(dimension, x_start, sigma, instance):
    print("---------------Running CMA-ES Optimized---------------")
    results_CMA_ES = []
    func_dimensions = (
        np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    )
    func_instances = (
        np.repeat(instance, 24)
        if instance > 0
        else np.random.randint(1, int(1e3) + 1, 24)
    )

    func_ids = []
    for i in tqdm(range(1, 25)):
        function = BenchmarkFunction(
            "bbob", i, func_dimensions[i - 1], func_instances[i - 1]
        )
        func_ids.append(function.id)
        _x_start = (
            np.zeros(func_dimensions[i - 1])
            if x_start == 0
            else np.random.uniform(-5, 5, func_dimensions[i - 1])
        )
        es = cma.CMAEvolutionStrategy(_x_start, sigma, {"verbose": -9})
        es.optimize(function)
        results_CMA_ES.append(abs(function(es.best.x) - function.best_value()))

    g_utils.print_pretty_table(
        func_dimensions=func_dimensions,
        func_instances=func_instances,
        func_ids=func_ids,
        results=results_CMA_ES,
    )
    print(f"Mean: {np.mean(results_CMA_ES)} +/- {np.std(results_CMA_ES)}")
