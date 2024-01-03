import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS


def run(dimension, x_start, sigma, instance, policy):
    print(f"---------------Running {policy}---------------")
    func_ids = np.arange(1, 25, dtype=int)
    func_dimensions = (
        np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    )
    func_instances = (
        np.repeat(instance, 24)
        if instance > 0
        else np.random.randint(1, int(1e3) + 1, 24)
    )

    functions = [
        BenchmarkFunction("bbob", int(func_id), int(func_dimension), int(func_instance))
        for func_id, func_dimension, func_instance in zip(
            func_ids, func_dimensions, func_instances
        )
    ]

    ppo_model = PPO(
        "MlpPolicy",
        CMA_ES_SS(objective_funcs=functions, x_start=x_start, sigma=sigma),
        verbose=0,
    )
    with open(
        f"Environments/Step_Size/Policies/{policy}_{dimension}D_{instance}I.pkl", "rb"
    ) as f:
        ppo_model.policy = pickle.load(f)

    function_ids = sorted(list(set([test_func.id for test_func in functions])))
    diffs = g_utils.evaluate_agent(test_funcs=functions, x_start=x_start, sigma=sigma, ppo_model=ppo_model, env_name="step_size", repeats=1)
    g_utils.print_pretty_table(
        func_dimensions=func_dimensions, func_instances=func_instances, func_ids=function_ids, results=diffs
    )
    print(f"Mean Difference: {np.mean(diffs)} +/- {np.std(diffs)}")
