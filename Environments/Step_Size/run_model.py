import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.ss_utils import evaluate_agent
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
        BenchmarkFunction("bbob", func_id, func_dimension, func_instance)
        for func_id, func_dimension, func_instance in zip(
            func_ids, func_dimensions, func_instances
        )
    ]

    ppo_model = PPO(
        "MlpPolicy",
        CMA_ES_SS(objetive_funcs=functions, x_start=x_start, sigma=sigma),
        verbose=0,
    )
    with open(
        f"Environments/Step_Size/Policies/{policy}_{dimension}D_{instance}I.pkl", "rb"
    ) as f:
        ppo_model.policy = pickle.load(f)

    diffs = evaluate_agent(test_funcs=functions, x_start=x_start, sigma=sigma, ppo_model=ppo_model)
    g_utils.print_pretty_table(
        func_dimensions=func_dimensions, func_instances=func_instances, results=diffs
    )
    print(f"Mean Difference: {np.mean(diffs)} +/- {np.std(diffs)}")