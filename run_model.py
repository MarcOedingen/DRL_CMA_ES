import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Damping.CMA_ES_DP_Env import CMA_ES_DP
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS
from Environments.Decay_Rate.CMA_ES_CS_Env import CMA_ES_CS


def get_path(dimension, instance, policy):
    if "_ss" in policy:
        path = "Environments/Step_Size/Policies/"
    elif "_cs" in policy:
        path = "Environments/Decay_Rate/Policies/"
    elif "_dp" in policy:
        path = "Environments/Damping/Policies/"
    else:
        raise NotImplementedError
    return path + f"{policy}_{dimension}D_{instance}I.pkl"


def get_env(functions, x_start, sigma, policy):
    if "_ss" in policy:
        return "step_size", CMA_ES_SS(
            objective_funcs=functions, x_start=x_start, sigma=sigma
        )
    elif "_cs" in policy:
        return "decay_rate", CMA_ES_CS(
            objective_funcs=functions, x_start=x_start, sigma=sigma
        )
    elif "_dp" in policy:
        return "damping", CMA_ES_DP(
            objective_funcs=functions, x_start=x_start, sigma=sigma
        )
    else:
        raise NotImplementedError


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

    env_name, env = get_env(functions, x_start, sigma, policy)

    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
    )
    with open(
        get_path(dimension=dimension, instance=instance, policy=policy), "rb"
    ) as f:
        ppo_model.policy = pickle.load(f)

    results = g_utils.evaluate_agent(
        test_funcs=functions,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name=env_name,
    )
    g_utils.print_pretty_table(
        results=results,
    )
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
