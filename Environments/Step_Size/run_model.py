import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS


def run(dimension, x_start, sigma, instance, policy):
    print(f"---------------Running {policy}---------------")
    func_ids = np.arange(1, 25, dtype=int)
    func_dimensions = np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    func_instances = np.repeat(instance, 24) if instance > 0 else np.random.randint(1, int(1e3) + 1, 24)

    functions = [
        BenchmarkFunction("bbob", func_id, func_dimension, func_instance)
        for func_id, func_dimension, func_instance in zip(func_ids, func_dimensions, func_instances)
    ]

    ppo_model = PPO("MlpPolicy", CMA_ES_SS(objetive_funcs=functions, x_start=x_start, sigma=sigma), verbose=0)
    with open(f"Environments/Step_Size/Policies/{policy}_{dimension}D_{instance}I.pkl", "rb") as f:
        ppo_model.policy = pickle.load(f)

    rewards = np.zeros(len(functions))
    for index, test_func in enumerate(functions):
        eval_env = CMA_ES_SS(objetive_funcs=[test_func], x_start=x_start, sigma=0.5)
        obs, _ = eval_env.reset(verbose=0)
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            action, _states = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
        rewards[index] = -reward

    results = np.abs(rewards - np.array([f.best_value() for f in functions]))
    g_utils.print_pretty_table(func_dimensions=func_dimensions, func_instances=func_instances, results=results)
    print(f"Mean Difference: {np.mean(results)} +/- {np.std(results)}")
