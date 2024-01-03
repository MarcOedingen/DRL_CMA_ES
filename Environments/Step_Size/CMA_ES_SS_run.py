import os
import pickle
import g_utils
import numpy as np
from Environments import ss_utils
from stable_baselines3 import PPO
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS


def run(dimension, x_start, sigma, instance):
    print("---------------Running learning for step-size adaptation---------------")
    func_dimensions = (
        np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    )
    func_instances = (
        np.repeat(instance, 24)
        if instance > 0
        else np.random.randint(1, int(1e3) + 1, 24)
    )

    train_funcs, test_funcs = ss_utils.split_train_test_functions(
        dimensions=func_dimensions, instances=func_instances
    )

    train_env = CMA_ES_SS(objective_funcs=train_funcs, x_start=x_start, sigma=sigma)
    ppo_model = PPO("MlpPolicy", train_env, verbose=0)
    if os.path.exists(
        f"Environments/Step_Size/Policies/ppo_policy_ss_{dimension}D_{instance}I.pkl"
    ):
        ppo_model.policy = pickle.load(
            open(
                f"Environments/Step_Size/Policies/ppo_policy_ss_{dimension}D_{instance}I.pkl",
                "rb",
            )
        )
    else:
        ppo_model.learn(
            total_timesteps=int(1e6), callback=ss_utils.StopOnAllFunctionsEvaluated()
        )
        pickle.dump(
            ppo_model.policy,
            open(
                f"Environments/Step_Size/Policies/ppo_policy_ss_{dimension}D_{instance}I.pkl",
                "wb",
            ),
        )

    print("Evaluating the agent on the test functions...")
    print(
        f"Test function ids: {sorted(list(set([test_func.id for test_func in test_funcs])))}"
    )
    diffs = ss_utils.evaluate_agent(test_funcs, x_start, sigma, ppo_model)
    g_utils.print_pretty_table(
        func_dimensions=func_dimensions, func_instances=func_instances, results=diffs
    )
    print(f"Mean Difference: {np.mean(diffs)} +/- {np.std(diffs)}")
