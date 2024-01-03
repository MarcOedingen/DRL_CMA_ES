import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from Environments.Decay_Rate.CMA_ES_CS_Env import CMA_ES_CS


def run(
    dimension, x_start, sigma, instance, max_eps_steps, train_repeats, test_repeats
):
    print(
        "---------------Running learning for decay-rate (cs) adaptation---------------"
    )
    func_dimensions = (
        np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    )
    func_instances = (
        np.repeat(instance, 24)
        if instance > 0
        else np.random.randint(1, int(1e3) + 1, 24)
    )

    train_funcs, test_funcs = g_utils.split_train_test_functions(
        dimensions=func_dimensions,
        instances=func_instances,
        train_repeats=train_repeats,
        test_repeats=test_repeats,
    )

    train_env = TimeLimit(
        CMA_ES_CS(objective_funcs=train_funcs, x_start=x_start, sigma=sigma),
        max_episode_steps=int(max_eps_steps),
    )
    ppo_model = PPO("MlpPolicy", train_env, verbose=0)
    if os.path.exists(
        f"Environments/Decay_Rate/Policies/ppo_policy_cs_{dimension}D_{instance}I.pkl"
    ):
        ppo_model.policy = pickle.load(
            open(
                f"Environments/Decay_Rate/Policies/ppo_policy_cs_{dimension}D_{instance}I.pkl",
                "rb",
            )
        )
    else:
        ppo_model.learn(
            total_timesteps=int(max_eps_steps * len(train_funcs) * train_repeats),
            callback=g_utils.StopOnAllFunctionsEvaluated(),
        )
        pickle.dump(
            ppo_model.policy,
            open(
                f"Environments/Decay_Rate/Policies/ppo_policy_cs_{dimension}D_{instance}I.pkl",
                "wb",
            ),
        )

    print("Evaluating the agent on the test functions...")
    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="decay_rate",
    )
    g_utils.print_pretty_table(
        results=results,
    )
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
