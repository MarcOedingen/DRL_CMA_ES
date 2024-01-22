import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from Environments.h_Sigma.CMA_ES_HS_Env import CMA_ES_HS


def run(
    dimension,
    x_start,
    reward_type,
    sigma,
    instance,
    max_eps_steps,
    train_repeats,
    test_repeats,
    split,
    p_class,
    seed,
):
    print(
        "---------------Running learning for Heaviside function (h_sigma) adaptation---------------"
    )
    train_funcs, test_funcs = g_utils.split_train_test(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_repeats=train_repeats,
        test_repeats=test_repeats,
        random_state=seed,
    )

    train_env = TimeLimit(
        CMA_ES_HS(objective_funcs=train_funcs, x_start=x_start, sigma=sigma, reward_type=reward_type),
        max_episode_steps=int(max_eps_steps),
    )

    ppo_model = g_utils.train_load_model(
        policy_path="Environments/h_Sigma/Policies/ppo_policy_hs",
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_env=train_env,
        max_evals=int(max_eps_steps * len(train_funcs) * train_repeats),
    )

    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="h_sigma",
        reward_type=reward_type
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"ppo_policy_hs_{dimension}D_{instance}I_{p_class}C"
    )
