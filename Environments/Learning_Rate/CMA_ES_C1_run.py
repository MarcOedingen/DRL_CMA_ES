import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from Environments.Learning_Rate.CMA_ES_C1_Env import CMA_ES_C1


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
        "---------------Running learning for learning-rate (c1) adaptation---------------"
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
        CMA_ES_C1(objective_funcs=train_funcs, x_start=x_start, sigma=sigma, reward_type=reward_type),
        max_episode_steps=int(max_eps_steps),
    )

    ppo_model = g_utils.train_load_model(
        policy_path="Environments/Learning_Rate/Policies/ppo_policy_c1",
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
        env_name="learning_rate_c1",
        reward_type=reward_type,
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"ppo_policy_c1_{dimension}D_{instance}I_{p_class}C"
    )
