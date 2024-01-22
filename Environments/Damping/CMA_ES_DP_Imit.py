import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
from imitation.data.types import Transitions
from Environments.Damping.CMA_ES_DP_Env import CMA_ES_DP
from Environments.Damping.CMA_ES_DP import collect_expert_samples


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
        "---------------Running imitation learning for damping (dp) adaptation---------------"
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
        CMA_ES_DP(objective_funcs=train_funcs, x_start=x_start, sigma=sigma, reward_type=reward_type),
        max_episode_steps=int(max_eps_steps),
    )

    expert_samples = collect_expert_samples(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        x_start=x_start,
        sigma=sigma,
        bbob_functions=train_funcs,
    )

    transitions = g_utils.create_Transitions(
        data=expert_samples,
        n_train_funcs=len(train_funcs),
    )

    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(42),
    )
    policy_path = "Environments/Damping/Policies/ppo_policy_dp_imit"
    if not os.path.exists(f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl"):
        print("Pre-training policy with expert samples...")
        bc_trainer.train(n_epochs=10)

    ppo_model = g_utils.train_load_model_imit(
        policy_path=f"Environments/Damping/Policies/ppo_policy_dp_imit",
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_env=train_env,
        max_evals=int(max_eps_steps * len(train_funcs) * train_repeats),
        bc_policy=bc_trainer.policy,
    )

    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="damping",
        reward_type=reward_type
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results,
        policy=f"ppo_policy_dp_imit_{dimension}D_{instance}I_{p_class}C",
    )
