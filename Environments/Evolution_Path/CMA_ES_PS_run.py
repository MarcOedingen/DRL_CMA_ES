import g_utils
import numpy as np
from gymnasium.wrappers import TimeLimit
from Environments.Evolution_Path.CMA_ES_PS_Env import CMA_ES_PS


def run(
    dimension,
    x_start,
    reward_type,
    sigma,
    instance,
    max_eps_steps,
    train_repeats,
    test_repeats,
    pre_train_repeats,
    split,
    p_class,
    seed,
):
    print(
        "---------------Running learning for evolution path (ps) adaptation---------------"
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
        CMA_ES_PS(
            objective_funcs=train_funcs,
            x_start=x_start,
            sigma=sigma,
            reward_type=reward_type,
        ),
        max_episode_steps=int(max_eps_steps),
    )

    ppo_model = g_utils.train_load_model(
        policy_path="Environments/Evolution_Path/Policies/ppo_policy_ps",
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_env=train_env,
        max_evals=int(max_eps_steps * len(train_funcs) * train_repeats),
        policy=g_utils.custom_Actor_Critic_Policy(train_env),
    )

    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="evolution_path_ps",
        reward_type=reward_type,
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"ppo_policy_ps_{dimension}D_{instance}I_{p_class}C"
    )
