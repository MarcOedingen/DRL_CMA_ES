import os
import g_utils
import numpy as np
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
from Environments.Combined.CMA_ES_ST_Env import CMA_ES_ST
from Environments.Combined.CMA_ES_ST import collect_expert_samples


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
        "---------------Running imitation learning for static parameters adaptation---------------"
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
        CMA_ES_ST(
            objective_funcs=train_funcs,
            x_start=x_start,
            sigma=sigma,
            reward_type=reward_type,
        ),
        max_episode_steps=max_eps_steps,
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
        rng=np.random.default_rng(seed=42),
    )

    policy_path = "Environments/Combined/Policies/policy_st_imit"
    p_class = p_class if split == "classes" else -1
    if not os.path.exists(f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl"):
        print("Pre-training policy with expert samples...")
        bc_trainer.train(n_epochs=10)

    ppo_model = g_utils.train_load_model_imit(
        policy_path=policy_path,
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
        reward_type=reward_type,
        ppo_model=ppo_model,
        env_name="static",
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"ppo_policy_st_imit{dimension}D_{instance}I_{p_class}C"
    )