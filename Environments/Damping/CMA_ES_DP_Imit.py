import os
import g_utils
import numpy as np
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
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
    pre_train_repeats,
    split,
    p_class,
    seed,
):
    print(
        "---------------Running imitation learning for damping (dp) adaptation---------------"
    )
    pre_train_funcs, _ = g_utils.split_train_test(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_repeats=pre_train_repeats,
        test_repeats=test_repeats,
        random_state=seed,
    )

    pre_train_env = TimeLimit(
        CMA_ES_DP(
            objective_funcs=pre_train_funcs,
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
        bbob_functions=pre_train_funcs,
    )

    transitions = g_utils.create_Transitions(
        data=expert_samples,
        n_train_funcs=len(pre_train_funcs),
    )

    n_epochs = 10
    indices = np.where(expert_samples["dones"])[0]
    diffs = np.diff(indices)
    batch_size = int(np.median(diffs) / (pre_train_repeats * n_epochs))

    bc_trainer = bc.BC(
        observation_space=pre_train_env.observation_space,
        action_space=pre_train_env.action_space,
        demonstrations=transitions,
        policy=g_utils.custom_Actor_Critic_Policy(pre_train_env),
        rng=np.random.default_rng(seed),
        batch_size=batch_size,
    )

    policy_path = "Environments/Damping/Policies/policy_dp_imit"
    if not os.path.exists(f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl"):
        print("Pre-training policy with expert samples...")
        bc_trainer.train(n_epochs=n_epochs)

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
        CMA_ES_DP(
            objective_funcs=train_funcs,
            x_start=x_start,
            sigma=sigma,
            reward_type=reward_type,
        ),
        max_episode_steps=max_eps_steps,
    )

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
        env_name="damping",
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    g_utils.save_results(
        results=results,
        policy=f"ppo_policy_dp_imit_{dimension}D_{instance}I_{p_class}C",
    )
