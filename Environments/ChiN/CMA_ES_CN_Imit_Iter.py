import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
from Environments.ChiN.CMA_ES_CN_Env import CMA_ES_CN
from Environments.ChiN.CMA_ES_CN import collect_expert_samples


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
        "---------------Running iterative imitation learning for ChiN adaptation---------------"
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

    iterations = 3

    expert_samples = collect_expert_samples(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        x_start=x_start,
        sigma=sigma,
        bbob_functions=np.repeat(train_funcs, iterations),
    )

    transitions = g_utils.create_Transitions(
        data=expert_samples,
        n_train_funcs=iterations * len(train_funcs),
    )

    trans_split_indices = np.where(transitions.dones)[0]
    transition_splits = []
    curr_index = 0
    for i in range(iterations):
        index_split = len(train_funcs) * (i + 1) - 1
        transition_splits.append(
            transitions[
                int(trans_split_indices[curr_index]) : trans_split_indices[index_split]
            ]
        )
        curr_index = index_split

    policy = None
    max_evals = len(train_funcs) * int(1e3) * dimension**2

    for i in range(iterations):
        print(f"Training policy in iteration {i + 1}...")

        np.random.shuffle(train_funcs)
        train_env = TimeLimit(
            CMA_ES_CN(
                objective_funcs=train_funcs,
                x_start=x_start,
                sigma=sigma,
                reward_type=reward_type,
            ),
            max_episode_steps=max_eps_steps,
        )

        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=transition_splits[i],
            rng=np.random.default_rng(seed=42),
            policy=g_utils.custom_Actor_Critic_Policy(train_env)
            if policy is None
            else policy,
        )

        bc_trainer.train(n_epochs=int(np.ceil(10 / np.power(2, np.sqrt(i)))))

        ppo_model = PPO("MlpPolicy", train_env, verbose=0)
        ppo_model.policy = bc_trainer.policy
        ppo_model.learn(
            total_timesteps=max_evals,
            callback=g_utils.StopOnAllFunctionsEvaluated(),
        )
        policy = ppo_model.policy

    # save the policy for the last iteration
    with open(
        f"Environments/ChiN/Policies/ppo_policy_cn_imit_iter_{dimension}D_{instance}I_{p_class}C.pkl",
        "wb",
    ) as f:
        pickle.dump(policy, f)

    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        reward_type=reward_type,
        ppo_model=ppo_model,
        env_name="chin",
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results,
        policy=f"ppo_policy_cn_imit_iter_{dimension}D_{instance}I_{p_class}C",
    )
