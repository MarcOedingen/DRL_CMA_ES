import pickle
import g_utils
import numpy as np
from torch import nn
from stable_baselines3 import PPO
from imitation.algorithms import bc
import stable_baselines3.common.vec_env
from gymnasium.wrappers import TimeLimit
from Environments.Evolution_Path.CMA_ES_PC_Env import CMA_ES_PC
from Environments.Evolution_Path.CMA_ES_PC import collect_expert_samples

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
        "---------------Running iterative imitation learning for evolution path (pc) adaption---------------"
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

    iterations = 2

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
            int(trans_split_indices[curr_index]): trans_split_indices[index_split]
            ]
        )
        curr_index = index_split

    np.random.shuffle(train_funcs)
    train_env = TimeLimit(
        CMA_ES_PC(
            objective_funcs=train_funcs,
            x_start=x_start,
            sigma=sigma,
            reward_type=reward_type,
        ),
        max_episode_steps=max_eps_steps,
    )
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512],
            vf=[512, 512],
        ),
        activation_fn=nn.Tanh,
    )
    ppo_model = PPO("MlpPolicy", train_env, ent_coef=1e-4, learning_rate=1e-5, policy_kwargs=policy_kwargs, verbose=0)
    ppo_model.policy = g_utils.custom_Actor_Critic_Policy(train_env)

    policy = None
    max_evals = len(train_funcs) * int(1e3) * dimension ** 2
    batch_size = 64
    ent_weight = 5e-2

    for i in range(iterations):
        print(f"Training policy in iteration {i + 1}...")

        bc_trainer = bc.BC(
            observation_space=train_env.observation_space,
            action_space=train_env.action_space,
            demonstrations=transition_splits[i],
            rng=np.random.default_rng(seed=42),
            policy=g_utils.custom_Actor_Critic_Policy(train_env)
            if policy is None
            else policy,
            batch_size=batch_size,
            ent_weight=ent_weight,
        )

        bc_trainer.train(n_epochs=int(np.ceil(5 / np.power(2, np.sqrt(i)))))
        bc_policy_parameters = {name: param.data for name, param in bc_trainer.policy.named_parameters()}
        ppo_model.policy.load_state_dict(bc_policy_parameters)

        ppo_model.learn(
            total_timesteps=max_evals,
            callback=g_utils.StopOnAllFunctionsEvaluated(),
        )
        policy = ppo_model.policy

        np.random.shuffle(train_funcs)
        train_env = TimeLimit(
            CMA_ES_PC(
                objective_funcs=train_funcs,
                x_start=x_start,
                sigma=sigma,
                reward_type=reward_type,
            ),
            max_episode_steps=max_eps_steps,
        )
        ppo_model.env = stable_baselines3.common.vec_env.DummyVecEnv([lambda: train_env])
        ppo_model.learning_rate = ppo_model.learning_rate * np.sqrt((iterations - 1) / (iterations * (i + 1)))
        ppo_model.ent_coef = ppo_model.ent_coef * np.sqrt((iterations - 1) / (iterations * (i + 1)))

    with open(
            f"Environments/Evolution_Path/Policies/ppo_policy_pc_imit_iter_{dimension}D_{instance}I_{p_class}C.pkl",
            "wb",
    ) as f:
        pickle.dump(policy, f)

    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        reward_type=reward_type,
        ppo_model=ppo_model,
        env_name="evolution_path_pc",
    )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results,
        policy=f"ppo_policy_pc_imit_iter_{dimension}D_{instance}I_{p_class}C",
    )