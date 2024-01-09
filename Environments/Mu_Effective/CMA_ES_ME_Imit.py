import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
from Environments.Mu_Effective.CMA_ES_ME_Env import CMA_ES_ME
from Environments.Mu_Effective.CMA_ES_ME import collect_expert_samples


def run(
    dimension, x_start, sigma, instance, max_eps_steps, train_repeats, test_repeats, seed
):
    print(
        "---------------Running imitation learning for mu effective adaptation---------------"
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
        random_state=seed,
    )

    train_env = TimeLimit(
        CMA_ES_ME(objective_funcs=train_funcs, x_start=x_start, sigma=sigma),
        max_episode_steps=int(max_eps_steps),
    )

    print("Collecting expert samples...")
    expert_samples = collect_expert_samples(
        dimension=dimension,
        instance=instance,
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

    print("Training the agent with expert samples...")
    bc_trainer.train(n_epochs=5)

    print("Continue training the agent with PPO...")
    ppo_model = PPO("MlpPolicy", train_env, verbose=0)

    if os.path.exists(
        f"Environments/Mu_Effective/Policies/ppo_policy_me_imit_{dimension}D_{instance}I.pkl"
    ):
        print("Loading the pre-trained policy...")
        ppo_model.policy = pickle.load(
            open(
                f"Environments/Mu_Effective/Policies/ppo_policy_me_imit_{dimension}D_{instance}I.pkl",
                "rb",
            )
        )
    else:
        ppo_model.policy = bc_trainer.policy
        ppo_model.learn(
            total_timesteps=int(max_eps_steps * len(train_funcs) * train_repeats),
            callback=g_utils.StopOnAllFunctionsEvaluated(),
        )
        pickle.dump(
            ppo_model.policy,
            open(
                f"Environments/Mu_Effective/Policies/ppo_policy_me_imit_{dimension}D_{instance}I.pkl",
                "wb",
            ),
        )

    print("Evaluating the agent on the test functions...")
    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="mu_effective",
    )
    g_utils.print_pretty_table(
        results=results,
    )
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
