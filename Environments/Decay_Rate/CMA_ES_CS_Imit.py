import os
import pickle
import g_utils
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from gymnasium.wrappers import TimeLimit
from imitation.data.types import Transitions
from Environments.Decay_Rate.CMA_ES_CS_Env import CMA_ES_CS
from Environments.Decay_Rate.CMA_ES_CS import collect_expert_samples


def create_Transitions(data, n_train_funcs):
    shifted_dones = np.roll(np.where(data["dones"])[0], 1)
    shifted_dones[0] = 0
    indices = shifted_dones + np.concatenate(([0], np.arange(2, n_train_funcs + 1)))

    # Adjust the indices array
    if len(indices) > 0:
        indices = np.concatenate((indices[1:], [len(data["observations"])]))
    else:
        indices = np.array([len(data["observations"])])

    # Calculate start and end indices for each segment
    starts = np.zeros(len(indices), dtype=int)
    ends = np.copy(indices) - 1

    # For starts, shift indices array by one position
    starts[1:] = indices[:-1]

    # Create the full range of indices
    full_range = np.arange(data["observations"].shape[0])

    # Generate masks for valid indices
    valid_obs_mask = (full_range[:, None] >= starts) & (full_range[:, None] < ends)
    valid_next_obs_mask = (full_range[:, None] > starts) & (full_range[:, None] <= ends)

    # Extract valid indices for obs and next_obs
    obs_indices = full_range[np.any(valid_obs_mask, axis=1)]
    next_obs_indices = full_range[np.any(valid_next_obs_mask, axis=1)]

    # Use advanced indexing to get obs and next_obs
    obs = data["observations"][obs_indices]
    next_obs = data["observations"][next_obs_indices]

    return Transitions(
        obs=obs,
        next_obs=next_obs,
        acts=data["actions"],
        dones=data["dones"],
        infos=np.array([{}] * len(obs)),
    )


def run(
    dimension, x_start, sigma, instance, max_eps_steps, train_repeats, test_repeats
):
    print(
        "---------------Running imitation learning for decay-rate (cs) adaptation---------------"
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
        max_episode_steps=max_eps_steps,
    )

    print("Collecting expert samples...")
    expert_samples = collect_expert_samples(
        dimension=dimension,
        instance=instance,
        x_start=x_start,
        sigma=sigma,
        bbob_functions=train_funcs,
    )

    transitions = create_Transitions(
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
        f"Environments/Decay_Rate/Policies/ppo_policy_cs_imit_{dimension}D_{instance}I.pkl"
    ):
        print("Loading the pre-trained policy...")
        ppo_model.policy = pickle.load(
            open(
                f"Environments/Decay_Rate/Policies/ppo_policy_cs_imit_{dimension}D_{instance}I.pkl",
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
                f"Environments/Decay_Rate/Policies/ppo_policy_cs_imit_{dimension}D_{instance}I.pkl",
                "wb",
            ),
        )

    print("Evaluating the agent on the test functions...")
    results = g_utils.evaluate_agent(
        test_funcs=test_funcs,
        x_start=x_start,
        sigma=sigma,
        ppo_model=ppo_model,
        env_name="decay_rate_cs",
    )
    g_utils.print_pretty_table(
        results=results,
    )
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
