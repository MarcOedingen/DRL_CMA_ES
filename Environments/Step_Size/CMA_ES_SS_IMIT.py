import os
import pickle
import g_utils
import numpy as np
from Environments import ss_utils
from stable_baselines3 import PPO
from imitation.algorithms import bc
from stable_baselines3.ppo import MlpPolicy
from imitation.data.types import Transitions
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS
from Environments.Step_Size.CMA_ES_SS import collect_expert_samples


def create_Transitions(data):
    condition = np.all(
        data["observations"] == np.concatenate([np.array([0.5]), np.zeros(81)]), axis=1
    )
    indices = np.where(condition)[0]

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


def run(dimension, x_start, sigma, instance):
    print(
        "---------------Running imitation learning for step-size adaptation---------------"
    )
    func_dimensions = np.repeat(dimension, 24) if dimension > 1 else np.random.randint(2, 40, 24)
    func_instances = np.repeat(instance, 24) if instance > 0 else np.random.randint(1, int(1e3) + 1, 24)

    train_funcs, test_funcs = ss_utils.split_train_test_functions(
        dimensions=func_dimensions, instances=func_instances
    )

    """x_start = (
        np.zeros(dimension)
        if x_start == 0
        else np.random.uniform(low=-5, high=5, size=dimension)
    )"""

    train_env = CMA_ES_SS(objetive_funcs=train_funcs, x_start=x_start, sigma=sigma)

    print("Collecting expert samples...")
    expert_samples = collect_expert_samples(
        dimension=dimension, instance=instance, x_start=x_start, sigma=sigma, bbob_functions=train_funcs
    )
    transitions = create_Transitions(expert_samples)

    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(42),
    )

    print("Training the agent with expert samples...")
    bc_trainer.train(n_epochs=1)

    print("Continue training the agent with PPO...")
    ppo_model = PPO(MlpPolicy, train_env, verbose=0)
    if os.path.exists(f"Environments/Step_Size/Policies/ppo_policy_ss_imit_{dimension}D_{instance}I.pkl"):
        ppo_model.policy = pickle.load(open(f"Environments/Step_Size/Policies/ppo_policy_ss_imit_{dimension}D_{instance}I.pkl", "rb"))
    else:
        ppo_model.policy = bc_trainer.policy
        ppo_model.learn(
            total_timesteps=int(1e6), callback=ss_utils.StopOnAllFunctionsEvaluated()
        )
        pickle.dump(ppo_model.policy, open(f"Environments/Step_Size/Policies/ppo_policy_ss_imit_{dimension}D_{instance}I.pkl", "wb"))

    print("Evaluating the agent on the test functions...")
    print(f"Test function ids: {sorted(list(set([test_func.id for test_func in test_funcs])))}")
    diffs = ss_utils.evaluate_agent(test_funcs, x_start, sigma, ppo_model)
    g_utils.print_pretty_table(func_dimensions=func_dimensions, func_instances=func_instances, results=diffs)
    print(f"Mean Difference: {np.mean(diffs)} +/- {np.std(diffs)}")
