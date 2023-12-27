import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from stable_baselines3.ppo import MlpPolicy
from imitation.data.types import Transitions
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS
from stable_baselines3.common.evaluation import evaluate_policy


def create_Transitions(data):
    obs, next_obs = [], []
    # Get inidices where data['observations'] is the initial state, i.e. [0.5, 0, 0, ..., 0] (1 x 0.5 and 81 x 0)
    indices = np.where(np.all(data['observations'] == np.concatenate([np.array([0.5]), np.zeros(81)]), axis=1))[0]
    indices = np.concatenate((indices[1:], np.array([len(data['observations'])])))

    # Get the observations and next observations
    samples_added = 0
    for i, index in enumerate(indices):
        if i == 0:
            obs.extend(data['observations'][:index - 1])
            next_obs.extend(data['observations'][1:index])
            samples_added += index
        else:
            obs.extend(data['observations'][indices[i - 1]:index - 1])
            next_obs.extend(data['observations'][indices[i - 1]+1:index])
            samples_added += index - indices[i - 1] - 1

    return Transitions(
        obs=np.array(obs),
        next_obs=np.array(next_obs),
        acts=data['actions'],
        dones=data['dones'],
        infos=np.array([{}] * len(obs))
    )


# Load the expert samples from the npz file
dimension = 2
data = np.load(f"Environments/Step_Size/CMA_ES_SS_Samples_{dimension}D.npz")

# Create the environment
action_space = gymnasium.spaces.Box(low=1e-10, high=1, shape=(1,), dtype=np.float64)
observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(2 + 2 * 40,), dtype=np.float64)

# Prepare data for imitation learning, i.e. format them in a Transitions object
transitions = create_Transitions(data)

bc_trainer = bc.BC(
    observation_space=observation_space,
    action_space=action_space,
    demonstrations=transitions,
    rng = np.random.default_rng(42)
)

# Train the behavior cloning model
bc_trainer.train(n_epochs=10)



