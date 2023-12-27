import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from stable_baselines3.ppo import MlpPolicy
from imitation.data.types import Transitions
from stable_baselines3.common.evaluation import evaluate_policy


def create_Transitions(data):
    obs, next_obs = [], []
    indices = np.where(data['dones'])[0]

    for index in indices:
        obs.extend(data['observations'][:index - 1])
        next_obs.extend(data['observations'][1:index])

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

# Prepare data for imitation learning, i.e. format them in a Transitions object
transitions = create_Transitions(data)
