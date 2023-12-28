import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data.types import Transitions

env = gymnasium.make("CartPole-v1")
if not os.path.exists("ppo_cartpole.zip"):
    model = PPO(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("ppo_cartpole")
else:
    model = PPO.load("ppo_cartpole")

print("Evaluating expert")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

action_list, done_list, info_list, next_obs_list, obs_list = [], [], [], [], []

for episode in range(50):
    obs, _ = env.reset()
    terminated, truncated = False, False
    episode_obs = [obs]

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, _, terminated, truncated, info = env.step(action)

        episode_obs.append(next_obs)
        action_list.append(action)
        info_list.append(info)
        done_list.append(terminated or truncated)
        obs = next_obs

    obs_list.extend(episode_obs[:-1])
    next_obs_list.extend(episode_obs[1:])

transitions = Transitions(
    obs=np.array(obs_list),
    next_obs=np.array(next_obs_list),
    acts=np.array(action_list),
    dones=np.array(done_list),
    infos=np.array(info_list),
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=np.random.default_rng(42),
)

mean_reward, std_reward = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

bc_trainer.train(n_epochs=5)

mean_reward, std_reward = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=5)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

new_model = PPO(MlpPolicy, env, verbose=1)
new_model.policy = bc_trainer.policy
new_model.learn(total_timesteps=50_000)
