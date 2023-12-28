import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms import bc
from stable_baselines3.ppo import MlpPolicy
from imitation.data.types import Transitions
from cocoex.function import BenchmarkFunction
from sklearn.model_selection import train_test_split
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS
from stable_baselines3.common.callbacks import BaseCallback
from Environments.Step_Size.CMA_ES_SS_Samples import collect_expert_samples


def create_Transitions(data):
    # Find indices of the initial state
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


class StopOnAllFunctionsEvaluated(BaseCallback):
    """
    Callback for stopping the training when all functions are evaluated.
    """

    def __init__(self, verbose=0):
        super(StopOnAllFunctionsEvaluated, self).__init__(verbose)
        self._stop = False

    def _on_step(self) -> bool:
        """
        This method will be called by the model.
        :return: True if the callback should continue to be called, False if the training should stop.
        """
        if self.model.env.envs[0].env._stop:
            return False
        return True


def run(dimension, x_start, sigma, instance):
    print("Starting imitation learning for step-size adaptation")
    print("Splitting the functions into train and test sets...")
    train_ids, test_ids = train_test_split(
        np.arange(1, 25), test_size=0.2, random_state=42
    )
    train_funcs = np.repeat(
        [
            BenchmarkFunction("bbob", int(train_id), dimension, instance)
            for train_id in train_ids
        ],
        100,
    )
    np.random.shuffle(train_funcs)
    test_funcs = [
        BenchmarkFunction("bbob", int(test_id), dimension, instance)
        for test_id in test_ids
    ]
    test_funcs = np.repeat(test_funcs, 10)

    x_start = (
        np.zeros(dimension)
        if x_start == "zero"
        else np.random.uniform(low=-5, high=5, size=dimension)
    )
    train_env = CMA_ES_SS(objetive_funcs=train_funcs, x_start=x_start, sigma=sigma)
    print("Collecting expert samples...")
    expert_samples = collect_expert_samples(
        dimension=dimension, x_start=x_start, sigma=sigma, bbob_functions=train_funcs
    )
    transitions = create_Transitions(expert_samples)

    bc_trainer = bc.BC(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        demonstrations=transitions,
        rng=np.random.default_rng(42),
    )

    print("Training the agent with expert samples...")
    bc_trainer.train(n_epochs=10)

    print("Continue training the agent with PPO...")
    ppo_model = PPO(MlpPolicy, train_env, verbose=0)
    ppo_model.policy = bc_trainer.policy
    ppo_model.learn(total_timesteps=int(1e6), callback=StopOnAllFunctionsEvaluated())

    print("Evaluating the agent on the test functions...")
    rewards = np.zeros(len(test_funcs))
    for index, test_func in enumerate(test_funcs):
        eval_env = CMA_ES_SS(objetive_funcs=[test_func], x_start=x_start, sigma=sigma)
        obs, _ = eval_env.reset(verbose=0)
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            action, _states = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
        rewards[index] = -reward

    for i in range(len(test_funcs)):
        print(
            "Function: ",
            test_funcs[i].function,
            " Reward: ",
            rewards[i],
            "Optimum: ",
            test_funcs[i].best_value(),
        )
