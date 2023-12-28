import numpy as np
from cocoex.function import BenchmarkFunction
from sklearn.model_selection import train_test_split
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS

from stable_baselines3.common.callbacks import BaseCallback


class StopOnAllFunctionsEvaluated(BaseCallback):
    def __init__(self, verbose=0):
        super(StopOnAllFunctionsEvaluated, self).__init__(verbose)
        self._stop = False

    def _on_step(self) -> bool:
        if self.model.env.envs[0].env._stop:
            return False
        return True


def split_train_test_functions(
    dimension,
    instance,
    n_functions=24,
    test_size=0.25,
    train_repeats=10,
    test_repeats=10,
    random_state=42,
):
    train_ids, test_ids = train_test_split(
        np.arange(1, n_functions + 1), test_size=test_size, random_state=random_state
    )
    train_funcs = np.repeat(
        [
            BenchmarkFunction("bbob", int(train_id), dimension, instance)
            for train_id in train_ids
        ],
        repeats=train_repeats,
    )
    np.random.shuffle(train_funcs)
    test_funcs = [
        BenchmarkFunction("bbob", int(test_id), dimension, instance)
        for test_id in test_ids
    ]
    test_funcs = np.repeat(test_funcs, repeats=test_repeats)
    return train_funcs, test_funcs


def evaluate_agent(test_funcs, x_start, sigma, ppo_model):
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

    results = np.zeros((len(test_funcs)))
    for i in range(len(test_funcs)):
        print(
            "Function: ",
            test_funcs[i].function,
            " Reward: ",
            rewards[i],
            "Optimum: ",
            test_funcs[i].best_value(),
            "Difference: ",
            abs(test_funcs[i].best_value() - rewards[i]),
        )
        results[i] = abs(test_funcs[i].best_value() - rewards[i])

    print(f"Mean difference: {np.mean(results)} +/- {np.std(results)}")
