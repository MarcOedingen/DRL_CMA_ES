import numpy as np
from prettytable import PrettyTable
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


def create_benchmark_functions(ids, dimensions, instances):
    funcs = np.array(
        [
            BenchmarkFunction("bbob", int(id_), int(dim), int(inst))
            for id_, dim, inst in zip(ids, dimensions, instances)
        ]
    )
    return funcs

def split_train_test_functions(
    dimensions,
    instances,
    n_functions=24,
    test_size=0.25,
    train_repeats=10,
    test_repeats=10,
    random_state=42,
):
    train_ids, test_ids = train_test_split(
        np.arange(1, n_functions + 1), test_size=test_size, random_state=random_state
    )

    if not np.all(dimensions == dimensions[0]):
        train_dimensions = np.random.randint(2, 41, size=len(train_ids) * train_repeats)
        test_dimensions = np.random.randint(2, 41, size=len(test_ids) * test_repeats)
    else:
        train_dimensions = np.repeat(dimensions[:len(train_ids)], repeats=train_repeats)
        test_dimensions = np.repeat(dimensions[:len(test_ids)], repeats=test_repeats)

    if not np.all(instances == instances[0]):
        train_instances = np.random.randint(1, 1001, size=len(train_ids) * train_repeats)
        test_instances = np.random.randint(1, 1001, size=len(test_ids) * test_repeats)
    else:
        train_instances = np.repeat(instances[:len(train_ids)], repeats=train_repeats)
        test_instances = np.repeat(instances[:len(test_ids)], repeats=test_repeats)

    train_ids = np.repeat(train_ids, repeats=train_repeats)
    test_ids = np.repeat(test_ids, repeats=test_repeats)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    train_funcs = create_benchmark_functions(train_ids, train_dimensions, train_instances)
    test_funcs = create_benchmark_functions(test_ids, test_dimensions, test_instances)

    return train_funcs, test_funcs


def get_env(env_name, test_func, x_start, sigma):
    if env_name == "step_size":
        return CMA_ES_SS(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    else:
        raise NotImplementedError


def evaluate_agent(test_funcs, x_start, sigma, ppo_model, env_name, repeats=10):
    rewards = np.zeros(len(test_funcs))
    for index, test_func in enumerate(test_funcs):
        eval_env = get_env(
            env_name=env_name, test_func=test_func, x_start=x_start, sigma=sigma
        )
        obs, _ = eval_env.reset(verbose=0)
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _states = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards[index] = -reward

    diffs = np.abs(
        rewards - np.array([test_func.best_value() for test_func in test_funcs])
    )
    return np.mean(diffs.reshape(-1, repeats), axis=1)


def print_pretty_table(func_dimensions, func_instances, func_ids, results):
    table = PrettyTable()
    table.field_names = [
        "Function",
        "Dimensions",
        "Instance",
        "Difference |f(x_best) - f(x_opt)|",
    ]
    for i in range(len(results)):
        table.add_row(
            [
                func_ids[i],
                func_dimensions[i],
                func_instances[i],
                f"{results[i]:.18f}",
            ]
        )
    print(table)
