import os
import pickle
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO
from prettytable import PrettyTable
from gymnasium.wrappers import TimeLimit
from imitation.data.types import Transitions
from cocoex.function import BenchmarkFunction
from sklearn.model_selection import train_test_split
from Environments.h_Sigma.CMA_ES_HS_Env import CMA_ES_HS
from Environments.Damping.CMA_ES_DP_Env import CMA_ES_DP
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS
from Environments.Decay_Rate.CMA_ES_CS_Env import CMA_ES_CS
from Environments.Decay_Rate.CMA_ES_CC_Env import CMA_ES_CC
from Environments.Mu_Effective.CMA_ES_ME_Env import CMA_ES_ME
from Environments.Learning_Rate.CMA_ES_C1_Env import CMA_ES_C1
from Environments.Learning_Rate.CMA_ES_CM_Env import CMA_ES_CM
from Environments.Evolution_Path.CMA_ES_PS_Env import CMA_ES_PS

from stable_baselines3.common.callbacks import BaseCallback


class StopOnAllFunctionsEvaluated(BaseCallback):
    def __init__(self, verbose=0):
        super(StopOnAllFunctionsEvaluated, self).__init__(verbose)
        self.stop = False

    def _on_step(self) -> bool:
        if self.model.env.envs[0].get_wrapper_attr("stop"):
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


def get_func_dim_inst(dimension, instance, dim_choices, inst_choices, n_funcs=24):
    func_dimensions = (
        np.repeat(dimension, n_funcs) if dimension > 1 else np.random.choice(dim_choices, n_funcs)
    )
    func_instances = (
        np.repeat(instance, n_funcs) if instance > 0 else np.random.choice(inst_choices, n_funcs)
    )
    return func_dimensions, func_instances


def get_class_func_ids(_class):
    class_range = {
        1: (1, 5),
        2: (6, 9),
        3: (10, 14),
        4: (15, 19),
        5: (20, 24),
    }
    return np.arange(class_range[_class][0], class_range[_class][1] + 1)


def split_train_test(
        dimension, instance, split, p_class, n_functions=24, test_size=0.25, train_repeats=10, test_repeats=10,
        random_state=42
):
    ids = get_class_func_ids(p_class) if split == "classes" else np.arange(1, n_functions + 1)
    test_size = 1 / len(ids) if split == "classes" else test_size
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=random_state)
    return generate_splits(dimension, instance, train_ids, test_ids, train_repeats, test_repeats)


def generate_splits(dimension, instance, train_ids, test_ids, train_repeats, test_repeats):
    dim_choices = [2, 3, 5, 10, 20, 40]
    inst_choices = [i for i in range(1, 11)]

    train_dimensions = _choose_or_repeat(dimension, dim_choices, len(train_ids) * train_repeats)
    test_dimensions = _choose_or_repeat(dimension, dim_choices, len(test_ids) * test_repeats)
    train_instances = _choose_or_repeat(instance, inst_choices, len(train_ids) * train_repeats)
    test_instances = _choose_or_repeat(instance, inst_choices, len(test_ids) * test_repeats)

    train_ids = np.repeat(train_ids, repeats=train_repeats)
    test_ids = np.repeat(test_ids, repeats=test_repeats)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    train_funcs = create_benchmark_functions(train_ids, train_dimensions, train_instances)
    test_funcs = create_benchmark_functions(test_ids, test_dimensions, test_instances)

    return train_funcs, test_funcs


def _choose_or_repeat(choice, choices, size):
    return np.random.choice(choices, size=size) if choice < 1 else np.repeat(np.array([choice]), repeats=size)


def train_load_model(policy_path, dimension, instance, split, p_class, train_env, max_evals):
    ppo_model = PPO("MlpPolicy", train_env, verbose=0)
    p_class = p_class if split == "classes" else -1
    if not os.path.exists(f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl"):
        print("The policy does not exist. Training the policy...")
        ppo_model.learn(
            total_timesteps=max_evals,
            callback=StopOnAllFunctionsEvaluated(),
        )
        pickle.dump(
            ppo_model.policy,
            open(
                f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl",
                "wb",
            ),
        )
    else:
        print("The policy exists. Loading the policy...")
        ppo_model.policy = pickle.load(
            open(f"{policy_path}_{dimension}D_{instance}I_{p_class}C.pkl", "rb")
        )
    return ppo_model



def get_env(env_name, test_func, x_start, sigma):
    if env_name == "step_size":
        env = CMA_ES_SS(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "decay_rate_cs":
        env = CMA_ES_CS(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "decay_rate_cc":
        env = CMA_ES_CC(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "damping":
        env = CMA_ES_DP(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "learning_rate_c1":
        env = CMA_ES_C1(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "learning_rate_cm":
        env = CMA_ES_CM(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "mu_effective":
        env = CMA_ES_ME(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "h_sigma":
        env = CMA_ES_HS(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    elif env_name == "evolution_path_ps":
        env = CMA_ES_PS(objective_funcs=[test_func], x_start=x_start, sigma=sigma)
    else:
        raise NotImplementedError
    return TimeLimit(env, max_episode_steps=int(1e3 * 40 ** 2))


def evaluate_agent(test_funcs, x_start, sigma, ppo_model, env_name):
    print("Evaluating the agent on the test functions...")
    groups = {}
    for index, test_func in enumerate(test_funcs):
        key = test_func.id
        if key not in groups:
            groups[key] = []
        groups[key].append(index)

    results = []
    for key, indices in tqdm(groups.items()):
        grp_rewards = np.zeros(len(indices))
        reward_index = 0
        for index in indices:
            eval_env = get_env(
                env_name=env_name,
                test_func=test_funcs[index],
                x_start=x_start,
                sigma=sigma,
            )
            obs, _ = eval_env.reset(verbose=0)
            terminated, truncated = False, False
            while not (terminated or truncated):
                action, _states = ppo_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
            grp_rewards[reward_index] = np.exp(-reward)
            reward_index += 1
        results.append(
            {
                "id": key,
                "stats": np.array(
                    [
                        np.mean(grp_rewards),
                        np.std(grp_rewards),
                        np.max(grp_rewards),
                        np.min(grp_rewards),
                    ]
                ),
            }
        )
    return results


def print_pretty_table_simp(func_dimensions, func_instances, func_ids, results):
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


def print_pretty_table(results):
    table = PrettyTable()
    table.field_names = [
        "Function",
        "Mean Difference |f(x_best) - f(x_opt)|",
        "Max Difference |f(x_best) - f(x_opt)|",
        "Min Difference |f(x_best) - f(x_opt)|",
    ]
    results = sorted(results, key=lambda k: k["id"])
    for i in range(len(results)):
        table.add_row(
            [
                results[i]["id"],
                f"{results[i]['stats'][0]:.18f} ± {results[i]['stats'][1]:.18f}",
                f"{results[i]['stats'][2]:.18f}",
                f"{results[i]['stats'][3]:.18f}",
            ]
        )
    print(table)


def create_Transitions(data, n_train_funcs):
    shifted_dones = np.roll(np.where(data["dones"])[0], 1)
    shifted_dones[0] = 0
    indices = shifted_dones + np.concatenate(([0], np.arange(2, n_train_funcs + 1)))

    if len(indices) > 0:
        indices = np.concatenate((indices[1:], [len(data["observations"])]))
    else:
        indices = np.array([len(data["observations"])])

    starts = np.zeros(len(indices), dtype=int)
    ends = np.copy(indices) - 1

    starts[1:] = indices[:-1]
    full_range = np.arange(data["observations"].shape[0])

    valid_obs_mask = (full_range[:, None] >= starts) & (full_range[:, None] < ends)
    valid_next_obs_mask = (full_range[:, None] > starts) & (full_range[:, None] <= ends)

    obs_indices = full_range[np.any(valid_obs_mask, axis=1)]
    next_obs_indices = full_range[np.any(valid_next_obs_mask, axis=1)]

    obs = data["observations"][obs_indices]
    next_obs = data["observations"][next_obs_indices]

    return Transitions(
        obs=obs,
        next_obs=next_obs,
        acts=data["actions"],
        dones=data["dones"],
        infos=np.array([{}] * len(obs)),
    )


def save_results(results, policy):
    results = sorted(results, key=lambda k: k["id"])
    path = f"Results/{policy}.npz"
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        diffs = data["diff"].tolist()
        diffs.extend([r["stats"][0] for r in results])
        np.savez(
            path,
            diff=diffs,
        )
    else:
        diffs = [r["stats"][0] for r in results]
        np.savez(
            path,
            diff=diffs,
        )
