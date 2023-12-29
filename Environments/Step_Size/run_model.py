import pickle
import numpy as np
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS


def main():
    print("---------------Starting testing for step-size adaptation---------------")
    func_ids = np.arange(1, 25)
    func_instances = np.random.randint(1, 101, size=len(func_ids))
    functions = [
        BenchmarkFunction("bbob", func_id, 20, func_instance)
        for func_id, func_instance in zip(func_ids, func_instances)
    ]
    x_start = "random"
    sigma = 0.5

    ppo_model = PPO("MlpPolicy", CMA_ES_SS(objetive_funcs=functions, x_start=x_start, sigma=sigma), verbose=0)
    with open("Environments/Step_Size/ppo_policy_ss_imit_2D.pkl", "rb") as f:
        ppo_model.policy = pickle.load(f)

    rewards = np.zeros(len(functions))
    for index, test_func in enumerate(functions):
        eval_env = CMA_ES_SS(objetive_funcs=[test_func], x_start="zero", sigma=0.5)
        obs, _ = eval_env.reset(verbose=0)
        terminated, truncated = False, False
        steps = 0
        while not (terminated or truncated):
            action, _states = ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
        rewards[index] = -reward


    results = np.zeros((len(functions)))
    for i in range(len(functions)):
        print(
            "Function: ",
            functions[i].function,
            " Reward: ",
            rewards[i],
            "Optimum: ",
            functions[i].best_value(),
            "Difference: ",
            abs(functions[i].best_value() - rewards[i]),
        )
        results[i] = abs(functions[i].best_value() - rewards[i])

    print(f"Mean difference: {np.mean(results)} +/- {np.std(results)}")

main()