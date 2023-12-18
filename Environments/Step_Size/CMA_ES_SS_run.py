from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS

def main():
    dim = 2
    func_1 = BenchmarkFunction("bbob", 1, dim, 1)
    func_2 = BenchmarkFunction("bbob", 2, dim, 1)

    env = CMA_ES_SS([func_1, func_2], 0.5)

    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=int(1e5))

main()