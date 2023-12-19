import gymnasium
from stable_baselines3 import PPO
from cocoex.function import BenchmarkFunction
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS

class CustomStopTrainingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class StopTrainingWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        if self.env._stop:
            raise CustomStopTrainingException("All functions evaluated")
        return self.env.reset(**kwargs)



def main():
    dim = 10
    funcs = [BenchmarkFunction("bbob", i, dim, 1) for i in range(1, 25)]

    env = CMA_ES_SS(objetive_funcs=funcs, sigma=0.5)
    env = StopTrainingWrapper(env)

    ppo_model = PPO("MlpPolicy", env, verbose=0)
    ppo_model.learn(total_timesteps=int(1e6))


main()
