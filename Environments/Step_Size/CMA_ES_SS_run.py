import pickle
from Environments import utils
from stable_baselines3 import PPO
from Environments.Step_Size.CMA_ES_SS_Env import CMA_ES_SS


def run(dimension, x_start, sigma, instance):
    print("---------------Starting learning for step-size adaptation---------------")
    train_funcs, test_funcs = utils.split_train_test_functions(
        dimension=dimension, instance=instance
    )

    train_env = CMA_ES_SS(objetive_funcs=train_funcs, x_start=x_start, sigma=sigma)
    ppo_model = PPO("MlpPolicy", train_env, verbose=0)
    ppo_model.learn(
        total_timesteps=int(1e6), callback=utils.StopOnAllFunctionsEvaluated()
    )
    pickle.dump(ppo_model.policy, open(f"Environments/Step_Size/Policies/ppo_policy_ss_{dimension}D.pkl", 'wb'))

    print("Evaluating the agent on the test functions...")
    utils.evaluate_agent(test_funcs, x_start, sigma, ppo_model)
