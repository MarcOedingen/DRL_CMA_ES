import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Run different CMA-ES algorithms")
    parser.add_argument("--algorithm", type=str, help="The dataset to use", choices=[
        "baseline", "optimized", "step_size", "step_size_imit", "decay_rate_cs",
        "decay_rate_cs_imit", "decay_rate_cc", "decay_rate_cc_imit", "damping",
        "damping_imit", "testing"], default="decay_rate_cc_imit")
    parser.add_argument("--dimension", type=int, help="The dimension of the problem",
                        choices=[i for i in range(2, 41)], default=2)
    parser.add_argument("--x_start", type=int, help="The x-values starting from",
                        choices=[-1, 0], default=0)
    parser.add_argument("--sigma", type=float, help="The initial sigma", default=0.5)
    parser.add_argument("--instance", type=int, help="The instance of the problem",
                        choices=[i for i in range(-1, int(1e3) + 1) if i != 0], default=1)
    parser.add_argument("--policy", type=str, help="The model to use", choices=[
        "ppo_policy_ss", "ppo_policy_ss_imit", "ppo_policy_cs", "ppo_policy_cs_imit",
        "ppo_policy_cc", "ppo_policy_cc_imit", "ppo_policy_dp", "ppo_policy_dp_imit"],
        default="ppo_policy_ss")
    parser.add_argument("--max_episode_steps", type=int, help="The max episode steps",
                        default=int(1e3 * 40**2))
    parser.add_argument("--train_repeats", type=int, help="The number of repeats for the training functions",
                        default=10)
    parser.add_argument("--test_repeats", type=int, help="The number of repeats for the test functions",
                        default=10)

    args = parser.parse_args()

    if args.algorithm == "testing":
        module = importlib.import_module("run_model")
        run_function = getattr(module, "run")
        run_function(args.dimension, args.x_start, args.sigma, args.instance, args.policy)
    else:
        module_path, function_name = get_module_and_function(args.algorithm)
        module = importlib.import_module(module_path)
        run_function = getattr(module, function_name)
        run_function(args.dimension, args.x_start, args.sigma, args.instance,
                     args.max_episode_steps, args.train_repeats, args.test_repeats)

def get_module_and_function(algorithm):
    mapping = {
        "baseline": ("Baseline.CMA_ES_Baseline", "run"),
        "optimized": ("Optimized.CMA_ES_Optimized", "run"),
        "step_size": ("Environments.Step_Size.CMA_ES_SS_run", "run"),
        "step_size_imit": ("Environments.Step_Size.CMA_ES_SS_imit_run", "run"),
        "decay_rate_cs": ("Environments.Decay_Rate.CMA_ES_CS_run", "run"),
        "decay_rate_cs_imit": ("Environments.Decay_Rate.CMA_ES_CS_imit_run", "run"),
        "decay_rate_cc": ("Environments.Decay_Rate.CMA_ES_CC_run", "run"),
        "decay_rate_cc_imit": ("Environments.Decay_Rate.CMA_ES_CC_imit_run", "run"),
        "damping": ("Environments.Damping.CMA_ES_Damping_run", "run"),
        "damping_imit": ("Environments.Damping.CMA_ES_Damping_imit_run", "run")
    }
    return mapping.get(algorithm, ("", ""))

if __name__ == "__main__":
    main()
