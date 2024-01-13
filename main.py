import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="Run different CMA-ES algorithms")
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The dataset to use",
        choices=[
            "baseline",
            "optimized",
            "step_size",
            "step_size_imit",
            "decay_rate_cs",
            "decay_rate_cs_imit",
            "decay_rate_cc",
            "decay_rate_cc_imit",
            "damping",
            "damping_imit",
            "learning_rate_c1",
            "learning_rate_c1_imit",
            "learning_rate_cm",
            "learning_rate_cm_imit",
            "mu_effective",
            "mu_effective_imit",
            "h_sigma",
            "h_sigma_imit",
            "evolution_path_ps",
            "evolution_path_ps_imit",
            "testing",
            "eval"
        ],
        default="evolution_path_ps_imit",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="The dimension of the problem",
        choices=[i for i in range(-1, 41) if i != 0 and i != 1],
        default=2,
    )
    parser.add_argument(
        "--x_start",
        type=int,
        help="The x-values starting from",
        choices=[-1, 0],
        default=0,
    )
    parser.add_argument(
        "--instance",
        type=int,
        help="The instance of the problem",
        choices=[i for i in range(-1, int(1e3) + 1) if i != 0],
        default=1,
    )
    parser.add_argument("--sigma", type=float, help="The initial sigma", default=0.5)
    parser.add_argument(
        "--policy",
        type=str,
        help="The model to use",
        choices=[
            "ppo_policy_ss",
            "ppo_policy_ss_imit",
            "ppo_policy_cs",
            "ppo_policy_cs_imit",
            "ppo_policy_cc",
            "ppo_policy_cc_imit",
            "ppo_policy_dp",
            "ppo_policy_dp_imit",
            "ppo_policy_c1",
            "ppo_policy_c1_imit",
            "ppo_policy_cm",
            "ppo_policy_cm_imit",
            "ppo_policy_me",
            "ppo_policy_me_imit",
            "ppo_policy_hs",
            "ppo_policy_hs_imit",
            "ppo_policy_ps",
            "ppo_policy_ps_imit",
        ],
        default="ppo_policy_ss",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        help="The max episode steps",
        default=int(1e3 * 40**2),
    )
    parser.add_argument(
        "--train_repeats",
        type=int,
        help="The number of repeats for the training functions",
        default=10,
    )
    parser.add_argument(
        "--test_repeats",
        type=int,
        help="The number of repeats for the test functions",
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generator",
        default=4567,
    )

    args = parser.parse_args()

    if args.algorithm == "testing":
        module = importlib.import_module("run_model")
        run_function = getattr(module, "run")
        run_function(
            args.dimension, args.x_start, args.sigma, args.instance, args.policy
        )
    elif args.algorithm == "baseline" or args.algorithm == "optimized":
        module_path, function_name = get_module_and_function(args.algorithm)
        module = importlib.import_module(module_path)
        run_function = getattr(module, function_name)
        run_function(args.dimension, args.x_start, args.sigma, args.instance)
    elif args.algorithm == "eval":
        module = importlib.import_module("Results.Eval_Results")
        run_function = getattr(module, "run")
        run_function(args.policy, args.dimension, args.instance)
    else:
        module_path, function_name = get_module_and_function(args.algorithm)
        module = importlib.import_module(module_path)
        run_function = getattr(module, function_name)
        run_function(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
            args.seed,
        )


def get_module_and_function(algorithm):
    mapping = {
        "baseline": ("Baseline.CMA_ES_Baseline", "run"),
        "optimized": ("Optimized.CMA_ES_Optimized", "run"),
        "step_size": ("Environments.Step_Size.CMA_ES_SS_run", "run"),
        "step_size_imit": ("Environments.Step_Size.CMA_ES_SS_Imit", "run"),
        "decay_rate_cs": ("Environments.Decay_Rate.CMA_ES_CS_run", "run"),
        "decay_rate_cs_imit": ("Environments.Decay_Rate.CMA_ES_CS_Imit", "run"),
        "decay_rate_cc": ("Environments.Decay_Rate.CMA_ES_CC_run", "run"),
        "decay_rate_cc_imit": ("Environments.Decay_Rate.CMA_ES_CC_Imit", "run"),
        "damping": ("Environments.Damping.CMA_ES_DP_run", "run"),
        "damping_imit": ("Environments.Damping.CMA_ES_DP_Imit", "run"),
        "learning_rate_c1": ("Environments.Learning_Rate.CMA_ES_C1_run", "run"),
        "learning_rate_c1_imit": ("Environments.Learning_Rate.CMA_ES_C1_Imit", "run"),
        "learning_rate_cm": ("Environments.Learning_Rate.CMA_ES_CM_run", "run"),
        "learning_rate_cm_imit": ("Environments.Learning_Rate.CMA_ES_CM_Imit", "run"),
        "mu_effective": ("Environments.Mu_Effective.CMA_ES_ME_run", "run"),
        "mu_effective_imit": ("Environments.Mu_Effective.CMA_ES_ME_Imit", "run"),
        "h_sigma": ("Environments.h_Sigma.CMA_ES_HS_run", "run"),
        "h_sigma_imit": ("Environments.h_Sigma.CMA_ES_HS_Imit", "run"),
        "evolution_path_ps": ("Environments.Evolution_Path.CMA_ES_PS_run", "run"),
        "evolution_path_ps_imit": ("Environments.Evolution_Path.CMA_ES_PS_Imit", "run"),
        "eval": ("Results.Eval_Results", "run"),
    }
    return mapping.get(algorithm, ("", ""))


if __name__ == "__main__":
    main()
