import argparse
import importlib
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run different CMA-ES algorithms")
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The dataset to use",
        choices=[
            "baseline",
            "baseline_ipop",
            "optimized",
            "step_size",
            "step_size_imit",
            "step_size_imit_iter",
            "decay_rate_cs",
            "decay_rate_cs_imit",
            "decay_rate_cs_imit_iter",
            "decay_rate_cc",
            "decay_rate_cc_imit",
            "decay_rate_cc_imit_iter",
            "damping",
            "damping_imit",
            "damping_imit_iter",
            "learning_rate_c1",
            "learning_rate_c1_imit",
            "learning_rate_c1_imit_iter",
            "learning_rate_cm",
            "learning_rate_cm_imit",
            "learning_rate_cm_imit_iter",
            "mu_effective",
            "mu_effective_imit",
            "mu_effective_imit_iter",
            "h_sigma",
            "h_sigma_imit",
            "h_sigma_imit_iter",
            "chi_n",
            "chi_n_imit",
            "chi_n_imit_iter",
            "chi_n_ipop",
            "chi_n_ipop_imit",
            "comb",
            "comb_imit",
            "comb_imit_iter",
            "static",
            "static_imit",
            "static_imit_iter",
            "evolution_path_ps",
            "evolution_path_ps_imit",
            "evolution_path_ps_imit_iter",
            "evolution_path_pc",
            "evolution_path_pc_imit",
            "evolution_path_pc_imit_iter",
            "optuna_st",
            "optuna_st_imit",
            "optuna_chi_n",
            "optuna_mu_effective",
            "optuna_cc",
            "optuna_cs",
            "optuna_c1",
            "optuna_cm",
            "optuna_dp",
            "eval",
        ],
        default="baseline",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="The dimension of the problem",
        choices=[-1, 2, 3, 5, 10, 20, 40],
        default=10,
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
        choices=[i for i in range(-1, 10 + 1) if i != 0],
        default=1,
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        help="The reward function",
        choices=["log_opt", "ecdf"],
        default="ecdf",
    )
    parser.add_argument("--sigma", type=float, help="The initial sigma", default=0.5)
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
        default=1,
    )
    parser.add_argument(
        "--test_repeats",
        type=int,
        help="The number of repeats for the test functions",
        default=25,
    )
    parser.add_argument(
        "--pre_train_repeats",
        type=int,
        help="The number of repeats for pre-training PPO",
        default=5,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="The split of the functions",
        choices=["functions", "classes"],
        default="classes",
    )
    parser.add_argument(
        "--p_class",
        type=int,
        help="The class of the functions",
        choices=[i for i in range(1, 6)],
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generator",
        default=7570,
    )

    subprocess.run(["python3", "experiment_prep.py"], check=True)

    args = parser.parse_args()
    if args.algorithm == "baseline" or args.algorithm == "optimized" or args.algorithm == "baseline_ipop":
        module_path, function_name = get_module_and_function(args.algorithm)
        module = importlib.import_module(module_path)
        run_function = getattr(module, function_name)
        run_function(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.split,
            args.p_class,
            args.test_repeats,
            args.seed,
        )
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
            args.reward_type,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
            args.pre_train_repeats,
            args.split,
            args.p_class,
            args.seed,
        )


def get_module_and_function(algorithm):
    mapping = {
        "baseline": ("Baseline.CMA_ES_Baseline", "run"),
        "baseline_ipop": ("Baseline.CMA_ES_Baseline_IPOP", "run"),
        "optimized": ("Optimized.CMA_ES_Optimized", "run"),
        "step_size": ("Environments.Step_Size.CMA_ES_SS_run", "run"),
        "step_size_imit": ("Environments.Step_Size.CMA_ES_SS_Imit", "run"),
        "step_size_imit_iter": ("Environments.Step_Size.CMA_ES_SS_Imit_Iter", "run"),
        "decay_rate_cs": ("Environments.Decay_Rate.CMA_ES_CS_run", "run"),
        "decay_rate_cs_imit": ("Environments.Decay_Rate.CMA_ES_CS_Imit", "run"),
        "decay_rate_cs_imit_iter": (
            "Environments.Decay_Rate.CMA_ES_CS_Imit_Iter",
            "run",
        ),
        "decay_rate_cc": ("Environments.Decay_Rate.CMA_ES_CC_run", "run"),
        "decay_rate_cc_imit": ("Environments.Decay_Rate.CMA_ES_CC_Imit", "run"),
        "decay_rate_cc_imit_iter": (
            "Environments.Decay_Rate.CMA_ES_CC_Imit_Iter",
            "run",
        ),
        "damping": ("Environments.Damping.CMA_ES_DP_run", "run"),
        "damping_imit": ("Environments.Damping.CMA_ES_DP_Imit", "run"),
        "damping_imit_iter": ("Environments.Damping.CMA_ES_DP_Imit_Iter", "run"),
        "learning_rate_c1": ("Environments.Learning_Rate.CMA_ES_C1_run", "run"),
        "learning_rate_c1_imit": ("Environments.Learning_Rate.CMA_ES_C1_Imit", "run"),
        "learning_rate_c1_imit_iter": (
            "Environments.Learning_Rate.CMA_ES_C1_Imit_Iter",
            "run",
        ),
        "learning_rate_cm": ("Environments.Learning_Rate.CMA_ES_CM_run", "run"),
        "learning_rate_cm_imit": ("Environments.Learning_Rate.CMA_ES_CM_Imit", "run"),
        "learning_rate_cm_imit_iter": (
            "Environments.Learning_Rate.CMA_ES_CM_Imit_Iter",
            "run",
        ),
        "mu_effective": ("Environments.Mu_Effective.CMA_ES_ME_run", "run"),
        "mu_effective_imit": ("Environments.Mu_Effective.CMA_ES_ME_Imit", "run"),
        "mu_effective_imit_iter": (
            "Environments.Mu_Effective.CMA_ES_ME_Imit_Iter",
            "run",
        ),
        "h_sigma": ("Environments.h_Sigma.CMA_ES_HS_run", "run"),
        "h_sigma_imit": ("Environments.h_Sigma.CMA_ES_HS_Imit", "run"),
        "h_sigma_imit_iter": ("Environments.h_Sigma.CMA_ES_HS_Imit_Iter", "run"),
        "chi_n": ("Environments.ChiN.CMA_ES_CN_run", "run"),
        "chi_n_imit": ("Environments.ChiN.CMA_ES_CN_Imit", "run"),
        "chi_n_imit_iter": ("Environments.ChiN.CMA_ES_CN_Imit_Iter", "run"),
        "chi_n_ipop": ("Environments.ChiN.CMA_ES_IPOP_CN_run", "run"),
        "chi_n_ipop_imit": ("Environments.ChiN.CMA_ES_IPOP_CN_Imit", "run"),
        "comb": ("Environments.Combined.CMA_ES_COMB_run", "run"),
        "comb_imit": ("Environments.Combined.CMA_ES_COMB_Imit", "run"),
        "comb_imit_iter": ("Environments.Combined.CMA_ES_COMB_Imit_Iter", "run"),
        "static": ("Environments.Combined.CMA_ES_ST_run", "run"),
        "static_imit": ("Environments.Combined.CMA_ES_ST_Imit", "run"),
        "static_imit_iter": ("Environments.Combined.CMA_ES_ST_Imit_Iter", "run"),
        "optuna_st": ("Optuna.CMA_ES_Optuna_ST", "run"),
        "optuna_st_imit": ("Optuna.CMA_ES_Optuna_ST_Imit", "run"),
        "optuna_chi_n": ("Optuna.CMA_ES_Optuna_CN", "run"),
        "optuna_mu_effective": ("Optuna.CMA_ES_Optuna_ME", "run"),
        "optuna_cc": ("Optuna.CMA_ES_Optuna_CC", "run"),
        "optuna_cs": ("Optuna.CMA_ES_Optuna_CS", "run"),
        "optuna_c1": ("Optuna.CMA_ES_Optuna_C1", "run"),
        "optuna_cm": ("Optuna.CMA_ES_Optuna_CM", "run"),
        "optuna_dp": ("Optuna.CMA_ES_Optuna_DP", "run"),
        "evolution_path_ps": ("Environments.Evolution_Path.CMA_ES_PS_run", "run"),
        "evolution_path_ps_imit": ("Environments.Evolution_Path.CMA_ES_PS_Imit", "run"),
        "evolution_path_ps_imit_iter": (
            "Environments.Evolution_Path.CMA_ES_PS_Imit_Iter",
            "run",
        ),
        "evolution_path_pc": ("Environments.Evolution_Path.CMA_ES_PC_run", "run"),
        "evolution_path_pc_imit": ("Environments.Evolution_Path.CMA_ES_PC_Imit", "run"),
        "evolution_path_pc_imit_iter": (
            "Environments.Evolution_Path.CMA_ES_PC_Imit_Iter",
            "run",
        ),
        "eval": ("Results.Eval_Results", "run"),
    }
    return mapping.get(algorithm, ("", ""))


if __name__ == "__main__":
    main()
