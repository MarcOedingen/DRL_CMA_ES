import argparse


def main():
    parser = argparse.ArgumentParser(description="Run different CMA-ES algorithms")
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The dataset to use",
        choices=["baseline", "optimized", "step_size", "step_size_imit", "testing"],
        default="step_size",
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
    parser.add_argument("--sigma", type=float, help="The initial sigma", default=0.5)
    parser.add_argument(
        "--instance",
        type=int,
        help="The instance of the problem",
        choices=[i for i in range(-1, int(1e3) + 1) if i != 0],
        default=1,
    )

    parser.add_argument(
        "--policy",
        type=str,
        help="The model to use",
        choices=["ppo_policy_ss", "ppo_policy_ss_imit"],
        default="ppo_policy_ss",
    )

    args = parser.parse_args()
    if args.algorithm == "baseline":
        from Baseline.CMA_ES_Baseline import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "optimized":
        from Optimized.CMA_ES_Optimized import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "step_size":
        from Environments.Step_Size.CMA_ES_SS_run import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "step_size_imit":
        from Environments.Step_Size.CMA_ES_SS_IMIT import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "testing":
        from run_model import run

        run(args.dimension, args.x_start, args.sigma, args.instance, args.policy)


if __name__ == "__main__":
    main()
