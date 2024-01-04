import argparse


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
            "testing",
        ],
        default="step_size_imit",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        help="The dimension of the problem",
        choices=[i for i in range(-1, 41) if i != 0 and i != 1],
        default=10,
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

    args = parser.parse_args()
    if args.algorithm == "baseline":
        from Baseline.CMA_ES_Baseline import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "optimized":
        from Optimized.CMA_ES_Optimized import run

        run(args.dimension, args.x_start, args.sigma, args.instance)

    elif args.algorithm == "step_size":
        from Environments.Step_Size.CMA_ES_SS_run import run

        run(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
        )

    elif args.algorithm == "step_size_imit":
        from Environments.Step_Size.CMA_ES_SS_IMIT import run

        run(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
        )

    elif args.algorithm == "decay_rate_cs":
        from Environments.Decay_Rate.CMA_ES_CS_run import run

        run(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
        )

    elif args.algorithm == "decay_rate_cs_imit":
        from Environments.Decay_Rate.CMA_ES_CS_Imit import run

        run(
            args.dimension,
            args.x_start,
            args.sigma,
            args.instance,
            args.max_episode_steps,
            args.train_repeats,
            args.test_repeats,
        )

    elif args.algorithm == "testing":
        from run_model import run

        run(args.dimension, args.x_start, args.sigma, args.instance, args.policy)


if __name__ == "__main__":
    main()
