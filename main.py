import argparse


def main():
    parser = argparse.ArgumentParser(description="Run different CMA-ES algorithms")
    parser.add_argument(
        "--algorithm",
        type=str,
        help="The dataset to use",
        choices=["baseline", "optimized", "step_size", "step_size_imit"],
        default="optimized",
    )
    parser.add_argument(
        "--dimension", type=int, help="The dimension of the problem", default=2
    )
    parser.add_argument(
        "--x_start",
        type=str,
        help="The x-values starting from",
        choices=["zero", "random"],
        default="zero",
    )
    parser.add_argument("--sigma", type=float, help="The initial sigma", default=0.5)
    parser.add_argument(
        "--instance", type=int, help="The instance of the problem", default=1
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


if __name__ == "__main__":
    main()
