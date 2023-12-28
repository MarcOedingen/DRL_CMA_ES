import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run different CMA-ES algorithms')
    parser.add_argument('--algorithm', type=str, help='The dataset to use', choices=['baseline', 'stepsize', 'stepsize_imit'],default='stepsize_imit')
    parser.add_argument('--dimension', type=int, help='The dimension of the problem', default=2)
    parser.add_argument('--xstart', type=str, help='The x-values starting from', choices=['zero', 'random'],default='zero')
    parser.add_argument('--sigma', type=float, help='The initial sigma', default=0.5)
    parser.add_argument('--instance', type=int, help='The instance of the problem', default=1)


    args = parser.parse_args()
    if args.algorithm == 'baseline':
        from Baseline.CMA_ES_Baseline import run
        x_start = np.zeros(args.dimension) if args.xstart == 'zero' else np.random.uniform(low=-5, high=5, size=args.dimension)
        run(args.dimension, x_start, args.sigma, args.instance)
    elif args.algorithm == 'stepsize':
        from Environments.Step_Size.CMA_ES_SS_run import run
        run(args.dimension, args.xstart, args.sigma, args.instance)
    elif args.algorithm == 'stepsize_imit':
        from Environments.Step_Size.CMA_ES_SS_IMIT import run
        run(args.dimension, args.xstart, args.sigma, args.instance)

if __name__ == '__main__':
    main()