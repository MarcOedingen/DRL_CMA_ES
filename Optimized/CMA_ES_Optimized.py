import cma
import g_utils
import numpy as np
from tqdm import tqdm


def run(dimension, x_start, sigma, instance, split, p_class, test_repeats, seed):
    print("---------------Running CMA-ES Optimized---------------")
    p_class = p_class if split == "classes" else -1
    _, functions = g_utils.split_train_test(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        test_repeats=test_repeats,
        random_state=seed,
    )

    results = []
    groups = {}
    for index, test_func in enumerate(functions):
        key = test_func.id
        if key not in groups:
            groups[key] = []
        groups[key].append(index)

    for key, indices in tqdm(groups.items()):
        grp_rewards = np.zeros(len(indices))
        reward_index = 0
        for index in indices:
            test_func = functions[index]
            _x_start = (
                np.random.uniform(
                    low=-5,
                    high=5,
                    size=test_func.dimension,
                )
                if x_start == -1
                else np.zeros(test_func.dimension)
            )
            es = cma.CMAEvolutionStrategy(_x_start, sigma, {"verbose": -9})
            while not es.stop():
                X = es.ask()
                fit = [test_func(x) for x in X]
                es.tell(X, fit)
            grp_rewards[reward_index] = np.abs(test_func.best_value() - min(fit))
            reward_index += 1
        results.append(
            {
                "id": key,
                "stats": np.array(
                    [
                        np.mean(grp_rewards),
                        np.std(grp_rewards),
                        np.max(grp_rewards),
                        np.min(grp_rewards),
                    ]
                ),
            }
        )
    g_utils.print_pretty_table(results=results)
    means = [row["stats"][0] for row in results]
    print(f"Mean difference of all test functions: {np.mean(means)} ± {np.std(means)}")
    p_class = p_class if split == "classes" else -1
    g_utils.save_results(
        results=results, policy=f"optimized_{dimension}D_{instance}I_{p_class}C"
    )
