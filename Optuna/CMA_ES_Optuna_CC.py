import optuna
import g_utils
import numpy as np
from tqdm import tqdm
from Baseline.CMA_ES_Baseline import CMAES


def run(
    dimension,
    x_start,
    reward_type,
    sigma,
    instance,
    max_eps_steps,
    train_repeats,
    test_repeats,
    pre_train_repeats,
    split,
    p_class,
    seed,
):
    print(
        "---------------Running Optuna for decay rate (cc) parameter learning---------------"
    )

    train_funcs, test_funcs = g_utils.split_train_test(
        dimension=dimension,
        instance=instance,
        split=split,
        p_class=p_class,
        train_repeats=train_repeats,
        test_repeats=test_repeats,
        random_state=seed,
    )

    def objective(trial):
        params = {
            "cc": trial.suggest_float("cc", 1e-3, 1),
        }
        try:
            objective_fvals = np.zeros(len(train_funcs))
            for index, train_func in enumerate(train_funcs):
                _x_start = (
                    np.zeros(train_func.dimension)
                    if x_start == 0
                    else np.random.uniform(low=-5, high=5, size=train_func.dimension)
                )
                es_optuna = CMAES(x_start=_x_start, sigma=sigma, parameters=params)
                while not es_optuna.stop():
                    X = es_optuna.ask()
                    fit = [train_func(x) for x in X]
                    es_optuna.tell(X, fit)
                objective_fvals[index] = g_utils.calc_reward(
                    optimum=train_func.best_value(),
                    min_eval=np.min(fit),
                    reward_type=reward_type,
                    reward_targets=g_utils.set_reward_targets(train_func.best_value()),
                )
            return np.sum(objective_fvals)
        except Exception as e:
            print(f"An error occurred in trial {trial.number}: {e}")
            return -np.inf

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=1)

    random_best_param = np.random.choice(
        np.where(
            np.array([trial.values[0] for trial in study.trials]) == study.best_value
        )[0]
    )
    best_params = study.trials[random_best_param].params

    p_class = p_class if split == "classes" else -1
    np.savez(
        f"Optuna/opt_cc_params_{dimension}D_{instance}I_{p_class}C.npz",
        **best_params,
    )

    results = []
    groups = {}
    for index, test_func in enumerate(test_funcs):
        key = test_func.id
        if key not in groups:
            groups[key] = []
        groups[key].append(index)

    for key, indices in tqdm(groups.items()):
        grp_rewards = np.zeros(len(indices))
        reward_index = 0
        for index in indices:
            test_func = test_funcs[index]
            _x_start = (
                np.random.uniform(
                    low=-5,
                    high=5,
                    size=test_func.dimension,
                )
                if x_start == -1
                else np.zeros(test_func.dimension)
            )
            es_optuna = CMAES(x_start=_x_start, sigma=sigma, parameters=best_params)
            while not es_optuna.stop():
                X = es_optuna.ask()
                fit = [test_func(x) for x in X]
                es_optuna.tell(X, fit)
            grp_rewards[reward_index] = np.abs(test_func.best_value() - np.min(fit))
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
    g_utils.save_results(
        results=results, policy=f"optuna_cc_{dimension}D_{instance}I_{p_class}C"
    )
