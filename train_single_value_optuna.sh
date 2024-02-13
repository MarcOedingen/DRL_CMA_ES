#!/bin/bash
declare algorithms=("optuna_chi_n" "optuna_mu_effective" "optuna_cc" "optuna_cs" "optuna_c1" "optuna_cm" "optuna_dp")
declare dimensions=(2 3 5 10 20 40)
declare x_start=-1
declare instance=1
declare reward_type="ecdf"
declare sigma=0.5
declare max_episode_steps=1600000
declare test_repeats=25
declare split="functions"
declare seeds=(7784 7570 7592 9466 3606 1143 2892 42 3290 3722)

for dimension in "${dimensions[@]}"
do
    for algorithm in "${algorithms[@]}"
    do
        for seed in "${seeds[@]}"
        do
            python main.py --algorithm $algorithm --dimension $dimension --x_start $x_start --instance $instance --reward_type $reward_type --sigma $sigma --max_episode_steps $max_episode_steps --test_repeats $test_repeats --split $split --seed $seed
        done
    done
done
```
