#!/bin/bash
declare algorithms=("chi_n_imit_iter" "comb_imit_iter" "static_imit_iter" "damping_imit_iter" "decay_rate_cc_imit_iter" "decay_rate_cs_imit_iter" "evolution_path_pc_imit_iter" "evolution_path_ps_imit_iter" "h_sigma_imit_iter" "learning_rate_c1_imit_iter" "learning_rate_cm_imit_iter" "mu_effective_imit_iter" "step_size_imit_iter")
declare dimensions=(2 3 5 10 20 40)
declare x_start=-1
declare reward_type="ecdf"
declare sigma=0.5
declare instance=1
declare max_episode_steps=1600000
declare train_repeats=2
declare test_repeats=25
declare pre_train_repeats=2
declare split="classes"
declare p_classes=(1 2 3 4 5)
declare seeds=(7784 7570 7592 9466 3606 1143 2892 42 3290 3722)

for dimension in "${dimensions[@]}"
do
    for algorithm in "${algorithms[@]}"
    do
      for p_class in "${p_classes[@]}"
      do
        for seed in "${seeds[@]}"
        do
            python3 main.py --algorithm $algorithm --dimension $dimension --x_start $x_start --reward_type $reward_type --sigma $sigma --instance $instance --max_episode_steps $max_episode_steps --train_repeats $train_repeats --test_repeats $test_repeats --pre_train_repeats $pre_train_repeats --split $split --p_class $p_class --seed $seed
        done
      done
    done
done
