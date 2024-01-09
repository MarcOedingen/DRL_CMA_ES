#!/bin/bash
declare algorithms=("step_size_imit" "decay_rate_cs" "decay_rate_cs_imit" "decay_rate_cc" "decay_rate_cc_imit" "damping" "damping_imit" "learning_rate_c1" "learning_rate_c1_imit" "learning_rate_cm" "learning_rate_c_imit" "mu_effective" "mu_effective_imit" "h_sigma" "h_sigma_imit")
declare dimension=2
declare x_start=0
declare instance=1
declare sigma=0.5
declare max_episode_steps=1600000
declare train_repeats=5
declare test_repeats=1

for algorithm in "${algorithms[@]}"
do
    python main.py --algorithm $algorithm --dimension $dimension --x_start $x_start --instance $instance --sigma $sigma --max_episode_steps $max_episode_steps --train_repeats $train_repeats --test_repeats $test_repeats
done