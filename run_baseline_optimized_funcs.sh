#!/bin/bash
declare algorithms=("baseline" "optimized")
declare dimensions=(2 3 5 10 20 40)
declare x_start=-1
declare reward_type="ecdf"
declare sigma=0.5
declare instance=1
declare max_episode_steps=1600000
declare train_repeats=2
declare test_repeats=25
declare pre_train_repeats=2
declare split="functions"
declare p_class=2
declare seeds=(7784 7570 7592 9466 3606 1143 2892 42 3290 3722)

for dimension in "${dimensions[@]}"
do
    for algorithm in "${algorithms[@]}"
    do
      for seed in "${seeds[@]}"
      do
          python main.py --algorithm $algorithm --dimension $dimension --x_start $x_start --reward_type $reward_type --sigma $sigma --instance $instance --max_episode_steps $max_episode_steps --train_repeats $train_repeats --test_repeats $test_repeats --pre_train_repeats $pre_train_repeats --split $split --p_class $p_class --seed $seed
      done
    done
done
