#!/bin/sh
env="AFE"
algo="rmappo"
dataset="openml"
dataset_name="586"  # 586 589 607 616 618 620 637
exp="check"

echo "env is ${env}, dataset is ${dataset}, dataset name is ${dataset_name}, algo is ${algo}, exp is ${exp}"
python -u train/train_afe.py --use_valuenorm --use_popart --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 --n_training_threads 1 --n_rollout_threads 4     --num_mini_batch 4 --episode_length 20 --num_env_steps 200000 --ppo_epoch 10 --use_ReLU --gain 0.01 --lr 3e-4 --critic_lr 3e-4 --use_eval --no_local --dataset ${dataset} --dataset_name ${dataset_name}