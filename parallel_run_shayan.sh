#!/usr/bin/env bash

dims=(50 1000)
gamma=(1 5 10 20 30 40 50 100)
temperature=1
lrs=(0.01  0.05  0.1)
batch_sizes=(512 1024)
negs=(10 50 100 500 1000)
dataset="FB15k"
model="RotatE"
train_with_groundings="false"
plot="false"
max_steps=400000

DATA_PATH="data/FB15k"
SAVE_PATH="models/$model/$dataset"
loss_func=("margin_ranking")



for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          for loss in "${loss_func[@]}";do
#          change next line to actual command before running on server
            command="python3 run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -dim $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH -de --loss $loss "
            echo  $command
done
done
done
done
done
done
