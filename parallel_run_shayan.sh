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

CODE_PATH="../codes"
DATA_PATH="../data/FB15k"
LOSS_FUNC=("rotate")
SAVE_PATH="../models/$model/$LOSS_FUNC/$dataset"

executed_flag="false"

echo "starting grid run on all variables"

for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          for loss in "${LOSS_FUNC[@]}";do
            executed_flag="false"
            while [ $executed_flag != "true" ];do
            for cpu_number in {0..3};do
               available_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i ${cpu_number})
#               extract the integer value in MB
               available_mem=${available_mem//[^0-9]/}
#               echo "available mem is: $available_mem"
               if [[ ${available_mem} -gt 1980 ]];then
                   echo "free memory of GPU $cpu_number: $available_mem"
                   command="CUDA_VISIBLE_DEVICES=$cpu_number python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH -de --loss $loss"
                   CUDA_VISIBLE_DEVICES=$cpu_number python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH -de --loss $loss &&
                   echo  "following command is executed"
                   echo  $command
                   executed_flag="true"
                   sleep 15
                   break
               fi
               sleep 5
            done
done
done
done
done
done
done
done

wait
echo "all executed commands are finished"