#!/usr/bin/env bash

temperature=1
batch_sizes=(512)
lrs=(0.01 0.05 0.1)
dims=(10 100 1000)
negs=(10 100 1000)
gamma=(1 10 20 30 40 50)
dataset="wn18rr"
models=("TransE" "RotatE" "ComplEx" "QuatE" "DistMult")
LOSS_FUNC=("rotate" "margin_ranking" "adaptive_margin")
max_steps=400000

RUN_ON_GPU=false
AVAIL_GPUS=4
FREE_MEM_THRESHOLD=2000

CODE_PATH="../codes"
DATA_PATH="../data/$dataset"


echo "starting grid run on all variables"

for model in "${models[@]}";do
for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          for loss in "${LOSS_FUNC[@]}";do
            if [ $RUN_ON_GPU = "true" ]
            then
              for gpu_number in $AVAIL_GPUS;do
                SAVE_PATH="../models/$model/$loss/$dataset"
                COMPLETE_SAVE_PATH="$SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/"
                available_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i ${gpu_number})
#               extract the integer value in MB
                available_mem=${available_mem//[^0-9]/}
                echo "available mem is: $available_mem"
                if [[ ${available_mem} -gt $FREE_MEM_THRESHOLD ]];then
                   echo "free memory of GPU $gpu_number: $available_mem"
                   command="CUDA_VISIBLE_DEVICES=$gpu_number python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/ -de --loss $loss"
                   $command
                   echo  "following command is executed"
                   echo  $command >> utils/commands.txt
                   sleep 15
                   break
               fi
               sleep 5
                done
            else
              SAVE_PATH="../models/$model/$loss/$dataset"
              COMPLETE_SAVE_PATH="$SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/"
              command="python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $COMPLETE_SAVE_PATH -de --loss $loss --init_checkpoint $COMPLETE_SAVE_PATH"
              echo  "following command is executed"
              echo  $command >> utils/commands.txt
              fi
done
done
done
done
done
done
done

wait
echo "all executed commands are finished"