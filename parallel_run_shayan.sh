#!/usr/bin/env bash

#dims=(10 100 1000)
#gamma=(1 10 20 30 40 50)
#temperature=1
#lrs=(0.01 0.05 0.1)
#batch_sizes=(1024)
#negs=(10 100 1000)
dims=(100)
gamma=(50)
temperature=1
lrs=(0.05)
batch_sizes=(512)
negs=(100)
#model="TransE"
models=("TransE")
#declare -a model=("TransE" "RotatE" "ComplEx" "QuatE" "Dismult")
dataset="wn18"
train_with_groundings="false"
plot="false"
max_steps=400000

CODE_PATH="../codes"
DATA_PATH="../data/$dataset"
LOSS_FUNC=("margin_ranking")
#LOSS_FUNC=("rotate" "margin_ranking" "adaptive_margin")

executed_flag="false"

echo "starting grid run on all variables"

for model in "${models[@]}";do
for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          for loss in "${LOSS_FUNC[@]}";do
            executed_flag="false"
#            while [ $executed_flag != "true" ];do
            for gpu_number in 0;do
                SAVE_PATH="../models/$model/$loss/$dataset"
#               available_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i ${gpu_number})
#               extract the integer value in MB
#               available_mem=${available_mem//[^0-9]/}
#               echo "available mem is: $available_mem"
#               if [[ ${available_mem} -gt 1980 ]];then
#                   echo "free memory of GPU $gpu_number: $available_mem"
#                   command="CUDA_VISIBLE_DEVICES=$gpu_number python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/ -de --loss $loss"
                   command="python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/ -de --loss $loss --init_checkpoint $SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/"
#                   srun --time=1:00:00 --nodes=1 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --partition=gpu2 -J "test-shayan" -o "test-shayan-slurm-%j.log" --mail-user="shayan.shahpasand@mailbox.tu-dresden.de" --mail-type="ALL" -A "p_ml_nimi" source /home/shsh829c/venv/env1/bin/activate OUTFILE="shayan-test-output.log"  CUDA_VISIBLE_DEVICES=$cpu_number python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH -de --loss $loss > "$OUTFILE" &
                   echo  "following command is executed"
                   echo  $command >> utils/commands.txt
#                   executed_flag="true"
#                   sleep 15
#                   break
#               fi
#               sleep 5
#            done
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