#!/bin/bash
#dims=(50 1000)
dims=(50)
#gamma=(1 5 10 20 30 40 50 100)
gamma=(1 5)
temperature=1
#lrs=(0.01  0.05  0.1)
lrs=(0.01)
#batch_sizes=(512 1024)
batch_sizes=(512)
negs=(10)
dataset="FB15k"
model="RotatE"
train_with_groundings="false"
plot="false"
max_steps=400000

CODE_PATH="../codes"
DATA_PATH="../data/FB15k"
LOSS_FUNC=("rotate")
SAVE_PATH="../models/$model/$LOSS_FUNC/$dataset"

#Submit this script with: sbatch thefilename

#SBATCH --time=14:00:00   # walltime
#SBATCH --array=1-2
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=62000M   # memory per CPU core
#SBATCH -J "test-shayan"   # job name
#SBATCH -o test-shayan-slurm-_%A_%a.out
#SBATCH --mail-user=shayan.shahpasand@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi

source /home/shsh829c/venv/env1/bin/activate

for d in "${dims[@]}";do
  for g in "${gamma[@]}";do
    for lr in "${lrs[@]}";do
      for b in "${batch_sizes[@]}";do
        for neg in "${negs[@]}";do
          for loss in "${LOSS_FUNC[@]}";do
           command="python3 $CODE_PATH/run.py --do_grid --cuda --do_test --data_path $DATA_PATH --model $model -d $d --negative_sample_size $neg --batch_size $b --gamma $g --adversarial_temperature $temperature --negative_adversarial_sampling -lr $lr --max_steps $max_steps -save $SAVE_PATH/dim-$d/gamma-$g/learning-rate-$lr/batch-size-$b/negative-sample-size-$neg/ -de --loss $loss"
#           srunCommand="srun --time=1:00:00 --nodes=1 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --partition=gpu2-interactive -J "test-shayan" -o "test-shayan-slurm-%j.log" --mail-user="shayan.shahpasand@mailbox.tu-dresden.de" --mail-type="ALL" -A "p_ml_nimi" $command "
           echo  "following command is executed"
           echo  $command
           $command
           echo $PWD
           sleep 1
done
done
done
done
done
done

#/scratch/ws/1/shsh829c-KGE/KGE_Pattern/parallel_run_shayan.sh

exit 0