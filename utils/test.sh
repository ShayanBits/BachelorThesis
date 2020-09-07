#!/usr/bin/env bash


counter=0
input="commands.txt"
while IFS= read -r line
do
source /home/shsh829c/venv/env1/bin/activate
tmux new-session -d -s my_session${counter} 'srun --time=1:00:00 --nodes=1 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --partition=gpu2-interactive --mem-per-cpu=10000M -J test-checkpoint -o test-checkpoint-slurm-%j.out --mail-user=shayan.shahpasand@mailbox.tu-dresden.de --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90 -A p_ml_nimi $line'
counter=${counter+1}
  echo my_session${counter}
  echo 'srun --time=1:00:00 --nodes=1 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --partition=gpu2-interactive --mem-per-cpu=10000M -J test-checkpoint -o test-checkpoint-slurm-%j.out --mail-user=shayan.shahpasand@mailbox.tu-dresden.de --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90 -A p_ml_nimi' ${line}
done < "$input"