#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=40:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=2  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=5000M   # memory per CPU core
#SBATCH -J "fb15k-new-grid-round-two"   # job name
#SBATCH -o fb15k-new-grid-round-two%j.out
#SBATCH --mail-user=shayan.shahpasand@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi
#SBATCH --reservation=p_ml_nimi_105
#SBATCH --array=1-270


source /home/shsh829c/venv/env1/bin/activate

srun $(head -n $SLURM_ARRAY_TASK_ID commands.txt | tail -n 1)

exit 0