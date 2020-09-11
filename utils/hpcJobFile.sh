#!/bin/bash

#Submit this script with: sbatch thefilename
#SBATCH --time=20:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=10000M   # memory per CPU core
#SBATCH -J "wn18-new-grid-fillup-checkpoint"   # job name
#SBATCH -o wn18-new-grid-fillup-checkpoint%j.out
#SBATCH --mail-user=shayan.shahpasand@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi
#SBATCH --array=1-3


source /home/sava096c/envs/env01/bin/activate

srun $(head -n $SLURM_ARRAY_TASK_ID commands.txt | tail -n 1)

exit 0