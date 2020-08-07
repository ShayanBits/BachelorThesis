#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --partition=gpu2
#SBATCH --mem-per-cpu=10000M   # memory per CPU core
#SBATCH -J "test-shayan"   # job name
#SBATCH --mail-user=shayan.shahpasand@mailbox.tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
#SBATCH -A p_ml_nimi


source /home/shsh829c/venv/env1/bin/activate

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="shayan-test-output.log"


echo $PWD
/scratch/ws/1/shsh829c-KGE/KGE_Pattern/parallel_run_shayan.sh  > "$OUTFILE"

exit 0