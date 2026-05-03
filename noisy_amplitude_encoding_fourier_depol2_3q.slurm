#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM

# Partition for the job:
#SBATCH --partition=gpu-h100

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="single_parallel_amplitude_encoding_fourier"

# The project ID which this job should run under:
#SBATCH --account="punim2554"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G

# Number of GPUs requested per node:
#SBATCH --gres=gpu:2
# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=5-0:00:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load Python/3.10.4
module load mpi4py/3.1.4
module load SciPy-bundle/2022.05
module load PyTorch/2.1.2-CUDA-12.2.0
module load torchvision/0.16.0-CUDA-12.2.0

source ~/venvs/env_noisy_qml/bin/activate

export MASTER_ADDR=127.0.0.1
PORT=$((29500 + $SLURM_JOB_ID % 1000))
if ss -ltn | grep ":$PORT " ; then
  echo "Port $PORT is already in use!"
  exit 1
else
  export MASTER_PORT=$PORT
fi

python3 noisy_amplitude_encoding_fourier_depol2_3q.py > $SLURM_JOBID.log

##Job monitor command to list the resource usage
my-job-stats -a -n -s