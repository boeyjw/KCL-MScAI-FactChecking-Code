#!/bin/bash -l

#SBATCH --job-name=jdistorch
#SBATCH --partition=cpu
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/slurm-out/%j.out
#SBATCH --mem=60G
#SBATCH --ntasks=1
#SBATCH --mincpus=12
#SBATCH --time=1-01:00

module load anaconda3/2021.05-gcc-9.4.0
module load openjdk/1.8.0_265-b01-gcc-9.4.0

# get unused socket per https://unix.stackexchange.com/a/132524
readonly DETAIL=$(python -c 'import datetime; print(datetime.datetime.now())')
readonly IPADDRESS=$(hostname -I | tr ' ' '\n' | grep '10.211.4.')
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

   ssh -NL 8889:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   and point your web browser to http://localhost:8889/lab?token=<add the token from the jupyter output below>

Time started: ${DETAIL}

When done using the notebook, terminate the job by
issuing the following command on the login node:

      scancel -f ${SLURM_JOB_ID}

END

source activate `which conda`
source activate /scratch/users/k21190024/envs/conda/p-dis-torch
jupyter-lab --port=${PORT} --ip=${IPADDRESS} --no-browser --notebook-dir=${HOME}/study/fact-check-transfer-learning

printf 'notebook exited' 1>&2
