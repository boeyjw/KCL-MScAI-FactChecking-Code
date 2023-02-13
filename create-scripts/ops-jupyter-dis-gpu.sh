#!/bin/bash -l

#SBATCH --job-name=jupyter-tensorflow-pnn
#SBATCH --partition=gpu
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/slurm-out/%j.out
#SBATCH --gres=gpu
#SBATCH --mem=20G
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8

module load python/3.8.12-gcc-9.4.0
module load py-tensorflow/2.4.1-gcc-9.4.0-cuda-python-3.8.12

# get unused socket per https://unix.stackexchange.com/a/132524
readonly DETAIL=$(python -c 'import datetime; print(datetime.datetime.now())')
readonly IPADDRESS=$(hostname -I | tr ' ' '\n' | grep '10.211.4.')
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

   ssh -NL 8888:${HOSTNAME}:${PORT} ${USER}@hpc.create.kcl.ac.uk

   and point your web browser to http://localhost:8888/lab?token=<add the token from the jupyter output below>

Time started: ${DETAIL}

When done using the notebook, terminate the job by
issuing the following command on the login node:

      scancel -f ${SLURM_JOB_ID}

END

source /scratch/users/k21190024/envs/p-dis-gpu/bin/activate
jupyter-lab --port=${PORT} --ip=${IPADDRESS} --no-browser --notebook-dir=${HOME}/study/fact-check-transfer-learning

printf 'notebook exited' 1>&2
