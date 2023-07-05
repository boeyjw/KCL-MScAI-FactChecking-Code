#!/bin/bash -l

#SBATCH --job-name=baseline-ir-6
#SBATCH --partition=cpu
#SBATCH --signal=USR2
#SBATCH --output=/scratch/users/%u/slurm-out/%j.out
#SBATCH --mem=500G
#SBATCH --ntasks=1
#SBATCH --mincpus=10
#SBATCH --time=1-01:00

module load anaconda3/2021.05-gcc-9.4.0

cd /users/k21190024/study/fact-checking-repos/fever/baseline
source activate `which conda`
source activate /scratch/users/k21190024/envs/conda/fever-baseline

python /users/k21190024/study/fact-check-transfer-learning/nb/03_CFEVER_Zero_Shot/01b_Baseline_IR-FEVER.py
