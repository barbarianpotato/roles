#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu[005]
#SBATCH --output=test.out
#SBATCH --job-name="roles"

hostname
# module load conda-python/3.7
module load cuda/9.0
module load cuDNN/cuda_9.2_7.2.1

# conda activate env1
# python predict.py karate
# python predict.py dolphins
# python predict.py jazz
# python predict.py usair97
python predict.py fb_U
python predict.py GrQc_U
python predict.py PB_U
python predict.py power
python predict.py hep-th
python predict.py cora
python predict.py celegansneural
python predict.py netscience
python predict.py yeast