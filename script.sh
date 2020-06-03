#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu[002]
#SBATCH --job-name="roles"

hostname
# module load conda-python/3.7
module load cuda/10.1
module load cuDNN/cuda_9.2_7.2.1

# conda activate roles
python run.py r karate
python run.py r dolphins
python run.py r jazz
python run.py r usair97
python run.py r fb_U
python run.py r GrQc_U
python run.py r PB_U
python run.py r power
python run.py r adjnoun
python run.py r celegansneural
python run.py r netscience
python run.py r LesMiserables
python run.py r football