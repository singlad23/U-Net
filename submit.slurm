#!/bin/bash
#SBATCH --nodes=1                               # Get one node
#SBATCH --cpus-per-task=2                       # Two cores per task
#SBATCH --ntasks=2                              # But only one task
#SBATCH --gres=gpu:1                            # And two GPU
#SBATCH --gres-flags=enforce-binding            # Insist on good CPU/GPU alignment
#SBATCH --time=1-00:00:00                       # Run for 7 days, at most
#SBATCH --job-name=Pretrained0.0001                      # Name the job so I can see it in squeue
#SBATCH --output=UCF101_Pytorch-%J.out                 # %J is for string formatting, job id
#SBATCH --mail-type=BEGIN,END,FAIL              # Send me email for various states
#SBATCH --mail-user singlad23@trinityprep.org   # Use this address


# Load modules
module load gcc/gcc-8.3.0
module load anaconda/anaconda3
module load cuda/cuda-10.2
module load opencv

module list                                     # Have Nvidia tell us the GPU/CPU mapping so we know

nvidia-smi topo -m

source activate reu2021

python3 --version

CUDA_VISIBLE_DEVICES=0 python3 -u Main.py
