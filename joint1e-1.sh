#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=4
#PBS -l walltime=50:00:00
#PBS -P 12003770
#PBS -q ai

#PBS -o experiments_lightning/mse+1e-1contGradnoPenumbra_out.txt
#PBS -e experiments_lightning/mse+1e-1contGradnoPenumbra_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script
# python finetune_sam_with_bbox.py
python soft_mask_diffusion_joint.py


# You can add additional commands here as needed
