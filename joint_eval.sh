#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -P 12003770

#PBS -o experiments_lightning/mse_gt2_UIUC_out.txt
#PBS -e experiments_lightning/mse_gt2_UIUC_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script
# python finetune_sam_with_bbox.py
#python soft_mask_diffusion_joint.py \
#version=test_LRSS_UIUC \
#phase=test \
#save_result_path: "./experiments_lightning/joint_tune_sam_diffusion/mse+1e-1contGradnoPenumbra/199_LRSS" \
#test.dataroot="./dataset/LRSS_2/test" \
#test_bbox_path="./dataset/LRSS_2/test/bounding_boxes.yaml" \
#test_normalize=true

python soft_mask_diffusion_joint.py

# You can add additional commands here as needed
