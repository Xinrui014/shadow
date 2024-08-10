#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=2
#PBS -l walltime=40:00:00
#PBS -P personal-xiyu004


#PBS -o experiments_lightning/Patch_detach_mse+1e-1contGradnoPenumbra_version_2.1_out.txt
#PBS -e experiments_lightning/Patch_detach_mse+1e-1contGradnoPenumbra_version_2.1_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script

python soft_mask_diffusion.py \
version=Patch_detach_mse+1e-1contGradnoPenumbra_version_2.1 \
phase=train \
patch_tricky=true \
datasets.train.dataroot="./dataset/SRD_sam_mask_B/train" \
datasets.train.gt_mask_dir="./experiments_lightning/tune_sam+diffusion/Patch_detach_mse+1e-1contGradnoPenumbra_version_2/train_to_diffusion" \
train.max_epochs=3000 \
train.every_n_epochs=500 \
train.optimizer.lr=3e-5 \
ckpt_path.ddpm="./experiments/official_test/checkpoint/SRD_dataset/I2130000_E12080" \
bbox_path="./dataset/SRD_sam_mask_B/train/bounding_boxes_DHAN.yaml"


