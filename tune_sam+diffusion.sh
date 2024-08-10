#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -P 12003770

#PBS -o experiments_lightning/train_test_train_gt2_1contGrad_eval_out.txt
#PBS -e experiments_lightning/train_test_train_gt2_1contGrad_eval_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

#python soft_mask_diffusion.py \
#phase=train \
#version=image_gt2_lora_sam_mse_1e-1contGrad_version_20 \
#datasets.train.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1e-1contGrad_version_20/99_to_train_diffusion/check_mask" \
#datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1e-1contGrad_version_20/99_to_test_diffusion/check_mask"

#python soft_mask_diffusion.py \
#phase=test \
#version=image_gt2_lora_sam_mse_1e-1contGrad_version_20 \
#datasets.train.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1e-1contGrad_version_20/99_to_train_diffusion/check_mask" \
#datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1e-1contGrad_version_20/99_to_test_diffusion/check_mask" \

python soft_mask_diffusion.py \
phase=test \
version=image_gt2_lora_sam_mse_1contGrad_version_21 \
datasets.train.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1contGrad_version_21/99_to_train_diffusion/check_mask" \
datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/lora_gt2_mseloss_1contGrad_version_21/99_to_test_diffusion/check_mask" \


#python soft_mask_diffusion.py \
#phase=train \
#version=ISTD+SRD_image_lora_sam_mse_1e-1contGradloss_version_1 \
#datasets.train.gt_mask_dir="./experiments_lightning/tune_sam/ISTD+SRD_lora_mseloss_1e-1contGradnoPenumbra_version_1/99_to_train_diffusion/check_mask" \
#datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/ISTD+SRD_lora_mseloss_1e-1contGradnoPenumbra_version_1/99_to_test_diffusion/check_mask" \

#
#python soft_mask_diffusion.py \
#phase=test \
#version=ISTD+SRD_image_lora_sam_mse_1e-1contGradloss_version_1 \
#datasets.train.gt_mask_dir="./experiments_lightning/tune_sam/ISTD+SRD_lora_mseloss_1e-1contGradnoPenumbra_version_1/99_to_train_diffusion/check_mask" \
#datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/ISTD+SRD_lora_mseloss_1e-1contGradnoPenumbra_version_1/99_to_test_diffusion/check_mask" \
#

# You can add additional commands here as needed