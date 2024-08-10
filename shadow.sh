#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=4
#PBS -l walltime=10:00:00
#PBS -P personal-xiyu004





#PBS -o experiments_lightning/image_gt_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD+_version_24.2_out.txt
#PBS -e experiments_lightning/image_gt_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD+_version_24.2_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/



#python finetune_sam_with_bbox.py \
#phase=train

#python finetune_sam_with_bbox.py \
#phase=test
#
#python finetune_sam_with_bbox.py \
#phase=test \
#save_result_path="./experiments_lightning/tune_sam/ISTD_lora_mseloss_1e-1contGradnoPenumbra_version_23/99_to_test_diffusion" \
#datasets.test.dataroot="./dataset/ISTD_adjusted_C/test" \
#datasets.test.gt_mask_dir="./dataset/ISTD_adjusted_C/test/division_filter_results" \
#test_bbox_path="./dataset/ISTD_adjusted_C/test/bounding_boxes_ISTD.yaml"

#python soft_mask_diffusion.py \
#version=image_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD_version_24 \
#ckpt_path.ddpm="./experiments/official_test/checkpoint/ISTD_dataset/I260000_E3096" \
#phase=train \
#train.max_epochs=2000 \
#train.optimizer.lr=3e-5
#
#python soft_mask_diffusion.py \
#version=image_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD_version_24 \
#phase=test


python soft_mask_diffusion.py \
version=image_gt_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD+_version_24.2 \
train.optimizer.lr=3e-5 \
phase=train

python soft_mask_diffusion.py \
version=image_gt_ISTD_lora_mseloss_1e-1contGradnoPenumbra_fromISTD+_version_24.2 \
phase=test

