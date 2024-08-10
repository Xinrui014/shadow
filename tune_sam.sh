#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=4
#PBS -l walltime=20:00:00
#PBS -P personal-xiyu004

#PBS -o experiments_lightning/ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24_out.txt
#PBS -e experiments_lightning/ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script

python finetune_sam_with_bbox.py \
version=ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24

python finetune_sam_with_bbox.py \
version=ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24 \
phase=test

python finetune_sam_with_bbox.py \
version=ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24 \
phase=test \
save_result_path="./experiments_lightning/tune_sam/ISTD+_lora_mseloss_1e-1contGradnoPenumbra_version_24/99_to_test_diffusion" \
datasets.test.dataroot="./dataset/ISTD_adjusted_C/test" \
datasets.test.gt_mask_dir="./dataset/ISTD_adjusted_C/test/test_B" \
test_bbox_path="./dataset/ISTD_adjusted_C/test/bounding_boxes_ISTD.yaml"



# You can add additional commands here as needed
