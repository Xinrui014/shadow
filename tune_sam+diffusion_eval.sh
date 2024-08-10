#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=1
#PBS -l walltime=1:00:00
#PBS -P 12003770

#PBS -o experiments_lightning/version_20_eval_out.txt
#PBS -e experiments_lightning/version_20_eval_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script
# python finetune_sam_with_bbox.py
python soft_mask_diffusion.py \
version=image_lora_sam_mse_1e-4orientation_version_20 \
phase=test \
save_result_path="./experiments_lightning/tune_sam+diffusion/image_lora_sam_mse_1e-4orientation_version_20/499" \
datasets.test.dataroot="./dataset/SRD_sam_mask_B/test" \
datasets.test.gt_mask_dir="./experiments_lightning/tune_sam/lora_mseloss_1e-4orientation_version_19/99_to_test_diffusion/check_mask" \
test_bbox_path="./dataset/SRD_sam_mask_B/test/bounding_boxes_DHAN.yaml"



# You can add additional commands here as needed