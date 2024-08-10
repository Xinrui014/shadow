#!/bin/bash
#PBS -l select=1:ncpus=32:mem=128gb:ngpus=4
#PBS -l walltime=10:00:00
#PBS -P 12003770

#PBS -o experiments_lightning/tune_sam_wobbox_version_9.1_out.txt
#PBS -e experiments_lightning/tune_sam_wobbox_version_9.1_error.txt

# Load the module for Conda if necessary
# module load anaconda3/2020.02 # Uncomment and modify if your system uses modules

# Activate the Conda environment
conda init bash
source ~/.bashrc
conda activate arldm  # or use 'conda activate my_env' depending on your Conda setup
cd scratch/ShadowDiffusion/

# Run a Python script
#python finetune_sam_with_bbox.py version=lora_mseloss_1e-3orientation_version_17
python finetune_sam_with_bbox.py \
version=lora_mseloss_1e-1contGradnoPenumbra_wobbox_version_9.1 \
phase=train \
samshadow_ckpt_path="./" \
save_result_path="./" \
datasets.train.dataroot="./dataset/SRD_sam_mask_B/train" \
datasets.train.gt_mask_dir="./dataset/SRD_sam_mask_B/train/division_filter_results" \
bbox_path="./dataset/SRD_sam_mask_B/train/bounding_boxes_DHAN.yaml" \
datasets.test.dataroot="./dataset/SRD_sam_mask_B/test" \
datasets.test.gt_mask_dir="./dataset/SRD_sam_mask_B/test/division_filter_results" \
test_bbox_path="./dataset/SRD_sam_mask_B/test/bounding_boxes_DHAN.yaml"


# You can add additional commands here as needed
