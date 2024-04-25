python soft_mask_diffusion.py phase=test samshadow_ckpt_path= '/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_3_separate_optimize/epoch_119.ckpt" \
save_result_path="/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_3_separate_optimize/119' \
test.gpu_ids=[2] && \
python soft_mask_diffusion.py phase=test samshadow_ckpt_path= '/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_3_separate_optimize/last.ckpt' \
save_result_path='/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_3_separate_optimize/last' \
test.gpu_ids=[2] && \
python soft_mask_diffusion.py phase=test samshadow_ckpt_path= '/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_4_separate_optimize/epoch_119.ckpt' \
save_result_path='/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_4_separate_optimize/119' \
test.gpu_ids=[2] && \
python soft_mask_diffusion.py phase=test samshadow_ckpt_path= '/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_4_separate_optimize/last.ckpt' \
save_result_path='/home/xinrui/projects/ShadowDiffusion/experiments_lightning/sam_head_removal_BCE+IOU/version_4_separate_optimize/last' \
test.gpu_ids=[2]