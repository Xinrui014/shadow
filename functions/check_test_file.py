import os
import re

rs_path =os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion_orig/experiments/srd_official_eval_240311_145038/results/")
mask_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion_orig/data/SRD_Dataset/srd_mask_DHAN/SRD_testmask")

gt_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion_orig/data/SRD_Dataset/test/test_C")
def extract_base_name(filename):
    return re.sub(r'_\d+_sr_process$', '', filename)

# List the base filenames in rs_path directory that end with 'sr_process.png'
rs_files ={extract_base_name(os.path.splitext(f)[0]) for f in os.listdir(rs_path) if f.endswith('sr_process.png')}
mask_files = {os.path.splitext(f)[0].split('.')[0] for f in os.listdir(mask_path)}
gt_files = {os.path.splitext(f)[0].replace("_free", "") for f in os.listdir(gt_path)}


# Find mismatches between the sets
rs_mask = rs_files - mask_files
rs_gt = rs_files - gt_files
mask_rs = mask_files - rs_files

# Print the mismatches

print("Files in rs_path that are not in mask_path: ", rs_mask)
# print("Files in rs_path that are not in gt_path: ", rs_gt)
print("Files in mask_path that are not in gt_path: ", mask_rs)

