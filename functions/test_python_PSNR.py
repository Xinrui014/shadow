import os
from PIL import Image
import torchvision
import numpy as np
rs_path =os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/experiments/srd_official_eval_240311_145038/results/IMG_1_5453_1_sr_process.png")
# mask_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/data/SRD_Dataset/srd_mask_DHAN/SRD_testmask")

gt_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/data/SRD_Dataset/test/test_C/IMG_1_5453_free.jpg")
preresize = torchvision.transforms.Resize([256, 256])
sr = Image.open(rs_path).convert("RGB")
sr = preresize(sr)
sr = np.array(sr)
hr = Image.open(gt_path).convert("RGB")
hr = preresize(hr)
hr = np.array(hr)

sr




