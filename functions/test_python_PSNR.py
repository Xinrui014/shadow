# import os
# from PIL import Image
# import torchvision
# import numpy as np
# rs_path =os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/experiments/srd_official_eval_240311_145038/results/IMG_1_5453_1_sr_process.png")
# # mask_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/data/SRD_Dataset/srd_mask_DHAN/SRD_testmask")
#
# gt_path = os.path.expanduser("~/Documents/projects/sam_shadow_removal/ShadowDiffusion/data/SRD_Dataset/test/test_C/IMG_1_5453_free.jpg")
# preresize = torchvision.transforms.Resize([256, 256])
# sr = Image.open(rs_path).convert("RGB")
# sr = preresize(sr)
# sr = np.array(sr)
# hr = Image.open(gt_path).convert("RGB")
# hr = preresize(hr)
# hr = np.array(hr)
#
# sr


import cv2
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8)/0.15) + 2.0)
def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be seperable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))

import os
path = '/home/user/Documents/projects/sam_shadow_removal/ShadowDiffusion/dataset/SRD_sam_mask_B/train/'
save = '/home/user/Documents/projects/sam_shadow_removal/ShadowDiffusion/dataset/SRD_sam_mask_B/train/division_filter_normalized_results'

if not os.path.exists(save):
    os.mkdir(save)

image1_path = path + 'train_A'
image2_path = path + 'train_C'

for image1_name in os.listdir(image1_path):
    image2_name = image1_name.replace('.jpg', '_no_shadow.jpg')
    image1 = cv2.imread(os.path.join(image1_path, image1_name))
    image2 = cv2.imread(os.path.join(image2_path, image2_name))
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image1_ycbcr = cv2.cvtColor(image1_rgb, cv2.COLOR_RGB2YCrCb)
    image2_ycbcr = cv2.cvtColor(image2_rgb, cv2.COLOR_RGB2YCrCb)
    y_channel1 = image1_ycbcr[:, :, 0]
    y_channel2 = image2_ycbcr[:, :, 0]
    # diff_y = cv2.absdiff(y_channel1, y_channel2)
    # diff = get_gaussian_blur(diff_y, ksize=0, sigma=5)
    # diff = np.log(diff)
    # threshold = 0.98
    # diff[diff < threshold] = 0

    division = y_channel2 / (y_channel1 + 1e-9)# Add a small constant to avoid division by zero
    # division = cv2.GaussianBlur(division, (3,3), 0)
    division = sigmoid(division)
    division = get_gaussian_blur(division, ksize=0, sigma=3)
    division = (division - np.min(division)) / (np.max(division) - np.min(division))

    threshold = 0.16
    division[division < threshold] = 0
    save_path = os.path.join(save, image1_name)
    division = (division * 255).astype(np.uint8)
    cv2.imwrite(save_path, division)

