
import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio
from utils.loader import get_validation_data
import utils
import cv2

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../ISTD_Dataset/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
args = parser.parse_args()




rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
psnr_val_rgb = []
ssim_val_rgb = []
rmse_val_rgb = []
psnr_val_s = []
ssim_val_s = []
psnr_val_ns = []
ssim_val_ns = []
rmse_val_s = []
rmse_val_ns = []



bm = torch.where(mask == 0, torch.zeros_like(mask), torch.ones_like(mask))  #binarize mask
bm = np.expand_dims(bm.cpu().numpy().squeeze(), axis=2)

# calculate SSIM in gray space
gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None))
ssim_val_ns.append(ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
psnr_val_ns.append(psnr_loss(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
psnr_val_s.append(psnr_loss(rgb_restored * bm, rgb_gt * bm))

# calculate the RMSE in LAB space
rmse_temp = np.abs(cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2LAB)).mean() * 3
rmse_val_rgb.append(rmse_temp)
rmse_temp_s = np.abs(cv2.cvtColor(rgb_restored * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * bm, cv2.COLOR_RGB2LAB)).sum() / bm.sum()
rmse_temp_ns = np.abs(cv2.cvtColor(rgb_restored * (1-bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt * (1-bm),
                                                                                       cv2.COLOR_RGB2LAB)).sum() / (1-bm).sum()
rmse_val_s.append(rmse_temp_s)
rmse_val_ns.append(rmse_temp_ns)


# save_images:
utils.save_img(rgb_restored*255.0, os.path.join(args.result_dir, filenames[0]))


psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
psnr_val_s = sum(psnr_val_s)/len(test_dataset)
ssim_val_s = sum(ssim_val_s)/len(test_dataset)
psnr_val_ns = sum(psnr_val_ns)/len(test_dataset)
ssim_val_ns = sum(ssim_val_ns)/len(test_dataset)
rmse_val_rgb = sum(rmse_val_rgb) / len(test_dataset)
rmse_val_s = sum(rmse_val_s) / len(test_dataset)
rmse_val_ns = sum(rmse_val_ns) / len(test_dataset)
print("PSNR: %f, SSIM: %f, RMSE: %f " %(psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
print("SPSNR: %f, SSSIM: %f, SRMSE: %f " %(psnr_val_s, ssim_val_s, rmse_val_s))
print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " %(psnr_val_ns, ssim_val_ns, rmse_val_ns))

