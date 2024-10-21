import argparse
from PIL import Image
import numpy as np
import torch
import os
import cv2
import glob
from natsort import natsorted

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--result_path', type=str,
                        default='/home/xinrui/projects/ShadowDiffusion/experiments_lightning/joint_tune_sam_diffusion/mse+1e-1contGradnoPenumbra/399_old_2')
    parser.add_argument('--mask_path', type=str,
                        default='/home/xinrui/projects/ShadowDiffusion/dataset/SRD_DHAN_mask_old/test/test_B')
    args = parser.parse_args()

    real_names = list(glob.glob('{}/*_hr.png'.format(args.result_path)))
    fake_names = list(glob.glob('{}/*_sr.png'.format(args.result_path)))
    mask_names = os.listdir(args.mask_path)

    real_names = natsorted(real_names)
    fake_names = natsorted(fake_names)
    mask_names = natsorted(mask_names)

    if len(real_names) == len(fake_names) == len(mask_names):
        print("All three lists have equal length.")
    else:
        print("Error: The lists have different lengths.")

    psnr_val_rgb = []
    ssim_val_rgb = []
    rmse_val_rgb = []
    psnr_val_s = []
    ssim_val_s = []
    psnr_val_ns = []
    ssim_val_ns = []
    rmse_val_s = []
    rmse_val_ns = []


    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    for rname, fname in zip(real_names, fake_names):
        idx += 1
        ridx = os.path.splitext(os.path.basename(rname))[0].rsplit("_hr", 1)[0]
        fidx = os.path.splitext(os.path.basename(fname))[0].rsplit("_sr", 1)[0]
        assert ridx == fidx, f'Image indices do not match: ridx:{ridx}, fidx:{fidx}'

        mname = os.path.join(args.mask_path, fidx+".jpg")

        rgb_gt = np.array(Image.open(rname))
        rgb_restored = np.array(Image.open(fname))
        mask = Image.open(mname)
        mask = mask.resize((rgb_gt.shape[1], rgb_gt.shape[0]), Image.BILINEAR)
        mask = np.array(mask)

        bm = np.where(mask <= 10, np.zeros_like(mask), np.ones_like(mask))  # binarize mask
        bm = np.expand_dims(bm.squeeze(), axis=2)

        # calculate SSIM in gray space
        gray_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2GRAY)
        gray_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_RGB2GRAY)
        ssim_val_rgb.append(ssim_loss(gray_restored, gray_gt, channel_axis=None))
        ssim_val_ns.append(
            ssim_loss(gray_restored * (1 - bm.squeeze()), gray_gt * (1 - bm.squeeze()), channel_axis=None))
        ssim_val_s.append(ssim_loss(gray_restored * bm.squeeze(), gray_gt * bm.squeeze(), channel_axis=None))

        psnr_val_rgb.append(psnr_loss(rgb_restored, rgb_gt))
        psnr_val_ns.append(psnr_loss(rgb_restored * (1 - bm), rgb_gt * (1 - bm)))
        psnr_val_s.append(psnr_loss(rgb_restored * bm, rgb_gt * bm))
        rgb_restored_ = (rgb_restored/255.).astype(np.float32)
        rgb_gt_ = (rgb_gt/255.).astype(np.float32)

        rgb_gt_ = np.clip(rgb_gt_, 0, 1)
        rgb_restored_ = np.clip(rgb_restored_, 0, 1)

        # calculate the RMSE in LAB space
        rmse_temp = np.abs(
            cv2.cvtColor(rgb_restored_, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt_, cv2.COLOR_RGB2LAB)).mean() * 3
        rmse_val_rgb.append(rmse_temp)
        rmse_temp_s = np.abs(cv2.cvtColor(rgb_restored_ * bm, cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt_ * bm,
                                                                                               cv2.COLOR_RGB2LAB)).sum() / bm.sum()
        rmse_temp_ns = np.abs(cv2.cvtColor(rgb_restored_ * (1 - bm), cv2.COLOR_RGB2LAB) - cv2.cvtColor(rgb_gt_ * (1 - bm),
                                                                                                      cv2.COLOR_RGB2LAB)).sum() / (
                                   1 - bm).sum()
        rmse_val_s.append(rmse_temp_s)
        rmse_val_ns.append(rmse_temp_ns)

    psnr_val_rgb = sum(psnr_val_rgb) / idx
    ssim_val_rgb = sum(ssim_val_rgb) / idx
    psnr_val_s = sum(psnr_val_s) / idx
    ssim_val_s = sum(ssim_val_s) / idx
    psnr_val_ns = sum(psnr_val_ns) / idx
    ssim_val_ns = sum(ssim_val_ns) / idx
    rmse_val_rgb = sum(rmse_val_rgb) / idx
    rmse_val_s = sum(rmse_val_s) / idx
    rmse_val_ns = sum(rmse_val_ns) / idx
    print("PSNR: %f, SSIM: %f, RMSE: %f " % (psnr_val_rgb, ssim_val_rgb, rmse_val_rgb))
    print("SPSNR: %f, SSSIM: %f, SRMSE: %f " % (psnr_val_s, ssim_val_s, rmse_val_s))
    print("NSPSNR: %f, NSSSIM: %f, NSRMSE: %f " % (psnr_val_ns, ssim_val_ns, rmse_val_ns))
