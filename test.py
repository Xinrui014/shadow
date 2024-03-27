import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
import os
import numpy as np
import wandb
from scipy.io import savemat
import torch.nn.functional as F
from model.sam_adapter.sam_adapt import SAM
from model.sam_adapter.datasets import TrainDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'],
                        help='Run val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'test', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))


    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # SAM-adapter epoch and min_learning rate
    # max_epoch = 200
    # image_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/test/test_A"
    # mask_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/test/test_B"
    # datasets = TrainDataset(image_folder=image_path, mask_folder=mask_path)
    # loader = DataLoader(datasets, batch_size=1)
    # num_cuda_devices= torch.cuda.device_count()

    device = torch.device('cuda')
    adp_model = SAM(inp_size=1024, loss='iou').to(device)
    sam_checkpoint = torch.load("./experiments/sam_shadow_240325_135509/checkpoint/adp_model_epoch_50.pth")
    adp_model.load_state_dict(sam_checkpoint['model_state_dict'], strict=False)
    for name, para in adp_model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    adp_model.to(device)
    adp_model.eval()

    logger.info('Begin Model Evaluation.')
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        filename = val_data['LR_path'][0].split('/')[-1].split('.')[0]
        val_data = {key: val for key, val in val_data.items() if key != "LR_path"}

        inp = val_data['SR']
        inp = F.interpolate(inp, size=(1024, 1024), mode='bilinear', align_corners=False)
        inp = inp.to(device)
        with torch.autocast(device_type="cuda"):
            pred = torch.sigmoid(adp_model.infer(inp))
            # is 256
            val_data['mask'] = F.interpolate(pred, size=(256, 256), mode='bilinear', align_corners=False)

            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8
        mask_img = Metrics.tensor2img(pred.detach().float().cpu())

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, idx, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, filename, idx))
            savemat('sr.mat', {'img': sr_img})
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, filename, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, filename, idx))
        savemat('{}/{}_{}_hr.mat'.format(result_path, filename, idx), {'img': hr_img})
        Metrics.save_img(
            lr_img, '{}/{}_{}_lr.png'.format(result_path, filename, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, filename, idx))
        Metrics.save_img(
            mask_img, '{}/{}_{}_mask.png'.format(result_path, filename, idx)
        )

        # generation
        res = Metrics.tensor2img(visuals['SR'][-1])
        savemat('{}/{}_{}_sr_process.mat'.format(result_path, filename, idx), {'img': res})
        avg_channel = np.mean(res, axis=(0, 1))
        avg_channel_gt = np.mean(hr_img, axis=(0, 1))
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)

        avg_psnr += eval_psnr
        avg_ssim += eval_ssim
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<ID:{:3d}, filename:{}> psnr: {:.4e}, ssim: {:.4e}'.format(
            idx, filename, eval_psnr, eval_ssim))

        # if wandb_logger and opt['log_eval']:
        #     wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info('psnr: {:.4e}, ssim: {:.4e}'.format(avg_psnr, avg_ssim))

    # if wandb_logger:
    #     if opt['log_eval']:
    #         wandb_logger.log_eval_table()
    #     wandb_logger.log_metrics({
    #         'PSNR': float(avg_psnr),
    #         'SSIM': float(avg_ssim)
    #     })