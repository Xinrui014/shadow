import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import random
import numpy as np
from tqdm import tqdm
from model.sam_adapter.sam_adapt import SAM
from model.sam_adapter.datasets import TrainDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/shadow.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))

    # Set Seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    # Init and load SAM_adapter
    image_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/train/train_A"
    mask_path = "/home/xinrui/projects/SAM-Adapter/data/ISTD_Dataset/train/train_B"
    datasets = TrainDataset(image_folder=image_path, mask_folder=mask_path)
    loader = DataLoader(datasets, batch_size=1)
    num_cuda_devices= torch.cuda.device_count()

    device = torch.device('cuda')
    adp_model = SAM(inp_size=1024, loss='iou').to(device)
    # print(model)
    sam_checkpoint = torch.load("./experiments/official_test/SAM_adapter_ckpt/model_epoch_best.pth")
    adp_model.load_state_dict(sam_checkpoint, strict=False)
    for name, para in adp_model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
    # adp_model.to(device)
    adp_model.train()

    # Initialize the AdamW optimizer for adp_model
    lr = 0.0002
    weight_decay = 0.0  # Set the weight decay value as needed
    adp_model_optimizer = optim.AdamW(adp_model.parameters(), lr=lr, weight_decay=weight_decay)

    # Set the optimizer in adp_model
    adp_model.optimizer = adp_model_optimizer

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    # print name paramter in diffusion and the sam model
    for name, para in adp_model.named_parameters():
        # print layer name and if it requires grad and if it is a leaf tensor
        # check if the tensor is not leaf but requires gradient
        if para.requires_grad and not para.is_leaf:
            print(name, para.requires_grad, para.is_leaf)
    for name, para in diffusion.netG.named_parameters():
        if para.requires_grad and not para.is_leaf:
            print(name, para.requires_grad, para.is_leaf)
    if opt['phase'] == 'train':
        n_epoch = 20000
        with tqdm(total=n_epoch, unit='epoch') as pbar:
            while current_epoch < n_epoch:
                current_epoch += 1
                for _, train_data in enumerate(train_loader):
                    current_step += 1
                    if current_step > n_iter:
                        break

                    # add SAM-adapter training mode


                    # inp = loader['inp']
                    # gt = loader['gt']
                    inp = train_data['SR']
                    gt = train_data['mask']
                    inp = F.interpolate(inp, size=(1024, 1024), mode='bilinear', align_corners=False)
                    gt = F.interpolate(gt, size=(1024, 1024), mode='bilinear', align_corners=False)

                    diffusion.netG.zero_grad()
                    adp_model.zero_grad()

                    with torch.autocast(device_type="cuda"):
                        adp_model.set_input(inp, gt)
                        pred_mask = adp_model.forward()
                        train_data['mask'] = F.interpolate(pred_mask, size=(160,160), mode='bilinear', align_corners=False)
                        diffusion.feed_data(train_data)

                        # Combine the two losses
                        l_pix = diffusion.netG(diffusion.data)
                        b, c, h, w = diffusion.data['HR'].shape
                        l_pix = l_pix.sum() / int(b * c * h * w)
                        # l_total = l_pix + adp_model.compute_loss()  # Assuming adp_model has a compute_loss() method

                        # Perform a single backward pass
                        l_pix.backward()
                    diffusion_gard = diffusion.netG.named_parameters()
                    SAM_gard = adp_model.named_parameters()

                    # Update the optimizers
                    diffusion.optG.step()
                    # adp_model.optimizer.step()
                    #
                    # print("Diffusion model gradients:")
                    # for name, param in diffusion.netG.named_parameters():
                    #     print(f"{name}: {param.grad}")
                    #     break
                    #
                    # print("SAM-adapter model gradients:")
                    # for name, param in adp_model.named_parameters():
                    #     print(f"{name}: {param.grad}")
                    #     break

                    diffusion.ema_helper.update(diffusion.netG)

                    # Set log
                    diffusion.log_dict['l_pix'] = l_pix.item()

                    # log
                    if current_step % opt['train']['print_freq'] == 0:
                        logs = diffusion.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                            current_epoch, current_step)
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                        logger.info(message)

                    if current_step % opt['train']['save_checkpoint_freq'] == 0:
                        logger.info('Saving models and training states.')
                        diffusion.save_network(current_epoch, current_step)

                pbar.update(1)
            logger.info('End of training.')