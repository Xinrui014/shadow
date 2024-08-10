import os
import cv2
import numpy as np
import torch
from PIL import Image
import hydra
import torch

import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.sam_adapter.sam_adapt import SAM
import core.metrics as Metrics
from data.LRHR_dataset import LRHRDataset, TrainDataset, TestDataset
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from omegaconf import DictConfig
import model.networks as networks
from model.sam_adapter.iou_loss import IOU
from lightning.pytorch.loggers import TensorBoardLogger


def apply_low_pass_filter(mask: object, kernel_size: object = 3) -> object:
    # Create a Gaussian kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask.device)

    # Apply the low-pass filter using convolution
    filtered_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    return filtered_mask


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class SamshadowDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = args.datasets.train.num_workers
        self.shuffle = args.datasets.train.use_shuffle

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_data = TrainDataset(self.args.datasets.train.dataroot, 1024)
        elif stage == "test":
            self.test_data = TestDataset(self.args.datasets.test.dataroot, 1024)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data,
                                batch_size=self.args.datasets.train.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        return dataloader


# define a LightningModule, in which I have shadowDiffusion and SAM_adapter two models
def laplacian_magnitude_loss(soft_mask):
    soft_mask = torch.sigmoid(soft_mask)
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=soft_mask.dtype, device=soft_mask.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
    laplacian = F.conv2d(soft_mask, laplacian_kernel, padding=1)
    magnitude = torch.abs(laplacian)
    mean_magnitude = torch.mean(magnitude)
    return mean_magnitude


def gradient_orientation_map(input_image):
    sobelx = F.conv2d(input_image, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    sobely = F.conv2d(input_image, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                                dtype=input_image.dtype,
                                                device=input_image.device), padding=1)
    gradient_orientation = torch.atan2(sobely, sobelx)
    return gradient_orientation


def gradient_orientation_loss(soft_mask, shadow_input):
    penumbra_area = torch.sigmoid(soft_mask)
    # set a threshold
    threshold_low = 0.1
    threshold_high = 0.995
    penumbra_area = torch.where((penumbra_area >= threshold_low) & (penumbra_area <= threshold_high), 1.0, 0.0)
    # convert the shadow input_image to gray
    shadow_input = torch.mean(shadow_input, dim=1, keepdim=True)
    # upscale shadow_input to 256x256
    shadow_input = F.interpolate(shadow_input, (penumbra_area.shape[2], penumbra_area.shape[3]), mode='bilinear', align_corners=False)
    # Apply low-pass filter to the shadow input_image
    shadow_input = apply_low_pass_filter(shadow_input)
    # Compute the gradient of the shadow input_image penumbra area
    shadow_input_penumbra = shadow_input*penumbra_area
    gradient_orientation_shadow_input = gradient_orientation_map(shadow_input_penumbra)
    gradient_orientation_soft_mask = gradient_orientation_map(soft_mask * penumbra_area)
    # Compute the cosine similarity between the gradient orientations
    # using "+" because the two maps has different orientation
    cosine_similarity = torch.cos(gradient_orientation_shadow_input + gradient_orientation_soft_mask)
    # Compute the gradient orientation loss
    gradient_loss = 1 - cosine_similarity
    gradient_loss = torch.mean(gradient_loss)
    return gradient_loss


def soft_mask_loss(soft_mask, shadow_input):
    laplacian_loss = laplacian_magnitude_loss(soft_mask)
    gradient_loss = gradient_orientation_loss(soft_mask, shadow_input)
    # soft_hard_mask_consistency = soft_hard_mask_consistency(soft_mask, hard_mask)
    return laplacian_loss, gradient_loss


class sam_shadow(L.LightningModule):
    def __init__(self, sam_adapter, shadowdiffusion, args, path=None, steps_per_epoch=168):
        super(sam_shadow, self).__init__()
        self.args = args
        self.steps_per_epoch = int(steps_per_epoch)
        self.diffusion = shadowdiffusion(args)
        self.optimizer_param = args.train.optimizer
        self.diffusion.set_loss()
        self.sam = sam_adapter(args.sam.input_size, args.sam.loss)
        # self.sam = self.sam.to(self.device)
        self.criterionBCE = torch.nn.BCEWithLogitsLoss()
        self.criterionIOU = IOU()
        if args.phase == 'train':
            self.diffusion.set_new_noise_schedule(args['model']['beta_schedule']['train'])

        if path is not None:
            self.load_pretrained_models(path)
            if args.phase == 'train':
                # optimizer
                opt_path = '{}_opt.pth'.format(path.ddpm)
                opt = torch.load(opt_path, map_location=self.device)
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

            # transfer output_hypernetworks_mlps parameter to soft_mask_decoder_adapter
            # for i in range(self.sam.mask_decoder.num_mask_tokens):
            #     self.sam.mask_decoder.soft_mask_decoder_adapter[i].load_state_dict(
            #         self.sam.mask_decoder.output_hypernetworks_mlps[i].state_dict()
            #     )

        self.save_hyperparameters()
        self.save_model_name = args.name
        self.PSNR = []
        self.SSIM = []


        # freeze sam
        for param in self.sam.parameters():
            param.requires_grad = False

        # open the second head to predict soft head
        # for name, param in self.sam.named_parameters():
        #     if "soft_mask_decoder_adapter" in name:
        #         param.requires_grad_(True)

    def training_step(self, train_data, batch_idx):

        # for name, param in self.sam.named_parameters():
        #     if param.requires_grad:
        #         print(name)

        sam_input = train_data['sam_SR']
        sam_mask = train_data['sam_mask']
        sam_pred_mask_, sam_pred_soft_mask = self.sam(sam_input)
        sam_pred_mask = torch.sigmoid(sam_pred_mask_)
        value = sam_pred_mask.squeeze(0).squeeze(0).detach().cpu().numpy()
        # value = (value * 255).astype(np.uint8)
        # # Save the normalized value as a single-channel image
        # cv2.imwrite('cv2_sam_pred_mask.png', value)
        # save_image(sam_pred_mask, 'sam_pred_mask.png')
        # here has a BCE+iou loss
        sam_pred_mask_for_loss = F.interpolate(sam_pred_mask, (1024, 1024), mode='bilinear', align_corners=False)
        self.hard_mask_loss = self.criterionBCE(sam_pred_mask_for_loss, sam_mask)
        self.hard_mask_loss += _iou_loss(sam_pred_mask_for_loss, sam_mask)
        self.log('sam_mask_loss', self.hard_mask_loss.item(), on_step=True)

        # soft mask loss
        # debug using hard mask
        # laplacian_loss_, gradient_loss_ = soft_mask_loss(sam_pred_soft_mask, train_data['HR'])
        # laplacian_loss, gradient_ori_loss = soft_mask_loss(sam_pred_mask_, train_data['HR'])

        # self.soft_mask_loss = laplacian_loss + 0.1 * gradient_ori_loss
        # self.log('laplacian_loss', laplacian_loss.item(), on_step=True)
        # self.log('gradient_loss', gradient_ori_loss.item(), on_step=True)
        # self.log('soft_mask_loss', self.soft_mask_loss.item(), on_step=True)

        train_data['mask'] = F.interpolate(sam_pred_mask, (160, 160), mode='bilinear', align_corners=False)
        l_pix = self.diffusion(train_data)
        b, c, h, w = train_data['HR'].shape
        diffusion_loss = l_pix.sum() / int(b * c * h * w)
        self.log('l_pix_loss', diffusion_loss.item(), on_step=True)

        return diffusion_loss

    def test_step(self, test_data):
        filename = test_data['LR_path'][0].split('/')[-1].split('.')[0]
        test_data = {key: val for key, val in test_data.items() if key != "LR_path"}
        # Sam adapter input 1024x1024
        sam_input = test_data['sam_SR']
        sam_pred_mask, sam_pred_soft_mask = self.sam(sam_input)
        sam_pred_mask = torch.sigmoid(sam_pred_mask)
        # diffusion input 256x256
        test_data['mask'] = F.interpolate(sam_pred_mask, (256, 256), mode='bilinear', align_corners=False)
        shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(test_data['SR'], test_data['mask'], continous=False)

        # calculate PSNR and SSIM here
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(test_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        self.PSNR.append(eval_psnr)
        self.SSIM.append(eval_ssim)

        self.log(f'{filename}_PSNR', eval_psnr, prog_bar=True)
        self.log(f'{filename}_SSIM', eval_ssim)
        save_path = "/home/xinrui/projects/ShadowDiffusion_orig/experiments_lightning/fix_sam/fix_sam_version_1/last/"
        # save SR and mask_pred
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        sr_path = os.path.join(save_path, f'{filename}_sr.png')
        sr_img = Image.fromarray(res)
        sr_img.save(sr_path)
        hr_path = os.path.join(save_path, f'{filename}_hr.png')
        hr_img = Image.fromarray(hr_img)
        hr_img.save(hr_path)
        mask = diffusion_mask_pred.squeeze().cpu().numpy() * 255
        mask_path = os.path.join(save_path, f'{filename}_mask.png')
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(mask_path)

    def on_test_end(self):
        avg_psnr = np.mean(self.PSNR)
        avg_ssim = np.mean(self.SSIM)
        print(f'Average PSNR: {avg_psnr}')
        print(f'Average SSIM: {avg_ssim}')


    def load_pretrained_models(self, path):
        gen_path = '{}_gen.pth'.format(path.ddpm)
        self.diffusion.load_state_dict(torch.load(gen_path), strict=False)

        sam_state_dict = torch.load(path.sam, map_location=self.device)
        self.sam.load_state_dict(sam_state_dict, strict=False)

    def configure_optimizers(self):
        param_groups = [
            {'params': self.sam.parameters(), 'lr': 0.0002},
            {'params': self.diffusion.parameters(), 'lr': 1e-05}
        ]
        optimizer = torch.optim.Adam(param_groups)
        warm_up_steps = self.args.warmup_epochs * self.steps_per_epoch
        max_step = self.args.max_epochs * self.steps_per_epoch
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warm_up_steps, max_epochs=max_step)
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'step',  # The unit of the scheduler's step size
            }
        }
        return optim_dict


# using SAM mask train diffusion
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.name + "_" + args.version + "_logger")
    data_module = SamshadowDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    # model = diffusion(networks.define_G, args, ckpt_path)
    model = sam_shadow(SAM, networks.define_G, args, ckpt_path)

    # load training parameters
    save_model_name = args.name
    max_epochs = args.train.max_epochs
    save_every_n_epochs = args.train.every_n_epochs
    log_every_n_steps = 1
    log_every_n_epochs = 1

    save_path = f"./experiments_lightning/{save_model_name}/{args.name}_{args.version}"
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=save_every_n_epochs,  # Save every 20 epochs
        save_last=True,  # Save the last model as well
    )
    trainer = L.Trainer(
        accelerator='gpu',
        precision=16,
        devices=args.train.gpu_ids,
        max_epochs=max_epochs,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
        # strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name=args.ckpt_name)
    data_module = SamshadowDataModule(args)
    data_module.setup("test")
    model = sam_shadow.load_from_checkpoint(args.samshadow_ckpt_path)
    tester = L.Trainer(
        accelerator='gpu',
        devices=args.test.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        logger=logger
    )
    test_result = tester.test(model, data_module)
    PSNR_SSIM_list_with_name = test_result[0]
    with open(f'./experiments_lightning/{args.name}/{args.ckpt_name}_PSNR_SSIM_list.log',
              'w') as file:
        for key, value in PSNR_SSIM_list_with_name.items():
            file.write(f"{key}: {value}\n")


@hydra.main(config_path="./config", config_name='sam_shadow_jointTraining_SRD', version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(1234)
    args.version = args.get('version', 'none_version')

    if args.phase == 'train':
        train(args)
    elif args.phase == 'test':
        sample(args)


if __name__ == "__main__":
    main()
