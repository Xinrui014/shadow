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


class SamShadowDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = args.datasets.train.num_workers
        self.shuffle = args.datasets.train.use_shuffle

    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_data = TrainDataset(self.args.datasets.train.dataroot, 1024)
            self.val_data = TestDataset(self.args.datasets.test.dataroot, 1024, data_len=3)
        elif stage == "test":
            self.test_data = TestDataset(self.args.datasets.test.dataroot, 1024)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data,
                                batch_size=self.args.datasets.train.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.val_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        return dataloader

    def predict_dataloader(self):
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
    soft_mask = torch.sigmoid(soft_mask)
    penumbra_area = soft_mask.detach()
    # set a threshold
    threshold_low = 0.01
    threshold_high = 0.995
    penumbra_area = torch.where((penumbra_area >= threshold_low) & (penumbra_area <= threshold_high), 1.0, 0.0)
    shadow_input = torch.mean(shadow_input, dim=1, keepdim=True)
    shadow_input = F.interpolate(shadow_input, (penumbra_area.shape[2], penumbra_area.shape[3]), mode='bilinear', align_corners=False)
    shadow_input = apply_low_pass_filter(shadow_input)

    gradient_orientation_shadow_input = gradient_orientation_map(shadow_input*penumbra_area)
    gradient_orientation_soft_mask = gradient_orientation_map(soft_mask * penumbra_area)
    # using "+" because the two maps has different orientation
    cosine_similarity = torch.cos(gradient_orientation_shadow_input + gradient_orientation_soft_mask)
    gradient_loss = 1 - cosine_similarity
    gradient_loss = torch.mean(gradient_loss)
    return gradient_loss


def soft_mask_loss_metric(soft_mask, shadow_input):
    laplacian_loss = laplacian_magnitude_loss(soft_mask)
    gradient_loss = gradient_orientation_loss(soft_mask, shadow_input)
    # soft_hard_mask_consistency = soft_hard_mask_consistency(soft_mask, hard_mask)
    return laplacian_loss, gradient_loss


class SamShadow(L.LightningModule):
    def __init__(self, sam_adapter, shadow_diffusion, args, path=None, steps_per_epoch=168):
        super(SamShadow, self).__init__()
        self.args = args
        self.steps_per_epoch = int(steps_per_epoch)
        self.diffusion = shadow_diffusion(args)
        self.optimizer_param = args.train.optimizer
        self.diffusion.set_loss()
        self.sam = sam_adapter(args.sam.input_size, args.sam.loss)
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
            for i in range(self.sam.mask_decoder.num_mask_tokens):
                self.sam.mask_decoder.soft_mask_decoder_adapter[i].load_state_dict(
                    self.sam.mask_decoder.output_hypernetworks_mlps[i].state_dict()
                )

        self.save_hyperparameters()
        self.save_model_name = args.name
        self.PSNR_SSIM_list_with_name = []
        self.PSNR = []
        self.SSIM = []
        # freeze sam
        for param in self.sam.parameters():
            param.requires_grad = False

        # open the second head to predict soft head
        for name, param in self.sam.named_parameters():
            if "soft_mask_decoder_adapter" in name:
                param.requires_grad = True

    def print_param_values(self):
        print("SAM parameters:")
        for name, param in self.sam.mask_decoder.soft_mask_decoder_adapter.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}")
            print(f"{name}: {param.data.mean().item()}")
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item()}")
            else:
                print(f"{name}: NO gradient")
            break

        print("Diffusion parameters:")
        for name, param in self.diffusion.named_parameters():
            print(f"{name}: {param.data.mean().item()}")
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item()}")
            else:
                print(f"{name}: NO gradient")
            break

    def training_step(self, train_data, batch_idx):
        # train_data = {key: val for key, val in train_data.items() if key != "filename"}
        sam_input = train_data['sam_SR']
        # sam_mask = train_data['sam_mask']
        sam_pred_mask_, sam_pred_soft_mask = self.sam(sam_input)
        # sam_pred_mask = torch.sigmoid(sam_pred_mask_)
        # # here has a BCE+iou loss
        # sam_pred_mask_for_loss = F.interpolate(sam_pred_mask, (1024, 1024), mode='bilinear', align_corners=False)
        self.hard_mask_loss = self.criterionBCE(sam_pred_mask_for_loss, sam_mask)
        # self.hard_mask_loss += _iou_loss(sam_pred_mask_for_loss, sam_mask)
        # self.log('sam_mask_loss', self.hard_mask_loss.item(), on_step=True)

        laplacian_loss, gradient_ori_loss = soft_mask_loss_metric(sam_pred_soft_mask, train_data['HR'])

        soft_mask_loss = 10 * (5 * laplacian_loss + gradient_ori_loss)
        self.log('laplacian_loss', laplacian_loss.item(), on_step=True)
        self.log('gradient_loss', gradient_ori_loss.item(), on_step=True)
        self.log('soft_mask_loss', soft_mask_loss.item(), on_step=True)

        sam_pred_soft_mask = torch.sigmoid(sam_pred_soft_mask)
        train_data['mask'] = F.interpolate(sam_pred_soft_mask, (160, 160), mode='bilinear', align_corners=False)
        l_pix = self.diffusion(train_data)
        b, c, h, w = train_data['HR'].shape
        diffusion_loss = l_pix.sum() / int(b * c * h * w)
        self.log('l_pix_loss', diffusion_loss.item(), on_step=True)


        # print("After training step:")
        # self.print_param_values()

        if self.global_step % 100 == 0:
            filename = train_data['filename'][0]
            save_path = f'./experiments_lightning/{self.args.name}/{self.args.version}/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_image(sam_pred_soft_mask[0], os.path.join(save_path, filename+'_soft_mask.png'))

        return {'loss': diffusion_loss, 'soft_mask': sam_pred_soft_mask}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step % 100 == 0:
            filename = batch['filename'][0]
            save_path = f'./experiments_lightning/{self.args.name}/{self.args.version}/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_image(outputs['soft_mask'][0], os.path.join(save_path, filename+'_soft_mask.png'))


    def validation_step(self, val_data, batch_idx):
        val_data = {key: val for key, val in val_data.items() if key != "filename"}
        sam_input = val_data['sam_SR']
        sam_pred_mask_, sam_pred_soft_mask = self.sam(sam_input)
        sam_pred_soft_mask = torch.sigmoid(sam_pred_soft_mask)
        val_data['mask'] = F.interpolate(sam_pred_soft_mask, (256, 256), mode='bilinear', align_corners=False)
        shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(val_data['SR'], val_data['mask'],
                                                                                 continous=False)
        self.logger.experiment.add_image('Val/Shadow_Removal', shadow_removal_sr[0], self.global_step)

        # Log the diffusion mask prediction
        self.logger.experiment.add_image('Val/Diffusion_Mask', diffusion_mask_pred[0], self.global_step)

    def predict_step(self, test_data):
        # test_data = {key: val for key, val in test_data.items() if key != 'filename'}
        # Sam adapter input 1024x1024
        sam_input = test_data['sam_SR']
        sam_pred_mask, sam_pred_soft_mask = self.sam(sam_input)
        # diffusion input 256x256
        sam_pred_soft_mask = torch.sigmoid(sam_pred_soft_mask)
        test_data['mask'] = F.interpolate(sam_pred_soft_mask, (256, 256), mode='bilinear', align_corners=False)
        shadow_removal_sr, diffusion_mask_pred = self.diffusion.super_resolution(test_data['SR'], test_data['mask'], continous=False)
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(test_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        filename = test_data['filename']
        return eval_psnr, eval_ssim, filename, sam_pred_soft_mask, shadow_removal_sr, diffusion_mask_pred, test_data['HR']



    def load_pretrained_models(self, path):
        gen_path = '{}_gen.pth'.format(path.ddpm)
        self.diffusion.load_state_dict(torch.load(gen_path), strict=False)

        sam_state_dict = torch.load(path.sam, map_location=self.device)
        self.sam.load_state_dict(sam_state_dict, strict=False)

    def configure_optimizers(self):
        # check if the adatper parameter is in self.sam
        # for name, param in self.sam.named_parameters():
        #     if "soft_mask_decoder_adapter" in name:
        #         print("adapter is in optimizer")


        param_groups = [
            {'params': self.sam.parameters(), 'lr': 0.0002},
            {'params': self.diffusion.parameters(), 'lr': 1e-05}
        ]
        optimizer = torch.optim.Adam(param_groups)
        # warm_up_steps = self.args.warmup_epochs * self.steps_per_epoch
        # max_step = self.args.max_epochs * self.steps_per_epoch
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=warm_up_steps, max_epochs=max_step)
        # optim_dict = {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,  # The LR scheduler instance (required)
        #         'interval': 'step',  # The unit of the scheduler's step size
        #     }
        # }
        return optimizer

# using SAM mask train diffusion
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.name + "_" + args.version + "_logger")
    data_module = SamShadowDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    # model = diffusion(networks.define_G, args, ckpt_path)
    model = SamShadow(SAM, networks.define_G, args, ckpt_path)

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
        gradient_clip_val=0.5,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        check_val_every_n_epoch=10,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name="sample")
    data_module = SamShadowDataModule(args)
    data_module.setup("test")
    model = SamShadow.load_from_checkpoint(args.samshadow_ckpt_path)
    predictor = L.Trainer(
        accelerator='gpu',
        devices=args.test.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        logger=logger
    )
    predictions = predictor.predict(model, data_module)
    PSNR_SSIM_list_with_name = []
    PSNR = []
    SSIM = []
    save_path = args.save_result_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(len(predictions)):
        eval_psnr, eval_ssim, filename, sam_pred_soft_mask, shadow_removal_sr, diffusion_mask_pred, gt_image = predictions[i]
        filename = filename[0]
        PSNR_SSIM_list_with_name.append((f'{filename}_PSNR', eval_psnr))
        PSNR_SSIM_list_with_name.append((f'{filename}_SSIM', eval_ssim))
        PSNR.append(eval_psnr)
        SSIM.append(eval_ssim)
        # Save SR image
        sr_path = os.path.join(save_path, f'{filename}_sr.png')
        res = Metrics.tensor2img(shadow_removal_sr)
        sr_img = Image.fromarray(res)
        sr_img.save(sr_path)
        # Save HR image
        hr_path = os.path.join(save_path, f'{filename}_hr.png')
        hr_img = Metrics.tensor2img(gt_image)
        hr_img = Image.fromarray(hr_img)
        hr_img.save(hr_path)
        # Save mask
        soft_mask = sam_pred_soft_mask.squeeze().cpu().numpy() * 255
        soft_mask_path = os.path.join(save_path, f'{filename}_soft_mask.png')
        soft_mask_img = Image.fromarray(soft_mask.astype(np.uint8))
        soft_mask_img.save(soft_mask_path)

        mask = diffusion_mask_pred.squeeze().cpu().numpy() * 255
        mask_path = os.path.join(save_path, f'{filename}_diffusion_mask.png')
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img.save(mask_path)

    psnr_mean = np.mean(PSNR)
    ssim_mean = np.mean(SSIM)
    print(f'PSNR: {psnr_mean:}')
    print(f'SSIM: {ssim_mean:}')

    with open(os.path.join(save_path, 'PSNR_SSIM_list.log'), 'w') as file:
        for key, value in PSNR_SSIM_list_with_name:
            file.write(f"{key}: {value}\n")


@hydra.main(config_path="./config", config_name='soft_mask_diffusion_SRD', version_base=None)
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
