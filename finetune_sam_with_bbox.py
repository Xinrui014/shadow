import os
import cv2
import numpy as np
from PIL import Image
import hydra
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model.sam_adapter.sam_adapt import SAM
import core.metrics as Metrics
from data.LRHR_dataset import TuneSAM
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
import model.networks as networks
from model.sam_adapter.iou_loss import IOU
from lightning.pytorch.loggers import TensorBoardLogger
from model.segment_anything import sam_model_registry
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.segment_anything.sam_lora import LoRA_Sam

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class SAM_with_bbox(nn.Module):
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super(SAM_with_bbox, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self, image, bbox):
        self.features = self.image_encoder(image)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=bbox,
            masks=None,
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings = self.features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        return low_res_masks # 256x256

class SamDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_workers = args.datasets.train.num_workers
        self.shuffle = args.datasets.train.use_shuffle


    def setup(self, stage="fit"):
        if stage == "fit":
            self.train_data = TuneSAM(self.args.datasets.train.dataroot, self.args.bbox_path)
            # self.val_data = TuneSAM(self.args.datasets.test.dataroot, self.args.bbox_path, data_len=3)
        elif stage == "test":
            self.test_data = TuneSAM(self.args.datasets.test.dataroot, self.args.test_bbox_path)

    def train_dataloader(self):
        dataloader = DataLoader(self.train_data,
                                batch_size=self.args.datasets.train.batch_size,
                                shuffle=self.shuffle,
                                num_workers=self.num_workers,
                                pin_memory=True)
        return dataloader

    # def val_dataloader(self):
    #     dataloader = DataLoader(self.val_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
    #                             num_workers=self.num_workers, pin_memory=True)
    #     return dataloader

    def predict_dataloader(self):
        dataloader = DataLoader(self.test_data, batch_size=self.args.datasets.test.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        return dataloader


class SAMFinetune(L.LightningModule):
    def __init__(self, args, path=None):
        super(SAMFinetune, self).__init__()
        self.args = args
        self.save_model_name = args.name
        # self.automatic_optimization = False

        self.sam_model = sam_model_registry[args.model_type](checkpoint=path.SAM)
        self.sam = SAM_with_bbox(image_encoder=self.sam_model.image_encoder,
                                 prompt_encoder=self.sam_model.prompt_encoder,
                                 mask_decoder=self.sam_model.mask_decoder)

        # self.sam_lora = LoRA_Sam(self.sam, args.sam_rank)
        # self.sam_lora = self.sam_lora.sam
        self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        self.save_hyperparameters()
        self.freeze_parameters(self.sam.image_encoder.parameters())
        self.freeze_parameters(self.sam.prompt_encoder.parameters())

    @staticmethod
    def freeze_parameters(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_parameters(params):
        for param in params:
            param.requires_grad = True

    def forward(self, train_data):
        sam_input = train_data['sam_SR']
        bbox = train_data['bbox']
        pred_mask = self.sam_lora(sam_input, bbox)
        pred_mask = torch.sigmoid(pred_mask)
        residual = (train_data['HR'] + 1) / 2 - (train_data['SR'] + 1) / 2
        residual = torch.mean(residual, dim=1, keepdim=True)
        residual_mask = torch.where(residual < 0.05, torch.zeros_like(residual), torch.ones_like(residual))
        self.segmentation_loss = self.criterionBCE(pred_mask, residual_mask)
        self.log('segmentation_loss', self.segmentation_loss.item(), on_step=True)

        return {'loss': self.segmentation_loss,
                'pred_mask': pred_mask,
                'residual_mask': residual_mask}


    def sample(self, data):
        sam_input = data['sam_SR']
        bbox = data['bbox']
        pred_mask = self.sam(sam_input, bbox)
        pred_mask = torch.sigmoid(pred_mask)
        return pred_mask

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

    def training_step(self, train_data, batch_idx):
        result = self(train_data)
        loss = result['loss']
        pred_mask = result['pred_mask']
        residual_mask = result['residual_mask']
        filename = train_data['filename']
        log_img = train_data['SR'][0].clamp_(-1, 1)
        lr_img = (log_img + 1) / 2



        # print("After training step:")
        # self.print_param_values()
        if self.global_step % 50 == 0:
            self.logger.experiment.add_image('Train/sam_pred_mask', pred_mask[0], self.global_step)
            self.logger.experiment.add_image('Train/residual_mask', residual_mask[0], self.global_step)
            self.logger.experiment.add_image('Train/lr_image', lr_img, self.global_step)
            self.logger.experiment.add_text('Train/filename_sam_pred_mask', filename[0], self.global_step)

        return loss


    def validation_step(self, val_data, batch_idx):
        shadow_removal_sr, diffusion_mask_pred, _ = self.sample(val_data)
        res = Metrics.tensor2img(shadow_removal_sr)
        hr_img = Metrics.tensor2img(val_data['HR'])
        eval_psnr = Metrics.calculate_psnr(res, hr_img)
        eval_ssim = Metrics.calculate_ssim(res, hr_img)
        log_img = shadow_removal_sr[0].clamp_(-1, 1)
        log_img = (log_img + 1) / 2
        self.logger.experiment.add_image('Val/Shadow_Removal', log_img, self.global_step)
        self.logger.experiment.add_image('Val/Diffusion_Mask', diffusion_mask_pred[0],self.global_step)
        self.log('Val/psnr', eval_psnr, sync_dist=True)
        self.log('Val/ssim', eval_ssim, sync_dist=True)


    def predict_step(self, test_data):
        pred_mask = self.sample(test_data)

        residual = (test_data['HR'] + 1) / 2 - (test_data['SR'] + 1) / 2
        residual = torch.mean(residual, dim=1, keepdim=True)
        residual_mask = torch.where(residual < 0.05, torch.zeros_like(residual), torch.ones_like(residual))
        filename = test_data['filename']
        return pred_mask, residual_mask, test_data['SR'], filename


    def load_pretrained_models(self, path):
        gen_path = '{}_gen.pth'.format(path.ddpm)
        self.diffusion.load_state_dict(torch.load(gen_path), strict=False)

        sam_state_dict = torch.load(path.sam, map_location=self.device)
        self.sam.load_state_dict(sam_state_dict, strict=False)

    def configure_optimizers(self):
        sam_optimizer = torch.optim.Adam(self.sam.parameters(), lr=1e-4)

        return sam_optimizer


# using SAM mask train diffusion
def train(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}",
                               name=args.version + "_logger")
    data_module = SamDataModule(args)
    data_module.setup("fit")
    ckpt_path = args.ckpt_path
    # model = diffusion(networks.define_G, args, ckpt_path)
    model = SAMFinetune(args, ckpt_path)

    # load training parameters
    save_model_name = args.name
    max_epochs = args.train.max_epochs
    save_every_n_epochs = args.train.every_n_epochs
    log_every_n_steps = 1
    log_every_n_epochs = 1

    save_path = f"./experiments_lightning/{save_model_name}/{args.version}"
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
        # gradient_clip_val=0.5,
        default_root_dir=save_path,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        num_sanity_val_steps=0,
        # check_val_every_n_epoch=10,
        log_every_n_steps=log_every_n_steps,
        accumulate_grad_batches=4 # mini batch size is 4 so the overall batchsize is 16
        # strategy=DDPStrategy(gradient_as_bucket_view=True)
    )
    trainer.fit(model, data_module)


def sample(args: DictConfig) -> None:
    logger = TensorBoardLogger(save_dir=f"./experiments_lightning/{args.name}", name="sample")
    data_module = SamDataModule(args)
    data_module.setup("test")
    model = SAMFinetune.load_from_checkpoint(args.samshadow_ckpt_path)
    predictor = L.Trainer(
        accelerator='gpu',
        devices=args.test.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        logger=logger
    )
    predictions = predictor.predict(model, data_module)
    BER = []
    save_path = os.path.join(args.save_result_path, 'check_mask')
    if not os.path.exists(args.save_result_path):
        os.makedirs(args.save_result_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in range(len(predictions)):
        pred_mask, residual_mask, lr_image, filename = predictions[i]
        filename = filename[0]

        ber = Metrics.calc_ber(pred_mask, residual_mask)


        lr_path = os.path.join(save_path,f'{filename}_lr.png')
        lr_img = Metrics.tensor2img(lr_image)
        lr_img = Image.fromarray(lr_img)
        lr_img.save(lr_path)
        # Save mask
        soft_mask = pred_mask.squeeze().cpu().numpy() * 255
        soft_mask_path = os.path.join(save_path, f'{filename}_pred_mask.png')
        soft_mask_img = Image.fromarray(soft_mask.astype(np.uint8))
        soft_mask_img.save(soft_mask_path)

        residual_mask = residual_mask.squeeze().cpu().numpy() * 255
        residual_mask_path = os.path.join(save_path, f'{filename}_residual_mask.png')
        residual_mask_img = Image.fromarray(residual_mask.astype(np.uint8))
        residual_mask_img.save(residual_mask_path)
        BER.append(ber[2])

    BER = np.mean(BER)
    print(f'BER: {BER}')



@hydra.main(config_path="./config", config_name='finetune_sam_with_bbox', version_base=None)
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
