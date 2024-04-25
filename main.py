import inspect
import os
from typing import Dict, Any

import einops
import hydra
import numpy as np
import time
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, DataParallelStrategy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.diffusers_override.unet_2d_condition import UNet2DConditionModel
from models.inception import InceptionV3
from models.bigDis import BigDis, MidDis, JuniorDis
import itertools
import json
from einops import rearrange




class LightningDataset(pl.LightningDataModule):
    def __init__(self, args: DictConfig):
        super(LightningDataset, self).__init__()
        self.kwargs = {"num_workers": args.num_workers, "persistent_workers": True if args.num_workers > 0 else False,
                       "pin_memory": True}
        self.args = args

    def setup(self, stage="fit"):
        if self.args.dataset == "pororo":
            import datasets.pororo as data
        elif self.args.dataset == 'flintstones':
            import datasets.flintstones as data
        elif self.args.dataset == 'vistsis':
            import datasets.vistsis as data
        elif self.args.dataset == 'vistdii':
            import datasets.vistdii as data
        elif self.args.dataset == 'flintstones_unseen':
            import datasets.flintstones_load_unseen as data
        elif self.args.dataset == 'pororo_unseen':
            import datasets.pororo_load_unseen as data
        else:
            raise ValueError("Unknown dataset: {}".format(self.args.dataset))
        if stage == "fit":
            self.train_data = data.StoryDataset("train", self.args)
        if stage == "adapt":
            self.train_data = data.StoryDataset("train", self.args)
        if stage == "test":
            self.test_data = data.StoryDataset("test", self.args)
        if stage == "custom":
            self.test_data = data.CustomStory(self.args)
        if stage == "train":
            self.train_data = data.StoryDataset("train", self.args)
        if stage == "test_seen":
            self.test_data = data.StoryDataset("test_seen", self.args)
        if stage == "test_unseen":
            self.test_data = data.StoryDataset("test_unseen", self.args)
        if stage == "custom_unseen":
            self.test_data = data.CustomStory(self.args)

    def train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True,
                                          drop_last=True,
                                          **self.kwargs)
        return self.trainloader

    def val_dataloader(self):
        if self.val_data is None:
            return None
        return DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.batch_size, shuffle=False, **self.kwargs)

    def get_length_of_train_dataloader(self):
        if not hasattr(self, 'trainloader'):
            self.trainloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        return len(self.trainloader)


class ARLDM(pl.LightningModule):
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(ARLDM, self).__init__()
        self.args = args
        self.steps_per_epoch = int(steps_per_epoch)
        self.nominal_name_mapping = json.load(open(os.path.join(args.get(args.dataset).data_dir, 'char_name_mapping.json'), 'r'))
        self.known_chars = args.get(args.dataset).new_tokens
        self.automatic_optimization = False
        self.step_count = 0
        """
            Configurations
        """
        self.task = args.task
        if self.args.adversarial != 0:
            if self.args.D_net == 'simple':
                if self.args.D_loss_type != "wgan":
                    self.netD = nn.Sequential(
                        nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2),
                        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(512, 1),
                        nn.Sigmoid()
                    )
                else:
                    if self.args.D_net == 'simple':
                        self.netD = nn.Sequential(
                            nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                            nn.LeakyReLU(0.2),
                            nn.AdaptiveAvgPool2d((1, 1)),
                            nn.Flatten(),
                            nn.Linear(256, 1)
                        )
            elif self.args.D_net == 'big':
                self.netD = BigDis()
            elif self.args.D_net == 'mid':
                self.netD = MidDis()
            elif self.args.D_net == 'junior':
                self.netD = JuniorDis()
            else:
                print("Unknown type of discriminator, using a simple discriminator instead.")
                self.netD = nn.Sequential(
                    nn.Conv2d(8, 64, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )

        if args.mode == 'sample' or args.mode == 'custom_unseen' or args.mode == 'test_unseen' or args.mode == 'test_seen':
            if args.scheduler == "pndm":
                self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               skip_prk_steps=True)
            elif args.scheduler == "ddim":
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            else:
                raise ValueError("Scheduler not supported")
            self.fid_augment = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.inception = InceptionV3([block_idx])

        if args.distillation != 0:
            self.teacher_model = None
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        # init clip and blip tokenizers same as dataloader to get the token id for unseen chars
        self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        self.ptm_clip_tokenizer_len = len(self.clip_tokenizer)
        self.ptm_blip_tokenizer_len = len(self.blip_tokenizer)

        nominal_names = []
        self.target_chars = args.get(args.dataset).target_chars
        for char in self.target_chars:
            char_nominal_name = self.nominal_name_mapping[char][1]  # 1st ele is the nominal name, 2nd is the base token
            nominal_names.append(char_nominal_name)
        self.clip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        self.blip_tokenizer.add_tokens(nominal_names, special_tokens=True)

        self.ada_clip_tokenizer_len = len(self.clip_tokenizer)
        self.ada_blip_tokenizer_len = len(self.blip_tokenizer)

        # get the token id for unseen chars
        self.clip_token_ls = {}
        self.blip_token_ls = {}
        for names in nominal_names:
            self.clip_token_ls[names] = self.clip_tokenizer.convert_tokens_to_ids(names)
            self.blip_token_ls[names] = self.blip_tokenizer.convert_tokens_to_ids(names)

        self.blip_image_processor = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.max_length = args.get(args.dataset).max_length

        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))).unsqueeze(0).float()
        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder="text_encoder")
        # resize_position_embeddings
        old_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.max_length)
        self.text_encoder.text_model.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.text_model.embeddings.position_ids = torch.arange(self.max_length).expand((1, -1))

        self.modal_type_embeddings = nn.Embedding(6, 768)
        if self.args.mode == "test_unseen" and self.args.use_reference_image:
            # resize the embedding during test mode to load ckpt correctly, requires the train model also using reference
            self.time_embeddings = nn.Embedding(5, 768)
        else:
            self.time_embeddings = nn.Embedding(5, 768)
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
            image_size=224, vit='large')
        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="unet", tuning=args.unet_model.tuning, low_cpu_mem_usage=args.unet_model.low_cpu_mem_usage)
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)

        self.text_encoder.resize_token_embeddings(self.ptm_clip_tokenizer_len)
        self.mm_encoder.text_encoder.resize_token_embeddings(self.ptm_blip_tokenizer_len)

        if args.mode == 'test_unseen':
            self.text_encoder.resize_token_embeddings(self.ada_clip_tokenizer_len)
            self.mm_encoder.text_encoder.resize_token_embeddings(self.ada_blip_tokenizer_len)
        elif args.mode == 'test_seen':
            self.text_encoder.resize_token_embeddings(self.ada_clip_tokenizer_len)
            self.mm_encoder.text_encoder.resize_token_embeddings(self.ada_blip_tokenizer_len)
        elif args.mode == 'custom_unseen':
            self.text_encoder.resize_token_embeddings(self.ada_clip_tokenizer_len)
            self.mm_encoder.text_encoder.resize_token_embeddings(self.ada_blip_tokenizer_len)

        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet and not args.inject_lora:
            self.freeze_params(self.unet.parameters())
            self.unfreeze_params([p for n, p in self.unet.named_parameters() if "attention" in n])

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())
            self.unfreeze_params(self.mm_encoder.text_encoder.embeddings.word_embeddings.parameters())

        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())
            self.unfreeze_params(self.text_encoder.text_model.embeddings.token_embedding.parameters())

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # List to hold keys to remove
        keys_to_remove = []

        # Iterate over the keys in the state_dict
        for key in checkpoint['state_dict'].keys():
            # Check if key matches the pattern for parts to exclude
            if key.startswith('teacher_model'):
                keys_to_remove.append(key)

        # Remove the specified keys from the state_dict
        for key in keys_to_remove:
            del checkpoint['state_dict'][key]

    def text_emb_resize(self, clip_len, blip_len):
        self.text_encoder.resize_token_embeddings(clip_len)
        self.mm_encoder.text_encoder.resize_token_embeddings(blip_len)

    def freeze_chars_emb(self, char_name, unfreeze=False):
        try:
            nominal_name = self.nominal_name_mapping[char_name][1]
        except KeyError:
            nominal_name = char_name
        if unfreeze:
            print("unfreezing char emb:", char_name)
            clip_token_id = self.clip_tokenizer.convert_tokens_to_ids(nominal_name)
            blip_token_id = self.blip_tokenizer.convert_tokens_to_ids(nominal_name)
            self.text_encoder.text_model.embeddings.token_embedding.weight.data[clip_token_id].requires_grad = True
            self.mm_encoder.text_encoder.embeddings.word_embeddings.weight.data[blip_token_id].requires_grad = True
        else:
            print("freezing char emb:", char_name)
            clip_token_id = self.clip_tokenizer.convert_tokens_to_ids(nominal_name)
            blip_token_id = self.blip_tokenizer.convert_tokens_to_ids(nominal_name)
            self.text_encoder.text_model.embeddings.token_embedding.weight.data[clip_token_id].requires_grad = False
            self.mm_encoder.text_encoder.embeddings.word_embeddings.weight.data[blip_token_id].requires_grad = False

    def resize_time_embeddings(self, num_time_steps):
        self.time_embeddings = nn.Embedding(num_time_steps, 768)

    def adversarial_diffusion(self, images, latent, noisy_latent, noise_pred, timesteps, refer_char, refer_img, texts):
        refer_latent = self.vae.encode(refer_img).latent_dist.sample() * 0.18215
        V = 5
        B = latent.shape[0] // V
        num_rows = len(texts[0])
        texts = [[col[i] for col in texts] for i in range(num_rows)]
        texts = list(itertools.chain.from_iterable(texts))
        pos_samples = []
        neg_samples = []
        refer_latents = []

        if B < 2:
            return torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)

        for i in range(B * V):
            if refer_char[i // V] in texts[i].lower():
                _tmp = refer_char[i // V]
                pos_samples.append(latent[i])
                neg_samples.append(self.estimate_x0(noisy_latent[i].unsqueeze(0), noise_pred[i].unsqueeze(0),
                                                    timesteps[i].unsqueeze(0)))
                refer_latents.append(refer_latent[i // V])

        if len(pos_samples) > 0:
            pos_samples = torch.stack(pos_samples)
            neg_samples = torch.cat(neg_samples)
            refer_latents = torch.stack(refer_latents)

            pos_samples = torch.cat([pos_samples, refer_latents], dim=1)
            neg_samples = torch.cat([neg_samples, refer_latents], dim=1)

            pos_preds = self.netD(pos_samples)
            neg_preds = self.netD(neg_samples)

            if self.args.D_loss_type == "hinge":

                # Hinge loss for discriminator
                d_loss = torch.mean(F.relu(1. - pos_preds)) + torch.mean(F.relu(1. + neg_preds))

                # Adjusted accuracy calculation for hinge loss
                pos_correct = (pos_preds > 0).float().mean()
                neg_correct = (neg_preds < 0).float().mean()
                overall_acc = (pos_correct + neg_correct) / 2
                self.log('d_acc', overall_acc, prog_bar=True)

                # Hinge loss for generator
                g_adv_loss = -torch.mean(neg_preds)
            elif self.args.D_loss_type == "bce":
                pos_labels = torch.ones_like(pos_preds)
                neg_labels = torch.zeros_like(neg_preds)

                pos_loss = F.binary_cross_entropy(pos_preds, pos_labels)
                neg_loss = F.binary_cross_entropy(neg_preds, neg_labels)
                d_loss = pos_loss + neg_loss

                overall_acc = ((pos_preds > 0.5).float().mean() + (neg_preds < 0.5).float().mean()) / 2
                self.log('d_acc', overall_acc, prog_bar=True)

                g_adv_loss = -torch.log(neg_preds + 1e-8).mean()
            elif self.args.D_loss_type == "wgan":
                d_loss = torch.mean(neg_preds) - torch.mean(pos_preds)
                g_adv_loss = -torch.mean(neg_preds)

            else:
                raise ValueError("Unknown type of discriminator loss")

        else:
            d_loss = torch.tensor(0.0, requires_grad=True).to(self.device)
            g_adv_loss = torch.tensor(0.0, requires_grad=True).to(self.device)

        return d_loss, g_adv_loss

    def estimate_x0(self, xt, pred, t):
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** 0.5 * pred) / alpha_prod_t ** (0.5)

        return x0_pred

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def configure_optimizers(self):
        param_groups = [
            {"params": self.unet.parameters(), "lr": 1e-5},
            {"params": self.text_encoder.parameters(), "lr": 1e-5},
            {"params": self.mm_encoder.parameters(), "lr": 1e-5}
        ]
        optimizer = torch.optim.AdamW(param_groups,
                                     lr=self.args.init_lr,
                                     weight_decay=1e-4)
        warm_up_steps = self.args.warmup_epochs * self.steps_per_epoch / self.args.grad_accumulation_steps
        max_step = self.args.max_epochs * self.steps_per_epoch / self.args.grad_accumulation_steps
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=warm_up_steps,
                                                  max_epochs=max_step)
        if self.args.adversarial != 0:
            optimizer_D = torch.optim.AdamW(self.netD.parameters(),
                                            lr=self.args.discriminator_lr,
                                            weight_decay=1e-4)
            scheduler_D = LinearWarmupCosineAnnealingLR(optimizer_D,
                                                        warmup_epochs=warm_up_steps,
                                                        max_epochs=max_step)
            optim_dict = [
                {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                    }
                },
                {
                    'optimizer': optimizer_D,
                    'lr_scheduler': {
                        'scheduler': scheduler_D,
                        'interval': 'step',
                    }
                }
            ]
            return optim_dict
        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The LR scheduler instance (required)
                'interval': 'step',  # The unit of the scheduler's step size
            }
        }
        return optim_dict

    def encoder_forward(self, captions, attention_mask, source_images, source_caption,
                             source_attention_mask):
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' else V
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)

        classifier_free_idx = np.random.rand(B * V) < 0.1

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal')
        source_embeddings = source_embeddings.reshape(B, src_V * S, -1)
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        if src_V == 6:
            time_seq = torch.tensor([0, 1, 2, 3, 4, 5], device=self.device)
            source_embeddings += self.time_embeddings(time_seq.repeat_interleave(S))
        else:
            time_seq = torch.arange(src_V, device=self.device)
            source_embeddings += self.time_embeddings(
                time_seq.repeat_interleave(S, dim=0))

        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        _attention_mask = attention_mask.detach().cpu().numpy()
        attention_mask[classifier_free_idx] = False

        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        return encoder_hidden_states, attention_mask

    def forward(self, batch):
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.args.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()
        images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts, index, \
           unseen_flag, refer_img, refer_char = batch

        if self.args.adversarial != 0:
            optimizer_g, optimizer_d = self.optimizers()

        # toggle the optimizer for the generator
        self.toggle_optimizer(optimizer_g, 0)

        # get encoder hidden states and attention mask
        encoder_hidden_states, clipblip_attention_mask = self.encoder_forward(captions, attention_mask, source_images,
                                                                     source_caption, source_attention_mask)

        # encode images
        latents = self.vae.encode(torch.flatten(images, 0, 1)).latent_dist.sample()
        latents = latents * 0.18215

        # add noise to latents
        noise = torch.randn(latents.shape, device=self.device)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)


        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, clipblip_attention_mask).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
        self.log('mse_loss', loss, prog_bar=True)
        if self.args.distillation and self.teacher_model is not None:
            with torch.no_grad():
                teacher_noise_pred = self.teacher_model.unet(noisy_latents, timesteps, encoder_hidden_states,
                                                             clipblip_attention_mask).sample
            seen_pred = noise_pred
            distill_loss = F.mse_loss(seen_pred, teacher_noise_pred, reduction='none').mean() * self.args.distillation
            self.log('dis_loss', distill_loss, prog_bar=True)
            loss += distill_loss
        d_loss, g_loss = self.adversarial_diffusion(images, latents, noisy_latents, noise_pred, timesteps, refer_char, refer_img, texts)
        self.log('g_loss', g_loss, prog_bar=True)
        if self.global_step > self.args.start_g_adv:
            loss += g_loss * self.args.adversarial
        self.manual_backward(loss)
        # clipping
        self.clip_gradients(optimizer_g, 1.0)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(0)


        self.toggle_optimizer(optimizer_d, 1)
        # encoder forward
        encoder_hidden_states, clipblip_attention_mask = self.encoder_forward(captions, attention_mask, source_images,
                                                                     source_caption, source_attention_mask)
        # encode images
        latents = self.vae.encode(torch.flatten(images, 0, 1)).latent_dist.sample()
        latents = latents * 0.18215

        # add noise to latents
        noise = torch.randn(latents.shape, device=self.device)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, clipblip_attention_mask).sample
        d_loss, adv_g_loss = self.adversarial_diffusion(images, latents, noisy_latents, noise_pred, timesteps, refer_char,
                                                refer_img, texts)
        loss = d_loss
        self.log('d_loss', d_loss, prog_bar=True)
        self.manual_backward(loss)
        self.clip_gradients(optimizer_d, 1.0)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(1)

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()
        self.step_count += 1

    def sample(self, batch):
        original_images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts, index, \
            unseen_flag, refer_img, refer_text = batch
        B, V, S = captions.shape
        src_V = V + 1 if self.task == 'continuation' or self.args.use_reference_image else V
        original_images = torch.flatten(original_images, 0, 1)
        captions = torch.flatten(captions, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_images = torch.flatten(source_images, 0, 1)
        source_caption = torch.flatten(source_caption, 0, 1)
        source_attention_mask = torch.flatten(source_attention_mask, 0, 1)
        # text emb for all target pairs
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        # mm emb for all pairs
        source_embeddings = self.mm_encoder(source_images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        if src_V == 6:
            time_seq = torch.tensor([0, 0, 1, 2, 3, 4], device=self.device)
            source_embeddings += self.time_embeddings(time_seq.repeat_interleave(S))
        else:
            time_seq = torch.arange(src_V, device=self.device)
            source_embeddings += self.time_embeddings(
                time_seq.repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0) # repeat for each target pair
        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)
        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        # Generate the image-level mask for auto-regressive generation
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1)
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1)
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=self.device).bool()
        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        attention_mask = attention_mask.reshape(2, B, V, (src_V + 1) * S)

        images = list()
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, (src_V + 1) * S, -1)

            new_image = self.diffusion(encoder_hidden_states[:, :, i].reshape(2 * B, (src_V + 1) * S, -1),
                                       attention_mask[:, :, i].reshape(2 * B, (src_V + 1) * S),
                                       self.args.resolution, self.args.resolution, self.args.num_inference_steps, self.args.guidance_scale, 0.0)
            images += new_image

            new_image = torch.stack([self.blip_image_processor(im) for im in new_image]).to(self.device)
            new_embedding = self.mm_encoder(new_image,  # B,C,H,W
                                            source_caption.reshape(B, src_V, S)[:, i + src_V - V],
                                            source_attention_mask.reshape(B, src_V, S)[:, i + src_V - V],
                                            mode='multimodal')  # B, S, D
            new_embedding = new_embedding.repeat_interleave(V, dim=0)
            new_embedding += self.modal_type_embeddings(torch.tensor(1, device=self.device))
            new_embedding += self.time_embeddings(torch.tensor(i + src_V - V, device=self.device))

            encoder_hidden_states = encoder_hidden_states[1].reshape(B * V, (src_V + 1) * S, -1)
            # update the BLIP embedding, i=1, [:, 1+1+5-4 * 768:1+2+5-4 * 768]
            encoder_hidden_states[:, (i + 1 + src_V - V) * S:(i + 2 + src_V - V) * S] = new_embedding
            encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])

        return original_images, images, texts, index, unseen_flag

    def training_step(self, batch, batch_idx):
        self(batch)

    def validation_step(self, batch, batch_idx):
        self(batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        BATCH_SIZE = batch[0].shape[0]
        STORY_LEN = batch[0].shape[1]
        original_images, images, texts, index, unseen_flag = self.sample(batch)
        original_images = original_images.cpu().numpy().astype('uint8')
        original_images = [Image.fromarray(im, 'RGB') for im in original_images]
        if self.args.calculate_fid:
            ori = self.inception_feature(original_images).cpu().numpy()
            gen = self.inception_feature(images).cpu().numpy()
        else:
            ori = None
            gen = None
        # transpose the texts
        reshape_text = []
        for i in range(len(texts[0])):
            _tmp = []
            for j in range(len(texts)):
                _tmp.append(texts[j][i])
            reshape_text.append(_tmp)
        # transpose the unseen flag
        unseen_flag = [t.cpu().numpy() for t in unseen_flag]
        reshape_unseen_flag = []
        for i in range(len(unseen_flag[0])):
            _tmp = []
            for j in range(len(unseen_flag)):
                _tmp.append(unseen_flag[j][i])
            reshape_unseen_flag.append(_tmp)
        # transpose the image list
        reshape_image = []
        for i in range(0, BATCH_SIZE):
            img_of_story = []
            for j in range(i, BATCH_SIZE * STORY_LEN, BATCH_SIZE):
                img_of_story.append(images[j])
            reshape_image.append(img_of_story)

        return reshape_image, ori, gen, original_images, reshape_text, index, reshape_unseen_flag

    def diffusion(self, encoder_hidden_states, attention_mask, height, width, num_inference_steps, guidance_scale, eta):
        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, height // 8, width // 8),
                              device=self.device)
        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states).sample
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]

        return pil_images

    def inception_feature(self, images):
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.inception(images)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)


def train(args: DictConfig) -> None:
    dataloader = LightningDataset(args)
    dataloader.setup('fit')
    model = ARLDM(args, steps_per_epoch=dataloader.get_length_of_train_dataloader())

    logger = TensorBoardLogger(save_dir=os.path.join(args.ckpt_dir, args.run_name), name='log', default_hp_metric=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, args.run_name),
        save_top_k=-1,
        every_n_epochs=args.save_freq,
        filename='{epoch}-{step}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callback_list = [lr_monitor, checkpoint_callback]

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        limit_val_batches=0.0
    )
    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)

def sample(args: DictConfig) -> None:
    assert args.test_model_file is not None, "test_model_file cannot be None"
    dataloader = LightningDataset(args)
    dataloader.setup('test')
    model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16
    )
    predictions = predictor.predict(model, dataloader)
    generated_images = [elem for sublist in predictions for elem in sublist[0]]
    original_images = [elem for sublist in predictions for elem in sublist[3]]
    texts = [elem for sublist in predictions for elem in sublist[4]]
    indexes = [elem for sublist in predictions for elem in sublist[5]]
    indexes = [int(i) for i in indexes]
    # unique identifiers for each story

    if not os.path.exists(args.sample_output_dir):
        os.makedirs(args.sample_output_dir, exist_ok=True)

    if args.sample_output_dir is not None:
        for i, story in enumerate(generated_images):
            character_output_dir = os.path.join(args.sample_output_dir)
            if not os.path.exists(character_output_dir):
                os.makedirs(character_output_dir)
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            if not os.path.exists(img_folder_name):
                os.makedirs(img_folder_name, exist_ok=True)
            for j, image in enumerate(story):
                image_path = os.path.join(img_folder_name, f'{j}_generated.png')
                image.save(image_path)

        # group original images based on length
        original_images = [original_images[i:i + 5] for i in range(0, len(original_images), 5)] \
            if args.task == 'visualization' else [original_images[i:i + 4] for i in range(0, len(original_images), 4)]
        for i, story in enumerate(original_images):
            character_output_dir = os.path.join(args.sample_output_dir)
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            if not os.path.exists(img_folder_name):
                os.makedirs(img_folder_name, exist_ok=True)
            for j, image in enumerate(story):
                image_path = os.path.join(img_folder_name, f'{j}_original.png')
                image.save(image_path)

        # saving texts for each story
        for i, story in enumerate(texts):
            character_output_dir = os.path.join(args.sample_output_dir)
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            text_path = os.path.join(img_folder_name, 'texts.json')
            with open(text_path, 'w') as f:
                json.dump(story, f, indent=4)

def train_unseen(args: DictConfig) -> None:
    target_chars = args.get(args.dataset).target_chars

    model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    logger = TensorBoardLogger(save_dir=os.path.join(args.ckpt_dir, args.run_name), name='log', default_hp_metric=False)

    if args.distillation != 0:
        teacher_model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        model.teacher_model = teacher_model

    model.text_emb_resize(model.ada_clip_tokenizer_len, model.ada_blip_tokenizer_len)

    # if args.use_reference_image:
    #     model.resize_time_embeddings(6)

    seen_chars = args.get(args.dataset).new_tokens
    for char in seen_chars:
        model.freeze_chars_emb(char, unfreeze=False)
    for char in target_chars:
        model.freeze_chars_emb(char, unfreeze=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, args.run_name),
        save_top_k=-1,
        every_n_epochs=args.save_freq,
        filename='{epoch}',
        save_weights_only=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callback_list = [lr_monitor, checkpoint_callback]
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=args.max_epochs,
        benchmark=True,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callback_list,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_val_batches=0.0,
        precision=32,
    )
    dataloader = LightningDataset(args)
    dataloader.setup('train')

    # Update the model with the new dataloader length
    new_steps_per_epoch = dataloader.get_length_of_train_dataloader() / len(args.gpu_ids)
    model.steps_per_epoch = new_steps_per_epoch

    trainer.fit(model, dataloader, ckpt_path=args.train_model_file)


def test_unseen(args: DictConfig) -> None:
    target_chars = args.get(args.dataset).target_chars
    for k, cur_char in enumerate(target_chars):
        args['history_char'] = []
        args['cur_char'] = cur_char
        for j in range(k + 1):
            args['history_char'].append(target_chars[j])
        model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

        predictor = pl.Trainer(
            accelerator='gpu',
            devices=args.gpu_ids,
            max_epochs=-1,
            benchmark=True,
            strategy=DDPStrategy(find_unused_parameters=True),
            precision=16
        )

        if args.use_reference_image:
            model.resize_time_embeddings(6)

        args['cur_char'] = cur_char
        dataloader = LightningDataset(args)
        dataloader.setup('test_unseen')

        predictions = predictor.predict(model, dataloader)
        generated_images = [elem for sublist in predictions for elem in sublist[0]]
        original_images = [elem for sublist in predictions for elem in sublist[3]]
        texts = [elem for sublist in predictions for elem in sublist[4]]
        indexes = [elem for sublist in predictions for elem in sublist[5]]
        indexes = [int(i) for i in indexes]
        unseen_flags = [elem for sublist in predictions for elem in sublist[6]]

        if not os.path.exists(args.sample_output_dir):
            os.makedirs(args.sample_output_dir, exist_ok=True)

        if args.sample_output_dir is not None:
            checkpoint_output_dir = os.path.join(args.sample_output_dir, f"{k}_{cur_char}")
            os.makedirs(checkpoint_output_dir, exist_ok=True)

            local_rank = predictor.local_rank

            time_delay = int(local_rank) * 2
            time.sleep(time_delay)
            print(f"Rank: {local_rank} is waiting for {time_delay} sec to begin saving images and texts!")

            for i, story in enumerate(generated_images):
                character_output_dir = checkpoint_output_dir
                if not os.path.exists(character_output_dir):
                    os.makedirs(character_output_dir)
                img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
                if not os.path.exists(img_folder_name):
                    os.makedirs(img_folder_name, exist_ok=True)
                for j, image in enumerate(story):
                    if unseen_flags[i][j]:
                        image_path = os.path.join(img_folder_name, f'{j}_generated_eval.png')
                    else:
                        image_path = os.path.join(img_folder_name, f'{j}_generated.png')
                    image.save(image_path)

            original_images = [original_images[i:i + 5] for i in range(0, len(original_images), 5)] \
                if args.task == 'visualization' or args.use_reference_image else [original_images[i:i + 4] for i in
                                                      range(0, len(original_images), 4)]

            for i, story in enumerate(original_images):
                character_output_dir = checkpoint_output_dir
                img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
                if not os.path.exists(img_folder_name):
                    os.makedirs(img_folder_name, exist_ok=True)
                for j, image in enumerate(story):
                    if unseen_flags[i][j]:
                        image_path = os.path.join(img_folder_name, f'{j}_original_eval.png')
                    else:
                        image_path = os.path.join(img_folder_name, f'{j}_original.png')
                    image.save(image_path)

            for i, story in enumerate(texts):
                character_output_dir = checkpoint_output_dir
                img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
                text_path = os.path.join(img_folder_name, 'texts.json')
                with open(text_path, 'w') as f:
                    json.dump(story, f, indent=4)
    time.sleep(10)

def test_seen(args: DictConfig) -> None:
    target_chars = args.get(args.dataset).target_chars
    last_name = target_chars[-1]

    args['history_char'] = target_chars
    args['cur_char'] = last_name
    test_model_file = os.path.join(args.test_model_file, f"epoch=99-{last_name}.ckpt")
    model = ARLDM.load_from_checkpoint(test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16
    )

    dataloader = LightningDataset(args)
    dataloader.setup('test_seen')

    predictions = predictor.predict(model, dataloader)
    generated_images = [elem for sublist in predictions for elem in sublist[0]]
    original_images = [elem for sublist in predictions for elem in sublist[3]]
    texts = [elem for sublist in predictions for elem in sublist[4]]
    indexes = [elem for sublist in predictions for elem in sublist[5]]
    indexes = [int(i) for i in indexes]
    unseen_flags = [elem for sublist in predictions for elem in sublist[6]]

    if args.sample_output_dir is not None:
        os.makedirs(args.sample_output_dir, exist_ok=True)
        checkpoint_output_dir = args.sample_output_dir

        for i, story in enumerate(generated_images):
            character_output_dir = os.path.join(checkpoint_output_dir, args['cur_char'])
            if not os.path.exists(character_output_dir):
                os.makedirs(character_output_dir)
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            if not os.path.exists(img_folder_name):
                os.makedirs(img_folder_name, exist_ok=True)
            for j, image in enumerate(story):
                if unseen_flags[i][j]:
                    image_path = os.path.join(img_folder_name, f'{j}_generated_eval.png')
                else:
                    image_path = os.path.join(img_folder_name, f'{j}_generated.png')
                image.save(image_path)

        original_images = [original_images[i:i + 5] for i in range(0, len(original_images), 5)] \
            if args.task == 'visualization' else [original_images[i:i + 4] for i in
                                                  range(0, len(original_images), 4)]
        for i, story in enumerate(original_images):
            character_output_dir = os.path.join(checkpoint_output_dir, args['cur_char'])
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            if not os.path.exists(img_folder_name):
                os.makedirs(img_folder_name, exist_ok=True)
            for j, image in enumerate(story):
                if unseen_flags[i][j]:
                    image_path = os.path.join(img_folder_name, f'{j}_original_eval.png')
                else:
                    image_path = os.path.join(img_folder_name, f'{j}_original.png')
                image.save(image_path)

        for i, story in enumerate(texts):
            character_output_dir = os.path.join(checkpoint_output_dir, args['cur_char'])
            img_folder_name = os.path.join(character_output_dir, f'{indexes[i]}')
            text_path = os.path.join(img_folder_name, 'texts.json')
            with open(text_path, 'w') as f:
                json.dump(story, f, indent=4)

def custom_unseen(args: DictConfig) -> None:
    dataloader = LightningDataset(args)
    dataloader.setup('custom_unseen')
    model = ARLDM.load_from_checkpoint(args.test_model_file, args=args, strict=False)

    predictor = pl.Trainer(
        accelerator='gpu',
        devices=args.gpu_ids,
        max_epochs=-1,
        benchmark=True,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=16
    )
    predictions = predictor.predict(model, dataloader)

    generated_images = [elem for sublist in predictions for elem in sublist[0]]

    if not os.path.exists(args.sample_output_dir):
        os.makedirs(args.sample_output_dir, exist_ok=True)

    if args.sample_output_dir is not None:
        for i, story in enumerate(generated_images):
            character_output_dir = os.path.join(args.sample_output_dir, 'custom_unseen')
            if not os.path.exists(character_output_dir):
                os.makedirs(character_output_dir)
            img_folder_name = os.path.join(character_output_dir, f'{i}')
            if not os.path.exists(img_folder_name):
                os.makedirs(img_folder_name, exist_ok=True)
            for j, image in enumerate(story):
                image_path = os.path.join(img_folder_name, f'{j}_generated.png')
                image.save(image_path)


@hydra.main(config_path="./config", config_name="config")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'train_unseen':
        train_unseen(args)
    elif args.mode == 'test_unseen':
        test_unseen(args)
    elif args.mode == 'test_seen':
        test_seen(args)
    elif args.mode == 'custom_unseen':
        custom_unseen(args)


if __name__ == '__main__':
    main()
