from io import BytesIO
import lmdb
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os
from torchvision import transforms
import yaml
import numpy as np


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution

        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        if split == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = 'train_B'
        else:
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = 'test_B'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.split == 'train':
                hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
            else:
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
            hr_name = hr_name.replace('_A', '_C')
            img_HR = Image.open(hr_name).convert("RGB")
            img_mask = Image.open(self.mask_path[index]).convert("1")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index, 'LR_path': self.sr_path[index]}
        else:
            [img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}

class TrainDataset(Dataset):
    def __init__(self, dataroot, sam_inp_size=1024, mask_inp_size=256):
        gt_dir = 'train_C'
        input_dir = 'train_A'
        mask_dir = 'train_B'

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((sam_inp_size, sam_inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_inp_size, mask_inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.hr_path)

    def __getitem__(self, index):
        img_LR = None

        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")

        hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("1")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="train", min_max=(-1, 1))

        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)




        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}

class TestDataset(Dataset):
    def __init__(self, dataroot, sam_inp_size=1024, mask_inp_size=256, data_len=-1):
        gt_dir = 'test_C'
        input_dir = 'test_A'
        mask_dir = 'test_B'
        self.data_len = data_len

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((sam_inp_size, sam_inp_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((mask_inp_size, mask_inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])


    def __len__(self):
        if self.data_len == -1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")
        hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
        hr_name = hr_name.replace('_A', '_C')
        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("1")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="test", min_max=(-1, 1))
        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)
        filename = self.sr_path[index].split('/')[-1].split('.')[0]


        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'filename': filename,
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask}

class TuneSAM(Dataset):
    def __init__(self, dataroot, gt_mask_dir, yaml_path, data_len=-1):
        self.data_len = data_len

        self.phase = dataroot.split('/')[-1]
        if self.phase == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = gt_mask_dir

        elif self.phase == 'test':
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = gt_mask_dir

        clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
        mask_files = sorted(os.listdir(mask_dir))
        # mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

        self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
        self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
        self.mask_path = [os.path.join(mask_dir, x) for x in mask_files]

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        with open(yaml_path, 'r') as file:
            self.bbox_data = yaml.safe_load(file)


    def __len__(self):
        if self.data_len==-1:
            return len(self.hr_path)
        else:
            return self.data_len

    def __getitem__(self, index):
        img_SR_original = Image.open(self.sr_path[index]).convert("RGB")


        if self.phase == 'train':
            hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
        elif self.phase == 'test':
            hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')

        hr_name = hr_name.replace('_A', '_C')

        img_HR_original = Image.open(hr_name).convert("RGB")
        img_mask_original = Image.open(self.mask_path[index]).convert("L")
        [shadow_img_SR, shadow_img_HR, shadow_img_mask] = Util.transform_augment([img_SR_original, img_HR_original, img_mask_original], split="train", min_max=(-1, 1))

        # sam input and sam mask
        sam_img_SR = self.img_transform(img_SR_original)
        sam_img_mask = self.mask_transform(img_mask_original)
        image_name = self.sr_path[index].split('/')[-1].split('.')[0]
        bbox = self.bbox_data[image_name]
        # bbox = bbox[:1]  # Take only the first four bounding boxes if more are present
        # while len(bbox) < 4:
        #     bbox.append([-1, -1, -1, -1])     # Pad with -1 if fewer than four bboxes
        # bbox = np.expand_dims(np.array(bbox), axis=0)
        bbox = np.array(bbox)
        filename = self.sr_path[index].split('/')[-1].split('.')[0]


        return {'HR': shadow_img_HR, 'SR': shadow_img_SR, 'mask': shadow_img_mask,
                'Index': index, 'LR_path': self.sr_path[index],
                'sam_SR': sam_img_SR, 'sam_mask': sam_img_mask, #sam gt mask, division mask
                'bbox': bbox, 'filename': filename,}