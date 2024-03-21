import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.img = torch.randn(100, 3, 1024, 1024)
        self.mask = torch.randint(2, (100, 1, 1024, 1024)).float()

    def __len__(self):
        return self.img.size(dim=0)

    def __getitem__(self, idx):
        image = self.img[idx]
        label = self.mask[idx]
        return image, label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transforms.Compose([
            transforms.Resize(1024, interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.img_list = os.listdir(img_dir)
        self.mask_list = os.listdir(mask_dir)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open()
        label = self.mask[idx]
        return image, label




class TrainDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_folder = image_folder
        self.mask_folder = mask_folder

        self.img_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

        self.image_names = [name for name in os.listdir(image_folder) if name.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_folder, img_name)

        mask_name = img_name
        # mask_name = img_name.replace('.png', '_free.jpg')
        mask_path = os.path.join(self.mask_folder, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        return {
            'inp': self.img_transform(img),
            'gt': self.mask_transform(mask),
            'name': img_name
        }