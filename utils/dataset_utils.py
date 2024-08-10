import torch
import os
import math
from torchvision import transforms

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor


### mix two images
class MixUp_AUG:
    def __init__(self):
        self.dist = torch.distributions.beta.Beta(torch.tensor([1.2]), torch.tensor([1.2]))

    def aug(self, rgb_gt, rgb_noisy, rgb_mask):
        bs = rgb_gt.size(0)
        indices = torch.randperm(bs)
        rgb_gt2 = rgb_gt[indices]
        rgb_noisy2 = rgb_noisy[indices]
        # gray_mask2 = gray_mask[indices]
        # gray_contour2 = gray_mask[indices]
        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()

        rgb_gt    = lam * rgb_gt + (1-lam) * rgb_gt2
        rgb_noisy = lam * rgb_noisy + (1-lam) * rgb_noisy2
        # gray_mask = lam * gray_mask + (1-lam) * gray_mask2
        # gray_mask = torch.where(gray_mask>0.01, torch.ones_like(gray_mask), torch.zeros_like(gray_mask))
        # gray_contour = lam * gray_contour + (1-lam) * gray_contour2
        return rgb_gt, rgb_noisy

def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    img_resize = transforms.Resize((1024, 1024))
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    small_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            small_data.append(cimgdata)
            cimgdata_ = img_resize(cimgdata)
            starts.append((hs, ws))
            split_data.append(cimgdata_)
    return split_data, small_data, starts


def splitbbox(image, bbox, crop_size=128, overlap_size=64):
    _, C, H, W = image.shape

    # Resize bounding boxes to match image size

    x1, y1, x2, y2 = bbox[0]
    width_scale = W / 1024
    height_scale = H / 1024
    x1 = int(x1 * width_scale)
    y1 = int(y1 * height_scale)
    x2 = int(x2 * width_scale)
    y2 = int(y2 * height_scale)
    resized_bboxes=[x1, y1, x2, y2]

    # Calculate crop starts
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)

    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)

    cropped_bboxes = []
    # Crop image and bounding boxes
    starts = []
    scale_factor = 1024 / crop_size
    for hs in hstarts:
        for ws in wstarts:
            starts.append((hs, ws))

            x1, y1, x2, y2 = resized_bboxes

            # Check if bbox intersects with the crop
            if (x1 < ws + crop_size and x2 > ws and
                    y1 < hs + crop_size and y2 > hs):
                # Adjust coordinates relative to the crop
                new_x1 = max(0, x1 - ws)
                new_y1 = max(0, y1 - hs)
                new_x2 = min(crop_size, x2 - ws)
                new_y2 = min(crop_size, y2 - hs)

                crop_bboxes = torch.tensor([[
                    int(new_x1 * scale_factor),
                    int(new_y1 * scale_factor),
                    int(new_x2 * scale_factor),
                    int(new_y2 * scale_factor)
                ]], device=image.device)
            else:
                crop_bboxes=torch.tensor([[0, 0, 0, 0]], device=image.device)

            cropped_bboxes.append(crop_bboxes)

    return cropped_bboxes, starts

def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6))
    return score

def mergeimage(split_data, starts, crop_size = 128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W)).to(split_data[0].device)
    merge_img = torch.zeros((B, C, H, W)).to(split_data[0].device)
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    scoremap = scoremap.to(split_data[0].device)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img