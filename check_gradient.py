import torch
import torch.nn.functional as F

def apply_low_pass_filter(mask: object, kernel_size: object = 3) -> object:
    # Create a Gaussian kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.double)

    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(mask.device)

    # Apply the low-pass filter using convolution
    filtered_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)

    return filtered_mask

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

soft_mask = torch.randn(1, 1, 256, 256, requires_grad=True, dtype=torch.double)
shadow_input = torch.randn(1, 3, 256, 256, requires_grad=True, dtype=torch.double)

def func(soft_mask, shadow_input):
    return gradient_orientation_loss(soft_mask, shadow_input)

test_passed = torch.autograd.gradcheck(func, (soft_mask, shadow_input), eps=1e-6, atol=1e-4)

if test_passed:
    print("Gradients are computed correctly!")
else:
    print("Gradients are not computed correctly.")