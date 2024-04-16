from typing import Union, Any

import cv2
import numpy as np
import os

# Directory paths
train_a_dir = '../dataset/SRD_sam_mask_B/train/train_A/'
train_b_dir = '../dataset/SRD_sam_mask_B/train/train_B/'
result_dir = 'boundary_result/result/'
result_inverse_dir = 'boundary_result/result_inverse/'
gradient_dir = 'boundary_result/gradient/'

# Create the result directories if they don't exist
os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_inverse_dir, exist_ok=True)

# Get the list of image files in train_A directory
image_files = os.listdir(train_a_dir)

# Process each image file
for image_file in image_files:
    # Read the image
    image_path = os.path.join(train_a_dir, image_file)
    image = cv2.imread(image_path)

    # Read the corresponding mask
    mask_file = image_file.replace('.jpg', '.png')
    mask_path = os.path.join(train_b_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to match the image dimensions
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Normalize the mask values to be between 0 and 1
    mask = mask / 255.0

    # Create the result image
    result = image.copy()
    for i in range(3):
        result[:, :, i] = image[:, :, i] * mask

    # Create the inverse result image
    mask_inverse = 1 - mask
    result_inverse = np.zeros_like(image)
    for i in range(3):
        result_inverse[:, :, i] = image[:, :, i] * mask_inverse

    # Save the result images
    # result_path = os.path.join(result_dir, image_file)
    # result_inverse_path = os.path.join(result_inverse_dir, image_file)
    # cv2.imwrite(result_path, result)
    # cv2.imwrite(result_inverse_path, result_inverse)

    # Convert the result to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply Sobel operator to compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    orientation = np.arctan2(sobely, sobelx) * (180 / np.pi)

    # Normalize the orientation values to the range [0, 255]
    orientation_normalized: Union[float, Any] = (orientation + 180) / 2
    orientation_normalized.astype(np.uint8)

    # Save the gradient orientation map
    gradient_path = os.path.join(gradient_dir, image_file)
    cv2.imwrite(gradient_path, orientation_normalized)