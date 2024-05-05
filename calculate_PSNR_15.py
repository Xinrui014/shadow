import os


def compare_image_names(dir1, dir2):
    # Get the list of image names in each directory (without file extensions)
    images1 = set(os.path.splitext(image)[0].replace('_free', '') for image in os.listdir(dir1))
    images2 = set(os.path.splitext(image)[0].replace('_free', '') for image in os.listdir(dir2))

    # Find the different image names
    different_images = images1.symmetric_difference(images2)

    # Count the number of different image names
    count = len(different_images)
    excude_name_1 = []
    excude_name_2 = []

    # Print the different image names and their count
    if different_images:
        print(f"Found {count} different image names:")
        for image in different_images:
            if image in images1:
                excude_name_1.append(image)
                print(f"{image} (in dir1)")
            else:
                excude_name_2.append(image)
                print(f"{image} (in dir2)")
    else:
        print("No differences found in image names.")

    return excude_name_1, excude_name_2

old_test_A = "dataset/SRD_DHAN_mask_old/test/test_A"
new_test_A = "dataset/SRD_DHAN_mask_B/test/test_A"
excude_name_1, excude_name_2 = compare_image_names(old_test_A, new_test_A)
ex = excude_name_1

with open("/home/xinrui/projects/ShadowDiffusion/experiments_lightning/train_ShadowDiffusion/version_5_old_DHAN_mask/199_mean_filter/PSNR_SSIM_list.log", "r") as f:
    log_data = f.readlines()

psnr_values = []
ssim_values = []
i = 0
for line in log_data:
    line = line.strip()
    if "_PSNR:" in line:
        image_name = line.split("_PSNR:")[0]
        if image_name not in ex:
            i += 1
            psnr = float(line.split(":")[1])
            psnr_values.append(psnr)
    if "_SSIM:" in line:
        image_name = line.split("_SSIM:")[0]
        if image_name not in ex:

            ssim = float(line.split(":")[1])
            ssim_values.append(ssim)

mean_PSNR = sum(psnr_values) / len(psnr_values)
print(mean_PSNR)
mean_SSIM = sum(ssim_values) / len(ssim_values)
print(mean_SSIM)

print(i)
print(len(psnr_values))