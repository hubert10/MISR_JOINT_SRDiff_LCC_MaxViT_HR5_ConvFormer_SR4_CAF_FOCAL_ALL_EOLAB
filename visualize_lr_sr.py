import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_multiband_image(path):
    """Reads a 4-channel image and returns only the RGB channels"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # shape: H x W x 4
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.shape[2] != 4:
        raise ValueError(f"Image does not have 4 channels: {path}")
    img_rgb = img[:, :, :3]  # R, G, B (drop NIR)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
    return img_rgb


def plot_time_series(lr_folder, sr_folder):
    lr_images = sorted(
        [
            os.path.join(lr_folder, f)
            for f in os.listdir(lr_folder)
            if f.endswith((".png", ".tif", ".jpg"))
        ]
    )
    sr_images = sorted(
        [
            os.path.join(sr_folder, f)
            for f in os.listdir(sr_folder)
            if f.endswith((".png", ".tif", ".jpg"))
        ]
    )

    n = min(len(lr_images), len(sr_images))
    plt.figure(figsize=(12, 4 * n))

    for i in range(n):
        lr_img = read_multiband_image(lr_images[i])
        sr_img = read_multiband_image(sr_images[i])

        # Plot LR
        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(lr_img)
        plt.title(f"LR Image {i+1}")
        plt.axis("off")

        # Plot SR
        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(sr_img)
        plt.title(f"SR Image {i+1}")
        plt.axis("off")

    plt.show()


lr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\LR\\D022_2021\\Z1_AA\\sen\\"
sr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\SR\\D022_2021\\Z1_AA\\sen\\"

# lr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\LR\\D015_2020\\Z16_UA\\sen\\"
# sr_folder = "D:\\kanyamahanga\\Datasets\\MISR_S2_Aer_LCC_x10_JOINT_SRDiff_SEG_SegFormer_HR_ConvFormer_SR_NIR_OPT_DATA_AUG_LPIPS\\SR\\D015_2020\\Z16_UA\\sen\\"


plot_time_series(lr_folder, sr_folder)
