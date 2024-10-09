import os

import matplotlib
import cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import numpy as np
import torchvision


def img_save(img, save_path="./", iteration=0, prefix="img"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Make a grid of images
    # npimg = torchvision.utils.make_grid(img, value_range=(-1, 1), padding=0, nrow=20)
    # Move the grid to the CPU and convert it to a NumPy array
    npimg = img.cpu().numpy()
    # Unnormalize the image
    npimg = npimg / 2 + 0.5
    # Transpose the image to get it in the right format for displaying
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)

def imagenet_save(img, save_path="./", iteration=0, prefix="img"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    import torchvision.transforms as transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    img = inv_normalize(img)
    # Make a grid of images
    # npimg = torchvision.utils.make_grid(img, value_range=(-1, 1), padding=0, nrow=20)
    # Move the grid to the CPU and convert it to a NumPy array
    npimg = img.cpu().numpy()
    # Unnormalize the image
    # Transpose the image to get it in the right format for displaying
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)

def img_save_batch(img, save_path, iteration=0, prefix="img"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Make a grid of images
    npimg = torchvision.utils.make_grid(img, value_range=(-1, 1), padding=0, nrow=4)
    # Move the grid to the CPU and convert it to a NumPy array
    npimg = npimg.cpu().numpy()
    # Unnormalize the image
    npimg = npimg / 2 + 0.5
    # Transpose the image to get it in the right format for displaying
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)


def resize(img, size):
    def process_single_image(image):
        if image.shape[0] < 10:  # Assuming this is for channels first (C, H, W)
            image = image.transpose(1, 2, 0)  # Convert to (H, W, C)
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            image = image.transpose(2, 0, 1)  # Convert back to (C, H, W)
        else:  # Assuming this is for channels last (H, W, C)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if necessary
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        return image

    if img.ndim == 3:  # Single image
        logging.info("Processing a single image.")
        return process_single_image(img)
    elif img.ndim == 4:  # Batch of images
        logging.info("Processing a batch of images.")
        batch = np.array([process_single_image(image) for image in img])
        return batch
    else:
        raise ValueError("Unsupported image dimensions: {}".format(img.shape))