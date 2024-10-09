import os
import signal
import sys
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import math
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import wandb
from accelerate import Accelerator
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.neural_network import register_module
from grl.utils.log import log
from improved_diffusion.unet import UNetModel
from improved_utilities import (find_max_param_and_grad, img_save, load_pt,
                                load_state, save_pt, save_state,img_save_batch)
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([4, 8]),
            dropout=0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=False,
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.unet(x, t, condition)
        return x


register_module(MyModule, "MyModule")


def make_config(device):
    x_size = (3, 64, 64)
    config = EasyDict(
        dict(
            device=device,
            diffusion_model=dict(
                device=device,
                x_size=x_size,
                alpha=1.0,
                solver=dict(
                    type="ODESolver",
                    args=dict(
                        library="torchdyn",
                        atol=1e-7,
                        rtol=1e-7,
                    ),
                ),
                path=dict(
                    type="gvp",
                ),
                model=dict(
                    type="velocity_function",
                    args=dict(
                        backbone=dict(
                            type="MyModule",
                            args={},
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=1e-4,
                iterations=200000,
                batch_size=50,
                eval_freq=10,
                checkpoint_freq=50,
                dataset_path="/mnt/nfs/xuerongkun/dataset/Words/words_dataset_64x64.pt",
                checkpoint_path="/mnt/nfs/xuerongkun/checkpoint/Words",
                accelerate_checkpoint_path="/root/generativeencoder/exp/toy/Best_word/data/checkpoint",
                video_save_path="/root/generativeencoder/exp/toy/Best_word/video1",
                device=device,
            ),
        )
    )

    return config


def pipeline():

    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device)
    log.info(f"device{device}, config: \n{config}")
    wandb.init(
        project="words_end_2",
        config=config,
        group="DDP",
        mode="offline",
    )
    diffusion_model = DiffusionModel(config=config.diffusion_model)

    # Create the dataset
    class CustomImageDataset(Dataset):
        def __init__(self, dataset_path, transform=None):
            # Load the dataset
            data = torch.load(dataset_path)
            self.images = data["images"]
            self.transform = transform
            self.labels = data["labels"]
            self.centers = data["centers"]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            center = self.centers[idx]
            return image, label, center

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    with accelerator.main_process_first():
        dataset = CustomImageDataset(config.parameter.dataset_path, transform)

    dataloader = DataLoader(
        dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )

    diffusion_model.model, optimizer, dataloader = accelerator.prepare(
        diffusion_model.model, optimizer, dataloader
    )

    last_iteration = load_state(
        config.parameter.accelerate_checkpoint_path, accelerator
    )
    # if last_iteration == -1:
    #     last_iteration = load_pt(
    #         config.parameter.checkpoint_path, accelerator, diffusion_model.model
    #     )

    counter = 0
    history_iteration = [-1]

    image_1_path = "/root/generativeencoder/exp/toy/Best_word/data/image1.png"
    image_1 = Image.open(image_1_path)
    image_2_path = "/root/generativeencoder/exp/toy/Best_word/data/image2.png"
    image_2 = Image.open(image_2_path)
    image_3_path = "/root/generativeencoder/exp/toy/Best_word/data/image3.png"
    image_3 = Image.open(image_3_path)
    image_4_path = "/root/generativeencoder/exp/toy/Best_word/data/image4.png"
    image_4 = Image.open(image_4_path)
    image_5_path = "/root/generativeencoder/exp/toy/Best_word/data/image5.png"
    image_5 = Image.open(image_5_path)
    image_6_path = "/root/generativeencoder/exp/toy/Best_word/data/image6.png"
    image_6 = Image.open(image_6_path)
    # Apply the transform to the imag    
    image_1_tensor = transform(image_1).unsqueeze(0).to(device)
    image_2_tensor = transform(image_2).unsqueeze(0).to(device)
    image_3_tensor = transform(image_3).unsqueeze(0).to(device)
    image_4_tensor = transform(image_4).unsqueeze(0).to(device)
    image_5_tensor = transform(image_5).unsqueeze(0).to(device)
    image_6_tensor = transform(image_6).unsqueeze(0).to(device)


    def transform_images_one_picture_one_more(diffusion_model, image_1_tensor, config,device,name=1):
        diffusion_model.eval()
        with torch.no_grad():
            t_span = torch.linspace(0.0, 1.0, 1000).to(device)
            transform = diffusion_model.forward_sample(t_span=t_span, x=image_1_tensor.to(device)).detach()
            x_t = diffusion_model.sample_forward_process(t_span=t_span, x_0=transform)[-1, ...]
            img_save(x_t[0], save_path=config.parameter.video_save_path, iteration=name, prefix="first")
            transform = diffusion_model.forward_sample(t_span=t_span, x=x_t.to(device)).detach()
            x_t = diffusion_model.sample_forward_process(t_span=t_span, x_0=transform)[-1, ...]
            img_save(x_t[0], save_path=config.parameter.video_save_path, iteration=name, prefix="second")

    def transform_images_one_picture(diffusion_model, image_1_tensor, config,device,name=1):
            diffusion_model.eval()
            with torch.no_grad():
                t_span = torch.linspace(0.0, 1.0, 10000).to(device)
                transform = diffusion_model.forward_sample(t_span=t_span, x=image_1_tensor.to(device)).detach()
                x_t = diffusion_model.sample_forward_process(t_span=t_span, x_0=transform)[-1, ...]
                img_save(x_t[0], save_path=config.parameter.video_save_path, iteration=name, prefix="img")

    def transform_images(diffusion_model, image_1_tensor, image_2_tensor, config,device,name=1):
        diffusion_model.eval()
        with torch.no_grad():
            t_span = torch.linspace(0.0, 1.0, 1000).to(device)
            transformed_1 = diffusion_model.forward_sample(t_span=t_span, x=image_1_tensor.to(device)).detach()
            transformed_2 = diffusion_model.forward_sample(t_span=t_span, x=image_2_tensor.to(device)).detach()
            lamda = torch.linspace(0, 1, 101).view(101, 1, 1, 1).to(device)
            transform = transformed_1 * (1 - lamda) + transformed_2 * lamda
            x_t = diffusion_model.sample_forward_process(t_span=t_span, x_0=transform)[-1, ...]
            img_save_batch(x_t.cpu().detach(), config.parameter.video_save_path, name, row=10)

    # transform_images(diffusion_model, image_3_tensor, image_4_tensor, config, device=device,name=34)
    # transform_images(diffusion_model, image_3_tensor, image_6_tensor, config, device=device,name=36)
    # transform_images(diffusion_model, image_4_tensor, image_5_tensor, config, device=device,name=45)
    # transform_images(diffusion_model, image_5_tensor, image_6_tensor, config, device=device,name=56)
    transform_images_one_picture(diffusion_model, image_3_tensor, config, device=device,name=3000)
def main():
    pipeline()


if __name__ == "__main__":
    main()

