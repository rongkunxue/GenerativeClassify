import os
import signal
import sys
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use("Agg")
import math
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
from accelerate import Accelerator
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.neural_network import register_module
from grl.utils.log import log
from improved_diffusion.unet import UNetModel
from improved_utilities import (img_save, load_pt, load_state, save_pt,
                                save_state)
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


class CombinedDataset(Dataset):
    def __init__(self, data, data_transform, value):
        self.data = data
        self.data_transform = data_transform
        self.value = value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        combined_data = torch.cat((self.data[idx], self.data_transform[idx]), dim=0)
        return combined_data, self.value[idx]


class OrginDataset(Dataset):
    def __init__(self, data, data_transform, value):
        self.data = data
        self.value = value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.value[idx]


class TransformDataset(Dataset):
    def __init__(self, data_transform, value):
        self.data = data_transform
        self.value = value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.value[idx]


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([2, 4]),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
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
    x_size = (3, 32, 32)
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
                lr=5e-4,
                iterations=1,
                batch_size=125,
                eval_freq=100,
                checkpoint_freq=100,
                dataset_path="/root/generativeencoder/exp/toy/Cifar_10/dataset",
                checkpoint_path="/root/generativeencoder/exp/toy/Cifar_10/Unet-cifar/data",
                video_save_path="/root/generativeencoder/exp/toy/Cifar_10/Unet-cifar/video",
                accelerate_checkpoint_path="/root/generativeencoder/exp/toy/Cifar_10/Unet-cifar/data",
                device=device,
            ),
        )
    )

    return config


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def pipeline():
    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device)
    log.info("config: \n{}".format(config))

    diffusion_model = DiffusionModel(config=config.diffusion_model).to(
        config.diffusion_model.device
    )

    with accelerator.main_process_first():
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.parameter.dataset_path,
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.parameter.dataset_path,
            train=False,
            download=True,
            transform=transform,
        )
        merged_dataset = ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )

    diffusion_model.model, optimizer, data_loader = accelerator.prepare(
        diffusion_model.model, optimizer, data_loader
    )

    last_iteration = load_pt(
        config.parameter.accelerate_checkpoint_path, accelerator, diffusion_model.model
    )

    import logging

    from tqdm import tqdm

    t_span = torch.linspace(0.0, 1.0, 1000)
    data_list = []
    data_transform_list = []
    value_list = []
    diffusion_model.eval()
    for batch_data, batch_label in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        with torch.no_grad():
            batch_data_transformed = diffusion_model.forward_sample(
                t_span=t_span, x=batch_data
            ).detach()
            img, img_transform, label = accelerator.gather_for_metrics(
                (batch_data, batch_data_transformed, batch_label)
            )
            data_list.append(img.cpu())
            value_list.append(label.cpu())
            data_transform_list.append(img_transform.cpu())

    data_ = torch.cat(data_list, dim=0)
    data_transform = torch.cat(data_transform_list, dim=0)
    value_ = torch.cat(value_list, dim=0)
    data_to_save = {"data": data_, "data_transform": data_transform, "value": value_}

    if accelerator.is_main_process:
        torch.save(
            data_to_save,
            "/root/generativeencoder/exp/toy/Cifar_10/dataset/cifar_transformed_test.pt",
        )


def main():
    pipeline()


if __name__ == "__main__":
    main()
