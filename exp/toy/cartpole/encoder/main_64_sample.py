import os
import signal
import sys
from typing import List, Optional, Tuple, Union

import dmc2gym
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
from improved_utilities import (ReplayMemoryDataset, find_max_param_and_grad,
                                img_save, load_pt, load_state, save_pt,
                                save_state)
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
                iterations=200000,
                batch_size=50,
                eval_freq=10,
                checkpoint_freq=50,
                dataset_path="/root/generativeencoder/exp/toy/cartpole/data",
                checkpoint_path="/mnt/nfs/xuerongkun/checkpoint/cart",
                accelerate_checkpoint_path="/root/generativeencoder/exp/toy/cartpole/data/checkpoint",
                video_save_path="/root/generativeencoder/exp/toy/cartpole/video",
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
        project="cartpole",
        config=config,
        group="DDP",
        mode="offline",
    )
    diffusion_model = DiffusionModel(config=config.diffusion_model)

    # with accelerator.main_process_first():
    #     dataset = ReplayMemoryDataset(0,config.parameter.dataset_path,1)
    # dataloader = DataLoader(dataset, batch_size=config.parameter.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )
    last_iteration = load_state(
        config.parameter.accelerate_checkpoint_path, accelerator
    )
    if last_iteration == -1:
        last_iteration = load_pt(
            config.parameter.checkpoint_path, accelerator, diffusion_model.model
        )

    diffusion_model.model, optimizer = accelerator.prepare(
        diffusion_model.model, optimizer
    )

    counter = 0
    history_iteration = [-1]

    for iteration in range(config.parameter.iterations):

        diffusion_model.eval()
        t_span = torch.linspace(0.0, 1.0, 1000)
        x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=4)[
            -1, ...
        ]
        x_t = accelerator.gather_for_metrics(x_t)
        x_t = torchvision.utils.make_grid(x_t, value_range=(-1, 1), padding=0, nrow=4)
        if accelerator.is_local_main_process:
            img_save(
                x_t.cpu().detach(), config.parameter.video_save_path, iteration, f"cart"
            )


def main():
    pipeline()


if __name__ == "__main__":
    main()
