import os
import dmc2gym
from typing import List, Optional, Tuple, Union
import signal
import sys
from torch.utils.data import Dataset, DataLoader
import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
import cv2
from improved_utilities import resize, SampleData, ReplayMemoryDataset
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from easydict import EasyDict
from matplotlib import animation

from accelerate import Accelerator

from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils.log import log
from grl.neural_network import register_module

from functools import partial
from improved_diffusion.unet import UNetModel
import wandb
from improved_utilities import (
    save_state,
    save_pt,
    load_pt,
    load_state,
    img_save,
    find_max_param_and_grad,
    ReplayMemoryDataset,
)


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
                lr=1e-4,
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

    # if accelerator.is_local_main_process:
    #     transform = transforms.Compose(
    #         [
    #             transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
    #             transforms.ToPILImage(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ]
    #     )
    #     import os
    #     os.environ["MUJOCO_GL"] = "egl"
    #     env = dmc2gym.make(domain_name='cartpole', task_name='balance', seed=1,from_pixels=True,visualize_reward=False)
    #     myData=SampleData(0,'/mnt/nfs/xuerongkun/dataset/Carpole',10,1050,(3,64,64),(1,))
    #     myData.start_sample_game(env,transform)
    # accelerator.wait_for_everyone()

    logging.info(f"device{device}, config: \n{config}")
    wandb.init(
        project="cartpole",
        config=config,
        group="DDP",
        mode="offline",
    )
    diffusion_model = DiffusionModel(config=config.diffusion_model)
    # last_iteration=load_pt(config.parameter.checkpoint_path,accelerator,diffusion_model.model)

    with accelerator.main_process_first():
        dataset = ReplayMemoryDataset(0, config.parameter.dataset_path, 1)
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

    counter = 0
    history_iteration = [-1]

    for iteration in range(config.parameter.iterations):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            diffusion_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = diffusion_model.sample_forward_process(t_span=t_span, batch_size=4)[
                -1, ...
            ]
            x_t = accelerator.gather_for_metrics(x_t)
            x_t = torchvision.utils.make_grid(
                x_t, value_range=(-1, 1), padding=0, nrow=4
            )
            if accelerator.is_local_main_process:
                img_save(
                    x_t.cpu().detach(),
                    config.parameter.video_save_path,
                    iteration,
                    f"word",
                )

        diffusion_model.train()
        for batch in track(dataloader, description=f"Epoch {iteration}"):
            batch_data = batch["state"].squeeze(1)
            img_save(batch_data[0])
            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            optimizer.zero_grad()
            accelerator.backward(loss)
            counter += 1
            optimizer.step()
            max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(
                diffusion_model.model
            )
            if accelerator.is_local_main_process:
                log.info(f"iteration {iteration}, step {counter}, loss {loss.item()}")
            wandb.log(
                {
                    "iteration": iteration,
                    "step": counter,
                    "loss": loss.item(),
                    "max_param_val": max_param_val,
                }
            )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_state(
                config.parameter.accelerate_checkpoint_path, accelerator, iteration
            )
        if (iteration + 1) % 500 == 0:
            if accelerator.is_local_main_process:
                save_pt(
                    config.parameter.checkpoint_path,
                    diffusion_model.model,
                    optimizer,
                    iteration,
                )


def main():
    pipeline()


if __name__ == "__main__":
    main()
