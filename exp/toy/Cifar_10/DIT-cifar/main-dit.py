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
import random
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.neural_network import register_module
from grl.neural_network.transformers.dit import DiT
from grl.utils.log import log
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.dit = DiT(
            input_size=32,
            patch_size=2,
            in_channels=3,
            hidden_size=256 * 3,
            depth=8,
            num_heads=2,
            learn_sigma=False,
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.dit(t, x, None)
        return x


register_module(MyModule, "MyModule")

device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")


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
                iterations=200000,
                batch_size=2000,
                eval_freq=50,
                checkpoint_freq=50,
                dataset_path="/root/code/data",
                checkpoint_path="/root/generativeencoder/exp/toy/DIT-cifar/checkpoint",
                video_save_path="/root/generativeencoder/exp/toy/DIT-cifar/video",
                device=device,
            ),
        )
    )

    return config


# Define transformations (if any)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def imshow(img, save_path, iteration, prefix):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Make a grid of images
    npimg = torchvision.utils.make_grid(img, value_range=(-1, 1), padding=0, nrow=20)
    # Move the grid to the CPU and convert it to a NumPy array
    npimg = npimg.cpu().numpy()
    # Unnormalize the image
    npimg = npimg / 2 + 0.5
    # Transpose the image to get it in the right format for displaying
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)


def pipeline(config):
    diffusion_model = DiffusionModel(config=config.diffusion_model).to(
        config.diffusion_model.device
    )
    diffusion_model = torch.compile(diffusion_model)

    # Create the dataset
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
    classes = train_dataset.classes
    print(classes)

    merged_dataset = ConcatDataset([train_dataset, test_dataset])
    data_loader = torch.utils.data.DataLoader(
        merged_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True
    )

    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )

    if config.parameter.checkpoint_path is not None:

        if (
            not os.path.exists(config.parameter.checkpoint_path)
            or len(os.listdir(config.parameter.checkpoint_path)) == 0
        ):
            log.warning(
                f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
            )
            last_iteration = -1
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.checkpoint_path)
                if f.endswith(".pt")
            ]
            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            checkpoint = torch.load(
                os.path.join(config.parameter.checkpoint_path, checkpoint_files[-1]),
                map_location="cpu",
            )

            from collections import OrderedDict

            diffusion_model.load_state_dict(checkpoint["model"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    counter = 0
    iteration = 0

    def render_video(
        data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""
    ):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(12, 12))

        ims = []

        for i, data in enumerate(data_list):

            grid = make_grid(
                data.view([-1, 3, 32, 32]),
                value_range=(-1, 1),
                padding=0,
                nrow=2,
            )
            grid = grid / 2.0 + 0.5
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(
                video_save_path,
                f"{prefix}_{iteration}.mp4",
            ),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    def save_checkpoint(model, optimizer, iteration):
        if not os.path.exists(config.parameter.checkpoint_path):
            os.makedirs(config.parameter.checkpoint_path)
        torch.save(
            dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                iteration=iteration,
            ),
            f=os.path.join(
                config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
            ),
        )

    history_iteration = [-1]
    optimizer.zero_grad()
    cell = 0
    for iteration in range(config.parameter.iterations):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            diffusion_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = (
                diffusion_model.sample_forward_process(t_span=t_span, batch_size=4)
                .cpu()
                .detach()
            )[-1, ...]
            imshow(x_t, config.parameter.video_save_path, iteration, "cifar")

        diffusion_model.train()

        for batch_data, batched_value in track(
            data_loader, description=f"Epoch {iteration}"
        ):
            batch_data = batch_data.to(config.device)
            if cell == 0:
                imshow(
                    batch_data, config.parameter.video_save_path, iteration, "imgdit"
                )
                cell += 1
            batched_value = batched_value.to(config.device)

            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            loss.backward()
            counter += 1
            optimizer.step()
            optimizer.zero_grad()

            log.info(f"iteration {iteration}, step {counter}, loss {loss.item()}")

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(diffusion_model, optimizer, iteration)


def main():
    config = make_config(device)
    log.info("config: \n{}".format(config))
    pipeline(config)


if __name__ == "__main__":
    main()
