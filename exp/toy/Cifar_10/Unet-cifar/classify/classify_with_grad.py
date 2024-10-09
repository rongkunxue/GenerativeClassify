import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
import wandb
from accelerate import Accelerator
from improved_utilities import create_data_loader, img_save_batch, save_pt
from rich.progress import BarColumn, Progress, TextColumn
from torch.utils.data import Dataset
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
import wandb
from accelerate import Accelerator
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.neural_network import register_module
from grl.utils.log import log
from improved_diffusion.unet import UNetModel
from improved_utilities import (find_max_param_and_grad, img_save, load_pt,
                                load_state, save_pt, save_state)
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import logging

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
    initialize(config_path="conf")
    cfg = compose(config_name="classify_with_grad", overrides=[f"device={device}"])
    print(OmegaConf.to_yaml(cfg))
    return cfg

class SimpleModel(nn.Module):
    def __init__(self, size):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * size * size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

def initialize_model(name, size=224):
    if name == "Resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
        return model
    elif name == "linear":
        model = SimpleModel(size)
        return model
    else:
        return -1

def train(
    accelerator,diffusion_model,model, loader, optimizer, loss_fn, epoch, name
):
    model.train()
    training_loss = 0.0
    for images, labels in track(loader,disable=not accelerator.is_local_main_process):
        t_span = torch.linspace(0.0, 1.0, 100)
        images=diffusion_model.forward_sample(x=images, t_span=t_span,with_grad=True)
        predict_label = model(images)
        loss = loss_fn(predict_label, labels)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        training_loss += loss.item()
        wandb.log(
            {
                f"{name}/epoch": + 1,
                f"{name}/training_loss": loss,
            },
            commit=True,
        )
    return 0

def eval(accelerator,diffusion_model,model, data_loader,epoch,name):
    model.eval()
    diffusion_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            t_span = torch.linspace(0.0, 1.0, 100)
            images=diffusion_model.forward_sample(x=images, t_span=t_span)
            outputs = model(images)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total * 100
    wandb.log(
        {
            f"{name}/epoch": + 1,
            f"{name}/accuracy":  accuracy,
        },
        commit=False,
    )
    return 0

def pipeline():
    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device
    # Initialize wandb
    wandb.init(project="cifar10_classification_grad_6")

    config=make_config(device)
    # Data loaders using the function
    diffusion_model = DiffusionModel(config=config.diffusion_model)
    load_pt( config.parameter.checkpoint_path, accelerator, diffusion_model.model)
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),     
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
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
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    # Initialize models, optimizers and loss functions
    models = {
        "transform": initialize_model("linear", 32),
    }

    optimizer = torch.optim.Adam([
        {'params': models["transform"].parameters()},
        {'params': diffusion_model.model.parameters()},
    ],lr=5e-4)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_loaders = {
        "transform": train_data_loader,
    }

    test_loaders = {
        "transform": test_data_loader,
    }

    epochs = 500000

    for name in models.keys():
        diffusion_model.model, models[name], optimizer, train_loaders[name], test_loaders[name] = accelerator.prepare(
            diffusion_model.model,
            models[name],
            optimizer,
            train_loaders[name],
            test_loaders[name]
        )

    # load_state(config.parameter.accelerate_checkpoint_path,accelerator)

    for epoch in range(epochs):
        for name in models.keys():
            train(
                accelerator,
                diffusion_model,
                models[name],
                train_loaders[name],
                optimizer,
                loss_fn,
                epoch,
                name,
            )
        eval(accelerator,diffusion_model,models[name],train_loaders[name],epoch,"train_data")
        eval(accelerator,diffusion_model,models[name],test_loaders[name],epoch,"test_data")


        if (epoch + 1) % 5 == 0:  
            save_state(config.parameter.accelerate_checkpoint_path, accelerator,  epoch)
            # save_pt(
            #     f"/root/generativeencoder/exp/toy/Cifar_10/Unet-cifar/classify/classify_{name}",
            #     models[name],
            #     optimizer,
            #     epoch,
            #     accelerator,
            # )     
        # if (epoch + 1) % 50 == 0:  
        #     if accelerator.is_local_main_process:
        #         save_pt(
        #             config.parameter.checkpoint_path,
        #             diffusion_model.model,
        #             optimizer,
        #             epoch,
        #             accelerator,
        #             "grad",
        #         )

    # Finish the wandb run
    wandb.finish()

def main():
    pipeline()

if __name__ == "__main__":
    main()