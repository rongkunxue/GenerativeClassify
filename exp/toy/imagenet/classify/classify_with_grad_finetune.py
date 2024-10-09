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
from timm.data.transforms_factory import create_transform

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
    initialize(config_path="conf")
    cfg = compose(config_name="classify_with_grad", overrides=[f"device={device}"])
    print(OmegaConf.to_yaml(cfg))
    return cfg

class SimpleModel(nn.Module):
    def __init__(self, size):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * size * size, 1000)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

def initialize_model(name, size=224):
    if name == "Resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 1000)
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
                f"{name}/epoch": epoch,
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
            if total==0 and accelerator.is_main_process:
                img_save_batch(images,"/root/generativeencoder/exp/toy/imagenet/video",epoch,f"{name}_real")
            images=diffusion_model.forward_sample(x=images, t_span=t_span)
            if total==0 and accelerator.is_main_process:
                img_save_batch(images,"/root/generativeencoder/exp/toy/imagenet/video",epoch,name)
            outputs = model(images)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total * 100
    wandb.log(
        {
            f"{name}/epoch": epoch,
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
    wandb.init(project="imagenet_classifiy_grad_2")

    config=make_config(device)
    # Data loaders using the function
    diffusion_model_pretrain = DiffusionModel(config=config.diffusion_model)
    load_pt(config.parameter.checkpoint_path, accelerator, diffusion_model_pretrain.model)
    diffusion_model_no_pretrain = DiffusionModel(config=config.diffusion_model)
    test_transform = create_transform(64, is_training=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = create_transform(64, is_training=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    with accelerator.main_process_first():
        train_dataset = torchvision.datasets.ImageFolder(os.path.join(config.parameter.dataset_path, 'train'),train_transform)
        test_dataset = torchvision.datasets.ImageFolder(os.path.join(config.parameter.dataset_path, 'train'),test_transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=config.parameter.numerworkers,
        pin_memory=True,
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=config.parameter.numerworkers,
        pin_memory=True,
    )
    # Initialize models, optimizers and loss functions
    models = {
        "pretrain": initialize_model("linear",config.parameter.imagesize),
        # "no_pretrain": initialize_model("linear",config.parameter.imagesize),
    }
    diffusion_model={
        "pretrain": diffusion_model_pretrain,
        # "no_pretrain": diffusion_model_no_pretrain,
    }

    optimizers = {
    "pretrain": torch.optim.Adam([
        {'params': models["pretrain"].parameters()},
        {'params': diffusion_model["pretrain"].model.parameters()},
    ], lr=config.parameter.lr),
    # "no_pretrain": torch.optim.Adam([
    #     {'params': models["no_pretrain"].parameters()},
    #     {'params': diffusion_model["no_pretrain"].model.parameters()},
    # ], lr=config.parameter.lr),
    
    }
    loss_fn = torch.nn.CrossEntropyLoss()

    train_loaders = {
        "pretrain": train_data_loader,
        # "no_pretrain": train_data_loader,
    }

    test_loaders = {
       "pretrain": test_data_loader,
    #    "no_pretrain": test_data_loader,
    }

    epochs = config.parameter.iterations

    for name in models.keys():
        diffusion_model[name].model, models[name], optimizers[name], train_loaders[name], test_loaders[name] = accelerator.prepare(
             diffusion_model[name].model,
            models[name],
            optimizers[name],
            train_loaders[name],
            test_loaders[name]
        )

    for epoch in range(epochs):
        for name in models.keys():
            # eval(accelerator,diffusion_model[name],models[name],test_loaders[name],epoch,f"{name}_test_data")
            train(
                accelerator,
                diffusion_model[name],
                models[name],
                train_loaders[name],
                optimizers[name],
                loss_fn,
                epoch,
                name,
            )
            save_state(config.parameter.accelerate_checkpoint_path, accelerator,  epoch)

        if (epoch + 1) % 2 == 0:  
            eval(accelerator,diffusion_model[name],models[name],test_loaders[name],epoch,f"{name}_test_data")
            # eval(accelerator,diffusion_model[name],models[name],train_loaders[name],epoch,f"{name}_train_data")
    # Finish the wandb run
    wandb.finish()

def main():
    pipeline()

if __name__ == "__main__":
    main()