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

class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


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
    def __init__(self, size,classes):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * size * size, classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

def initialize_model(name, size=224,classes=200):
    if name == "Resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, classes)
        return model
    elif name == "linear":
        model = SimpleModel(size,classes)
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
    wandb.init(project="tiny_imagenet_classifiy_grad_1",mode="offline")

    config=make_config(device)
    # Data loaders using the function
    diffusion_model_pretrain = DiffusionModel(config=config.diffusion_model)
    load_pt(config.parameter.checkpoint_path, accelerator, diffusion_model_pretrain.model)
    diffusion_model_no_pretrain = DiffusionModel(config=config.diffusion_model)
    test_transform = create_transform(32, is_training=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = create_transform(32, is_training=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    with accelerator.main_process_first():
        train_dataset = TinyImageNet(config.parameter.dataset_path, train=True,transform=train_transform)
        test_dataset = TinyImageNet(config.parameter.dataset_path, train=False,transform=test_transform)
    
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
        shuffle=False,
        num_workers=config.parameter.numerworkers,
        pin_memory=True,
    )
    # Initialize models, optimizers and loss functions
    models = {
        "pretrain": initialize_model("linear",config.parameter.imagesize,config.parameter.classes),
        "no_pretrain": initialize_model("linear",config.parameter.imagesize,config.parameter.classes),
    }
    diffusion_model={
        "pretrain": diffusion_model_pretrain,
        "no_pretrain": diffusion_model_no_pretrain,
    }

    optimizers = {
    "pretrain": torch.optim.Adam([
        {'params': models["pretrain"].parameters()},
        {'params': diffusion_model["pretrain"].model.parameters()},
    ], lr=config.parameter.lr),
    "no_pretrain": torch.optim.Adam([
        {'params': models["no_pretrain"].parameters()},
        {'params': diffusion_model["no_pretrain"].model.parameters()},
    ], lr=config.parameter.lr),
    }
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loaders = {
        "pretrain": train_data_loader,
        "no_pretrain": train_data_loader,
    }
    test_loaders = {
       "pretrain": test_data_loader,
       "no_pretrain": test_data_loader,
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
    wandb.finish()

def main():
    pipeline()

if __name__ == "__main__":
    main()