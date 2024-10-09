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
                                load_state, save_pt, save_state,imagenet_save)
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from ptflops import get_model_complexity_info

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

class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            t_tensor = torch.tensor([0.0285], dtype=torch.float32).to(x.device)
            return self.model(t_tensor, x)
        
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
                iterations=200000,
                batch_size=400,
                eval_freq=100,
                checkpoint_freq=100,
                dataset_path="/mnt/nfs3/zhangjinouwen/dataset/tiny-imagenet-200",
                checkpoint_path="/mnt/nfs3/zhangjinouwen/checkpoint/tiny_image",
                accelerate_checkpoint_path="/root/Github/generativeencoder/exp/toy/Tiny_image/data",
                video_save_path="/root/Github/generativeencoder/exp/toy/Tiny_image/video",
                device=device,
            ),
        )
    )

    return config

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def pipeline():
    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device)
    log.info(f"device{device}, config: \n{config}")
    wandb.init(
        project="imagenet_tiny",
        config=config,
        group="DDP",
        mode="offline",
    )
    diffusion_model = DiffusionModel(config=config.diffusion_model)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean= mean, std=std, inplace=True)
    ])
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    with accelerator.main_process_first():
        train_dataset=TinyImageNet(config.parameter.dataset_path, train=True,transform=transform)

    dataloader = DataLoader(
        train_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    
    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )
    
    wrapped_model = ModelWrapper(diffusion_model.model).eval()
    with torch.no_grad():
        ops, params = get_model_complexity_info(wrapped_model, (3, 64, 64), as_strings=False, print_per_layer_stat=True, verbose=True)

    diffusion_model.model, optimizer, dataloader = accelerator.prepare(
        diffusion_model.model, optimizer, dataloader
    )

    last_iteration = load_state(
        config.parameter.accelerate_checkpoint_path, accelerator
    )
    if last_iteration == -1:
        last_iteration = load_pt(
            config.parameter.checkpoint_path, accelerator, diffusion_model.model
        )

    counter = 0
    history_iteration = [-1]

    total_flops = 0  
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
                imagenet_save(
                    x_t.cpu().detach(),
                    config.parameter.video_save_path,
                    iteration,
                    f"tint_image",
                )

        diffusion_model.train()
        for batch_data,label in track(
            dataloader, description=f"Epoch {iteration}"
        ):
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
            
            total_flops +=(ops*config.parameter.batch_size*2.0)/1e9
            wandb.log(
                {
                    "epoch": iteration+1,
                    "step": counter,
                    "loss": loss.item(),
                    "max_param_val": max_param_val,
                    "flops" : total_flops*accelerator.state.num_processes/1e4,
                }
            )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_state(
                config.parameter.accelerate_checkpoint_path, accelerator, iteration
            )
        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            if accelerator.is_local_main_process:
                save_pt(
                    config.parameter.checkpoint_path,
                    diffusion_model.model,
                    optimizer,
                    iteration,
                    accelerator,
                )
def main():
    pipeline()


if __name__ == "__main__":
    main()
