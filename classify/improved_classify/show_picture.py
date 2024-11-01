import os
from easydict import EasyDict
import torch
from torch.utils.data import Dataset
import ipdb
import wandb
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
from accelerate import Accelerator
from data import build_loader
from models import build_model
from torch_tool import build_optimizer, build_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import os
import matplotlib
import cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
import numpy as np
import torchvision
from grl.utils.model_utils import save_model, load_model
from rich.progress import track
import torchvision


def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

def imagenet_save(img, save_path="./", iteration=0, prefix="img"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Mean and standard deviation for ImageNet
    mean = [0.5,0.5,0.5]
    std  = [0.5,0.5,0.5]

    # Undo the normalization
    img = unnormalize(img, mean, std)
    img = torch.clamp(img, 0, 1)  # Ensure pixel values are between 0 and 1

    # Convert tensor to numpy array
    npimg = img.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # Rearrange dimensions for plotting

    # Plotting
    plt.imshow(npimg)
    plt.axis("off")
    plt.tight_layout()

    # Save the image
    save_filename = os.path.join(save_path, f"{prefix}_{iteration}.pdf")
    save_filename_png = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_filename_png,bbox_inches="tight", pad_inches=0)
    plt.savefig(save_filename, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()


def train(config, accelerator):
    if accelerator.is_main_process:
        wandb_mode = "online" if accelerator.state.num_processes > 1 else "offline"
        wandb.init(
            project=config.PROJECT_NAME,
            config=config,
            mode=wandb_mode  
        )
        if hasattr(config, "extra"):
            wandb.run.name=config.extra
            wandb.run.save()
    accelerator.wait_for_everyone()
        
    data_loader_train, data_loader_vl, mixup_fn = build_loader(config,True)
    model = build_model(config)
    
    if (
            hasattr(config.DATA, "checkpoint_path")
            and config.DATA.checkpoint_path is not None
        ):
            load_model(
                    path=config.DATA.checkpoint_path,
                    model=model.grlEncoder.diffusionModel.model,
                    optimizer=None,
                    prefix="DiffusionModel_Pretrain",
            )
    
    if hasattr(config.DATA, "checkpoint_path") and config.DATA.checkpoint_path is not None:
        load_model(
            path=config.DATA.checkpoint_path,
            model=model,
            optimizer=None,
            prefix="GenerativeClassify",
        )
            
    else:
        raise NotImplementedError
    (
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_train,
    ) = accelerator.prepare(
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_train,
    )
        
    for data,label in data_loader_train:
        img = data[0].unsqueeze(0)
        t_span = torch.linspace(0.0, 1.0, 100, device=data.device)
        x_t = model.grlEncoder.diffusionModel.forward_sample_process(
            x=img, t_span=t_span, with_grad=False
        )
        for i in range(x_t.size(0)):
            frame = x_t[i][0]  # Get the i-th frame
            imagenet_save(frame, save_path="/root/Github/exp/tiny", iteration=i, prefix="img")
            
        break
        
                


                    