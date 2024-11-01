import os
import torch
import random
import numpy as np
from accelerate import Accelerator
from torch.utils.data import Dataset
from rich.progress import track
import torchvision
import wandb

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from grl.utils.log import log
from grl.utils.model_utils import save_model
from grl.generative_models.metric import compute_likelihood
from data import build_loader
from models import build_model
from torch_tool import build_optimizer, build_scheduler

# Model loading function
def load_model(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, 
               prefix="checkpoint", nth: int = 1) -> int:
    """
    Load model state_dict, optimizer state_dict, and training iteration from disk.
    Select every nth checkpoint file named as 'prefix_iteration.pt'.
    
    Args:
        path (str): Path to load the model from.
        model (torch.nn.Module): Model to load.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load. Defaults to None.
        prefix (str): Prefix of the checkpoint file. Defaults to "checkpoint".
        nth (int): Select every nth checkpoint file. Defaults to 1.
    
    Returns:
        int: The iteration of the loaded checkpoint.
    """
    last_iteration = -1

    if not os.path.exists(path) or not os.listdir(path):
        log.warning(f"Checkpoint path {path} does not exist or is empty")
        return last_iteration

    # Get all checkpoint files that match the prefix and end with '.pt'
    checkpoint_files = sorted(
        [f for f in os.listdir(path) if f.endswith(".pt") and f.startswith(prefix)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
    )

    # Filter for every nth checkpoint file
    nth_index = min(len(checkpoint_files), nth) - 1

    if not checkpoint_files:
        log.warning(f"No checkpoint files found in {path}")
        return last_iteration

    checkpoint_file = os.path.join(path, checkpoint_files[nth_index])

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    last_iteration = checkpoint.get("iteration", -1)
    
    # Remove unnecessary prefixes from the model state dict
    ori_state_dict = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(ori_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

    log.warning(f"{last_iteration}_checkpoint file has been loaded")
    return last_iteration


# # Utility class to track averages
# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, name, fmt=":f"):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         """Reset all values to zero."""
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         """Update the meter with a new value."""
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         """String representation of the meter."""
#         return f"{self.name} {self.val{self.fmt}} ({self.avg{self.fmt}})"


# Main function to perform picture analysis
def picture_analysis(config, accelerator):
    """
    Perform picture analysis using pre-trained models and calculate log likelihood on train and validation sets.

    Args:
        config: Configuration for the training process.
        accelerator: Accelerator for distributed training.
    """
    if config.TRAIN.method == "Pretrain":
        data_loader_train, data_loader_val, mixup_fn = build_loader(config, if_analyse=True)
        real_model = build_model(config)

        # Iterate over checkpoints (nth)
        for i in range(0, 20):
            import copy
            model = copy.deepcopy(real_model)
            optimizer = build_optimizer(config, model)

            # Load the diffusion model from checkpoint
            diffusion_model_train_epoch = load_model(
                path=config.DATA.checkpoint_path,
                model=model.grlEncoder.diffusionModel.model,
                optimizer=None,
                prefix="DiffusionModel_Pretrain",
                nth=i,
            )

            # Prepare the model and data loaders with accelerator
            model.grlEncoder.diffusionModel.model, data_loader_train, data_loader_val, optimizer = accelerator.prepare(
                model.grlEncoder.diffusionModel.model, data_loader_train, data_loader_val, optimizer
            )

            j_train_mean = []
            j_val_mean = []
            model.eval()
            # Perform log likelihood calculation for two iterations
            for j in range(2):
                logp_train_list = []
                logp_val_list = []

                # Train set likelihood
                count = 0
                for samples, targets in track(data_loader_train, disable=not accelerator.is_local_main_process):
                    with torch.no_grad():
                        log_p = compute_likelihood(
                            model=model.grlEncoder.diffusionModel,
                            x=samples,
                            t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                            using_Hutchinson_trace_estimator=True,
                        
                        )
                        logp_train_list.append(log_p)
                        count += 1
                        if count > 40:
                            break

                # Validation set likelihood
                count = 0
                for samples, targets in track(data_loader_val, disable=not accelerator.is_local_main_process):
                    with torch.no_grad():
                        log_p = compute_likelihood(
                            model=model.grlEncoder.diffusionModel,
                            x=samples,
                            t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                            using_Hutchinson_trace_estimator=True,
                        )
                        logp_val_list.append(log_p)
                        count += 1
                        if count > 40:
                            break

                # Calculate mean likelihoods
                logp_train_mean =  torch.stack(logp_train_list).mean().item()/(32.0*32.0)
                logp_val_mean = torch.stack(logp_val_list).mean().item()/(32.0*32.0)

                j_train_mean.append(logp_train_mean)
                j_val_mean.append(logp_val_mean)

            # Calculate train and validation statistics
            train_mean = np.array(j_train_mean).mean()
            train_std = np.array(j_train_mean).std()
            val_mean = np.array(j_val_mean).mean()
            val_std = np.array(j_val_mean).std()

            # Log the results to wandb
            wandb.log({
                "train_mean": train_mean,
                "train_std": train_std,
                "val_mean": val_mean,
                "val_std": val_std,
                "nth": i,
                "diffusion_model_train_epoch": diffusion_model_train_epoch, 
            })
            
@torch.no_grad()
# Main function to perform picture analysis
def strightness_analysis(config, accelerator):
    """
    Perform picture analysis using pre-trained models and calculate log likelihood on train and validation sets.

    Args:
        config: Configuration for the training process.
        accelerator: Accelerator for distributed training.
    """
    from grl.generative_models.metric import compute_straightness
    if config.TRAIN.method == "Pretrain":
        real_model = build_model(config)

        # Iterate over checkpoints (nth)
        for i in range(0, 20):
            import copy
            model = copy.deepcopy(real_model)

            # Load the diffusion model from checkpoint
            diffusion_model_train_epoch = load_model(
                path=config.DATA.checkpoint_path,
                model=model.grlEncoder.diffusionModel.model,
                optimizer=None,
                prefix="DiffusionModel_Pretrain",
                nth=i,
            )

            # Prepare the model and data loaders with accelerator
            model.grlEncoder.diffusionModel.model = accelerator.prepare(
                model.grlEncoder.diffusionModel.model
            )

            j_train_mean = []
            model.eval()
            # Perform log likelihood calculation for two iterations
            for j in range(10):
                stright=compute_straightness(model.grlEncoder.diffusionModel,128)
                j_train_mean.append(stright.item())
            # Calculate train and validation statistics
            mean = np.array(j_train_mean).mean()
            std = np.array(j_train_mean).std()
            # Log the results to wandb
            wandb.log({
                "str/mean": mean,
                "str/std": std,
                "str/nth": i,
                "str/diffusion_model_train_epoch": diffusion_model_train_epoch, 
            })