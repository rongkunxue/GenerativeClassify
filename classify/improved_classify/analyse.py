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

from typing import Union

import torch
import treetensor
from tensordict import TensorDict
from torch.distributions import Independent, Normal

from grl.numerical_methods.numerical_solvers.ode_solver import (
    ODESolver,
)
from grl.utils import find_parameters

def compute_likelihood_return_all(
    model,
    x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
    t: torch.Tensor = None,
    condition: Union[torch.Tensor, TensorDict] = None,
    using_Hutchinson_trace_estimator: bool = True,
) -> torch.Tensor:
    """
    Overview:
        Compute Likelihood of samples in generative model for gaussian prior.
    Arguments:
        - model (:obj:`Union[Callable, nn.Module]`): The model.
        - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
        - t (:obj:`torch.Tensor`): The input time.
        - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        - using_Hutchinson_trace_estimator (:obj:`bool`): Whether to use Hutchinson trace estimator. It is an approximation of the trace of the Jacobian of the drift function, which is faster but less accurate. We recommend setting it to True for high dimensional data.
    Returns:
        - log_likelihood (:obj:`torch.Tensor`): The likelihood of the samples.
    """
    # TODO: Add support for EnergyConditionalDiffusionModel; Add support for t; Add support for treetensor.torch.Tensor

    if model.get_type() == "EnergyConditionalDiffusionModel":
        raise NotImplementedError(
            "EnergyConditionalDiffusionModel is not supported yet."
        )
    elif model.get_type() == "DiffusionModel":
        model_drift = model.diffusion_process.forward_ode(
            function=model.model, function_type=model.model_type, condition=condition
        ).drift
        model_params = find_parameters(model.model)
    elif model.get_type() in [
        "IndependentConditionalFlowModel",
        "OptimalTransportConditionalFlowModel",
    ]:
        model_drift = lambda t, x: -model.model(1 - t, x, condition)
        model_params = find_parameters(model.model)
    elif model.get_type() == "FlowModel":
        model_drift = lambda t, x: model.model(t, x, condition)
        model_params = find_parameters(model.model)
    else:
        raise ValueError("Invalid model type: {}".format(model.get_type()))

    def compute_trace_of_jacobian_general(dx, x):
        # Assuming x has shape (B, D1, ..., Dn)
        shape = x.shape[1:]  # get the shape of a single element in the batch
        outputs = torch.zeros(
            x.shape[0], device=x.device, dtype=x.dtype
        )  # trace for each batch
        # Iterate through each index in the product of dimensions
        for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
            if len(index.shape) > 0:
                index = tuple(index)
            else:
                index = (index,)
            grad_outputs = torch.zeros_like(x)
            grad_outputs[(slice(None), *index)] = (
                1  # set one at the specific index across all batches
            )
            grads = torch.autograd.grad(
                outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True
            )[0]
            outputs += grads[(slice(None), *index)]
        return outputs

    def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
        """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

        fn_eps = torch.sum(dx * eps)
        grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
        outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
        return outputs

    def composite_drift(t, x):
        # where x is actually x0_and_diff_logp, (x0, diff_logp), which is a tuple containing x and logp_xt_minus_logp_x0
        with torch.set_grad_enabled(True):
            t = t.detach()
            x_t = x[0].detach()
            logp_xt_minus_logp_x0 = x[1]

            x_t.requires_grad = True
            t.requires_grad = True

            dx = model_drift(t, x_t)
            if using_Hutchinson_trace_estimator:
                noise = torch.randn_like(x_t, device=x_t.device)
                logp_drift = -compute_trace_of_jacobian_by_Hutchinson_Skilling(
                    dx, x_t, noise
                )
                # logp_drift = - divergence_approx(dx, x_t, noise)
            else:
                logp_drift = -compute_trace_of_jacobian_general(dx, x_t)

            return dx, logp_drift

    # x.shape = [batch_size, state_dim]
    x0_and_diff_logp = (x, torch.zeros(x.shape[0], device=x.device))

    if t is None:
        eps = 1e-3
        t_span = torch.linspace(eps, 1.0, 1000).to(x.device)
    else:
        t_span = t.to(x.device)

    solver = ODESolver(library="torchdiffeq_adjoint")

    x1_and_logp1 = solver.integrate(
        drift=composite_drift,
        x0=x0_and_diff_logp,
        t_span=t_span,
        adjoint_params=model_params,
    )

    logp_x1_minus_logp_x0 = x1_and_logp1[1][-1]
    x1 = x1_and_logp1[0][-1]
    x1_1d = x1.reshape(x1.shape[0], -1)
    logp_x1 = Independent(
        Normal(
            loc=torch.zeros_like(x1_1d, device=x1_1d.device),
            scale=torch.ones_like(x1_1d, device=x1_1d.device),
        ),
        1,
    ).log_prob(x1_1d)

    log_likelihood = logp_x1 - logp_x1_minus_logp_x0

    return log_likelihood, logp_x1, logp_x1_minus_logp_x0


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
    # checkpoint_files = [f for f in checkpoint_files if f.startswith(prefix)]

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
    
    if config.TRAIN.method == "analyse":
        data_loader_train, data_loader_val, mixup_fn = build_loader(config, if_analyse=True)
        real_model = build_model(config)

        # Iterate over checkpoints (nth)
        for i in range(1, 300):
            import copy
            model = copy.deepcopy(real_model)

            # Load the diffusion model from checkpoint
            diffusion_model_train_epoch = load_model(
                path=config.DATA.checkpoint_path,
                model=model,
                optimizer=None,
                prefix="GenerativeClassify",
                nth=i,
            )

            # Prepare the model and data loaders with accelerator
            model.grlEncoder.diffusionModel.model, data_loader_train, data_loader_val = accelerator.prepare(
                model.grlEncoder.diffusionModel.model, data_loader_train, data_loader_val
            )
            
            logp_train_list = []
            logp_val_list = []
            count = 0
            model.eval()
            # Perform log likelihood calculation for two iterations
           
            for samples, targets in track(data_loader_train, disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    log_p = compute_likelihood(
                        model=model.grlEncoder.diffusionModel,
                        x=samples,
                        t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                        using_Hutchinson_trace_estimator=True,
                    
                    )
                    log_p=accelerator.gather_for_metrics(log_p)
                    logp_train_list.append(log_p)
                    count += 1
                    if count > 10:
                        break

            count = 0
            for samples, targets in track(data_loader_val, disable=not accelerator.is_local_main_process):
                with torch.no_grad():
                    log_p = compute_likelihood(
                        model=model.grlEncoder.diffusionModel,
                        x=samples,
                        t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                        using_Hutchinson_trace_estimator=True,
                    )
                    log_p=accelerator.gather_for_metrics(log_p)
                    logp_val_list.append(log_p)
                    count += 1
                    if count > 10:
                        break

                # Calculate mean likelihoods
                logp_train_mean =  torch.stack(logp_train_list).mean().item()/(32.0*32.0)
                logp_val_mean = torch.stack(logp_val_list).mean().item()/(32.0*32.0)

            if accelerator.is_main_process:
                wandb.log({
                    "train_mean": logp_train_mean,
                    "val_mean": logp_val_mean,
                    "nth": i-1,
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
        for i in range(1, 300):
            import copy
            model = copy.deepcopy(real_model)

            # Load the diffusion model from checkpoint
            diffusion_model_train_epoch = load_model(
                path=config.DATA.checkpoint_path,
                model=model.grlEncoder.diffusionModel.model,
                optimizer=None,
                prefix="GenerativeClassify",
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