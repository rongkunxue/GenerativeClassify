import os
from easydict import EasyDict
import torch
from torch.utils.data import Dataset

import torch
import numpy as np
import torch.backends.cudnn as cudnn
import random
from accelerate import Accelerator
from data import build_loader
from models import build_model
from torch_tool import build_optimizer, build_scheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from grl.utils.model_utils import save_model
from improved_utilities import  imagenet_save, find_max_param_and_grad
from rich.progress import track
import torchvision
import os
import torch
from grl.utils.log import log
from grl.generative_models.metric import compute_likelihood
import wandb

def load_model(
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer = None,
        prefix="checkpoint",
        nth: int = 1,
    ) -> int:

    """
    Overview:
        Load model state_dict, optimizer state_dict, and training iteration from disk. Select every nth checkpoint file named as 'prefix_iteration.pt'.
    Arguments:
        path (:obj:`str`): Path to load model.
        model (:obj:`torch.nn.Module`): Model to load.
        optimizer (:obj:`torch.optim.Optimizer`): Optimizer to load.
        prefix (:obj:`str`): Prefix of the checkpoint file.
        nth (:obj:`int`): Select every nth checkpoint file.
    Returns:
        - last_iteration (:obj:`int`): The iteration of the loaded checkpoint.
    """

    last_iteration = -1
    checkpoint_path = path
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
            log.warning(
                f"Checkpoint path {checkpoint_path} does not exist or is empty"
            )
            return last_iteration

        # Get all the checkpoint files that match the prefix and end with '.pt'
        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_path) if f.endswith(".pt") and f.startswith(prefix)],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )

        # Filter for every nth checkpoint file
        nth_index = min(len(checkpoint_files), nth) - 1

        if not checkpoint_files:
            log.warning(f"No checkpoint files found in {checkpoint_path}")
            return last_iteration

        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[nth_index])

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        last_iteration = checkpoint.get("iteration", -1)
        ori_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
        ori_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ori_state_dict.items()}
        model.load_state_dict(ori_state_dict)
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        log.warning(f"{last_iteration}_checkpoint file has been loaded")
        return last_iteration

    return last_iteration

# def generative_picture(accelerator, model, epoch):
#     img=model.samplePicture()
#     img=accelerator.gather(img)
#     if accelerator.is_local_main_process:
#         img = torchvision.utils.make_grid(
#                 img, value_range=(-1, 1), padding=0, nrow=4
#             )
#         imagenet_save(
#             img.cpu().detach(),
#             config.DATA.video_save_path,
#             epoch,
#             f"Tinyimage",
#         )

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res

# def validate(accelerator, model, val_loader, criterion, epoch):
#     losses = AverageMeter("Loss", ":.4e")
#     top1 = AverageMeter("Acc@1", ":6.2f")
#     top5 = AverageMeter("Acc@5", ":6.2f")
#     # switch to evaluate mode
#     model.eval()
#     with torch.no_grad():
#         for idx, (images, targets) in enumerate(val_loader):
#             # compute output
#             outputs = model(images)
#             outputs, targets = accelerator.gather_for_metrics((outputs, targets))
#             loss = criterion(outputs, targets)
#             # measure accuracy and record loss
#             acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
#             losses.update(loss.item(), outputs.size(0))
#             top1.update(acc1[0], outputs.size(0))
#             top5.update(acc5[0], outputs.size(0))
#         wandb.log(
#             {
#                 f"eval/acc1": top1.avg,
#                 f"eval/acc5": top5.avg,
#                 f"eval/loss": losses.avg,
#                 f"eval/epoch": epoch,
#             },
#             commit=False,
#         )
#     return 0

def picture_analysis(config, accelerator):
    if config.TRAIN.method == "Pretrain":
        ### Load the data
        data_loader_train, data_loader_val, mixup_fn = build_loader(config,if_analyse=True)
        model = build_model(config)
        optimizer = build_optimizer(config, model)
        logp=[]
        for i in range(15):
            diffusion_model_train_epoch = load_model(
                path=config.DATA.checkpoint_path,
                model=model.grlEncoder.diffusionModel.model,
                optimizer=None,
                prefix="DiffusionModel_Pretrain",
                nth=i,) 
            (
                model.grlEncoder.diffusionModel.model,
                data_loader_train,
                data_loader_val,
                optimizer,
            ) = accelerator.prepare(
                model.grlEncoder.diffusionModel.model,
                data_loader_train,
                data_loader_val,
                optimizer,
            )
            for samples, targets in track(
                data_loader_train, disable=not accelerator.is_local_main_process
            ):
                log_p= compute_likelihood(
                        model=model.grlEncoder.diffusionModel,
                        x=samples,
                        t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                        using_Hutchinson_trace_estimator=True,
                )
                logp.append(log_p)
            log_p_mean = torch.stack(logp).mean()
            log_p_max = torch.stack(logp).max()
            log_p_min = torch.stack(logp).min()
            wandb.log(
                {
                    f"train/log_p_mean": log_p_mean,
                    f"train/log_p_max": log_p_max,
                    f"train/log_p_min": log_p_min,
                    f"iteration":i,
                },
                commit=True,
            )
            #print or log log_p
            log.info("i",log_p_mean,log_p_max,log_p_min)
                
        
