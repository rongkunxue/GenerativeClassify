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
from improved_utilities import load_pt, imagenet_save, find_max_param_and_grad
from rich.progress import track


def train_epoch(accelerator, model, criterion, data_loader, optimizer, epoch):
    model.train()
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        # max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(model)
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch},
            commit=True,
        )
    return 0


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(accelerator, model, val_loader, criterion, epoch):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            # compute output
            outputs = model(images)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            loss = criterion(outputs, targets)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), outputs.size(0))
            top1.update(acc1[0], outputs.size(0))
            top5.update(acc5[0], outputs.size(0))
        wandb.log(
            {
                f"eval/acc1": top1.avg,
                f"eval/acc5": top5.avg,
                f"eval/loss": losses.avg,
                f"eval/epoch": epoch,
            },
            commit=False,
        )
    return 0


def main(config, accelerator):
    data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)

    last_iteration = load_pt(
        config.DATA.checkpoint_path, accelerator, model.grlEncoder.diffusionModel.model
    )

    if config.TRAIN.loss_function == "CrossEntropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif config.TRAIN.loss_function == "LabelSmoothingCrossEntropy":
        criterion = LabelSmoothingCrossEntropy(smoothing=config.TRAIN.label_smoothing)
    elif config.TRAIN.loss_function == "SoftTargetCrossEntropy":
        criterion = SoftTargetCrossEntropy()
    else:
        raise NotImplementedError
    
    
    (
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_train,
        data_loader_val,
        optimizer,
    ) = accelerator.prepare(
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_train,
        data_loader_val,
        optimizer,
    )
    for epoch in range(config.TRAIN.iteration):
        if epoch % config.TEST.eval_freq == 0:
            validate(accelerator, model, data_loader_val, criterion, epoch)
        train_epoch(accelerator, model, criterion, data_loader_train, optimizer, epoch)


def make_config(device):
    classes = 200
    image_size = 64
    config = EasyDict(
        dict(
            PROJECT_NAME="Classify_Tiny_imagent_label_smooth",
            DEVICE=device,
            DATA=dict(
                batch_size=180,
                classes=classes,
                img_size=image_size,
                dataset_path="/mnt/nfs3/zhangjinouwen/dataset/tiny-imagenet-200",
                checkpoint_path="/mnt/nfs3/zhangjinouwen/checkpoint/tiny_image_64",
                video_save_path="/root/Github/generativeencoder/classify/improved_classify/video",
                dataset="Tinyimagenet",
                AUG=dict(
                    interpolation="bicubic",
                    color_jitter=0.4,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    reprob=0.25,
                    remode="pixel",
                    recount=1,
                ),
            ),
            MODEL=dict(
                TYPE="GenerativeClassify",
                t_span=20,
                image_size=image_size,
                classes=classes,
                diffusion_model=dict(
                    device=device,
                    x_size=(3, image_size, image_size),
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq_adjoint",
                        ),
                    ),
                    path=dict(
                        type="gvp",
                    ),
                    model=dict(
                        type="velocity_function",
                        args=dict(
                            backbone=dict(
                                type="unet",
                                args={},
                            ),
                        ),
                    ),
                ),
            ),
            TRAIN=dict(
                loss_function="LabelSmoothingCrossEntropy", #LabelSmoothingCrossEntropy or SoftTargetCrossEntropy
                label_smoothing=0.1,
                training_loss_type="flow_matching",
                optimizer_type="adam",
                lr=1e-4,
                iteration=2000,
                device=device,
            ),
            TEST=dict(
                seed=0,
                crop=True,
                eval_freq=5,
            ),
        )
    )
    return config


if __name__ == "__main__":
    accelerator = Accelerator()
    config = make_config(accelerator.device)
    import wandb

    wandb.init(
        project=config.PROJECT_NAME,
        config=config,
    )
    main(config, accelerator)
