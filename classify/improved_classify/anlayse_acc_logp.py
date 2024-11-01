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



def train_epoch(accelerator, model, criterion, data_loader, optimizer,lr_scheduler, epoch):
    model.train()
    num_steps=len(data_loader)
    idx=0
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process,
        description=f"Processing Train epoch {epoch}",
    ):
        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        idx+=1
        # max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(model)
        if accelerator.is_main_process:
            wandb.log(
                {"train/loss": loss, "train/epoch": epoch},
                commit=True,
            )
    return 0

def train_flow_matching(accelerator, model, data_loader, optimizer, epoch):
    model.train()
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        loss = model.matchingLoss(samples)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            wandb.log(
                {"train/loss": loss, "train/epoch": epoch},
                commit=True,
            )
    return 0

def train_icfm_flow_matching(accelerator, model, data_loader, optimizer, epoch):
    model.train()
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        x0 = model.grlEncoder.diffusionModel.gaussian_generator(samples.shape[0]).to(samples.device)
        loss = model.matchingLoss(x0=x0,x1=samples)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            wandb.log(
                {"train/loss": loss, "train/epoch": epoch},
                commit=True,
            )
    return 0


@torch.no_grad()
def analysis_logp(accelerator, model, data_loader, config):
    model.eval()
    from grl.generative_models.metric import compute_likelihood,compute_straightness
    count = 0
    logp_train_list = []
    for samples, targets in track(data_loader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            log_p = compute_likelihood(
                model=model.grlEncoder.diffusionModel,
                x=samples,
                t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                using_Hutchinson_trace_estimator=True,
            
            )
            log_p = accelerator.gather_for_metrics(log_p)
            logp_train_list.append(log_p)
            count += 1
            if count > 30:
                break
    logp_train_mean =  torch.stack(logp_train_list).mean().item()/(32.0*32.0)
    return logp_train_mean


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


def validate(accelerator, model, val_loader, criterion, epoch,mixup_fn=None):
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            # compute output
            if mixup_fn is not None:
                images, targets_mixup=mixup_fn(images, targets)
            else :
                targets_mixup=targets
            outputs = model(images,False)
            loss = criterion(outputs, targets_mixup)
            
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), outputs.size(0))
            top1.update(acc1[0], outputs.size(0))
            top5.update(acc5[0], outputs.size(0))
        if accelerator.is_main_process:
            wandb.log(
                {
                    f"eval/acc1": top1.avg,
                    f"eval/acc5": top5.avg,
                    f"eval/loss": losses.avg,
                    f"eval/epoch": epoch,
                },
                commit=False,
            )
    return top1.avg


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
    log_dict={"acc1":[],"train_logp":[],"eval_logp":[],"epoch":[]}
    
    data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    data_loader_train_analyse, data_loader_val_analyse, mixup_fn = build_loader(config,True)
    
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    
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
        data_loader_train_analyse,
        data_loader_val_analyse,
        data_loader_val,
        optimizer,
    ) = accelerator.prepare(
        model.grlEncoder.diffusionModel.model,
        model.grlHead,
        data_loader_train,
        data_loader_train_analyse,
        data_loader_val_analyse,
        data_loader_val,
        optimizer,
    )
    
    lr_scheduler= build_scheduler(config, optimizer,len(data_loader_train))
    
    for epoch in range(config.TRAIN.iteration):
        acc_1=validate(accelerator, model, data_loader_val, criterion, epoch,mixup_fn)
        train_logp=analysis_logp(accelerator, model, data_loader_train_analyse, epoch)
        eval_logp=analysis_logp(accelerator, model, data_loader_val_analyse, epoch)
        train_epoch(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)    
        
        if accelerator.is_main_process:
            wandb.log(
            {"analyse/train_logp": train_logp, "analyse/epoch": epoch,"analyse/eval_logp": eval_logp},
            commit=False,
            )
        log_dict["acc1"].append(acc_1)
        log_dict["train_logp"].append(train_logp)
        log_dict["eval_logp"].append(eval_logp)
        log_dict["epoch"].append(epoch)

        if (epoch) % config.TEST.generative_freq == 0:
            if accelerator.is_local_main_process:
                import pickle
                with open("/home/xrk/EXP/Cifar-10-icfm/log_dict.pkl", "wb") as f:
                    pickle.dump(log_dict, f)
                accelerator.wait_for_everyone()
                
                
        if hasattr (config.TEST,"sms_freq") and (epoch) % config.TEST.sms_freq == 0:
            if accelerator.is_local_main_process:
                message = f"Project : {config.PROJECT_NAME} Special : {config.extra}\n"
                message += f"epoch : {epoch + 1}\n"
                message += f"acc_1 : {acc_1}\n"
                import http.client, urllib
                conn = http.client.HTTPSConnection("api.pushover.net:443")
                conn.request("POST", "/1/messages.json",
                urllib.parse.urlencode({
                    "token": "a7rgfcc4v14rfpkv3j8i5mnsvuiwsv",  
                    "user": "ufam55v8r8425rc79e45aeo2thb1xc",    
                    "message": message, 
                }), { "Content-type": "application/x-www-form-urlencoded" })
                conn.getresponse()
            accelerator.wait_for_everyone()