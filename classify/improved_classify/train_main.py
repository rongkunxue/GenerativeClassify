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


def imagenet_save(img, save_path="./", iteration=0, prefix="img"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    import torchvision.transforms as transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    img = inv_normalize(img)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)

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

def train_recitified_flow_matching(accelerator, model, data_loader, optimizer, epoch):
    model.train()
    for x0, x1 in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        loss = model.matchingLoss(x0=x0,x1=x1)
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
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        logp=compute_likelihood(
                model=model.grlEncoder.diffusionModel,
                x=samples,
                t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                using_Hutchinson_trace_estimator=True,
            )
        logp=accelerator.gather_for_metrics(logp)
        logp_mean=logp.mean()
        break
    #todo: need to check the normalization
    ipdb.set_trace()
    mean_norm = logp_mean/(samples.shape[2]*samples.shape[3])
    return mean_norm

@torch.no_grad()
def analysis_straightness(accelerator, model,config):
    model.eval()
    from grl.generative_models.metric import compute_straightness
    straightness=compute_straightness(model=model.grlEncoder.diffusionModel,batch_size=config.DATA.batch_size)
    straightness_gather=accelerator.gather_for_metrics(straightness)
    mean_straightness = straightness_gather.mean()
    if accelerator.is_main_process:
        wandb.log(
            {"eval/straightness": mean_straightness},
            commit=False,
        )
    return mean_straightness

@torch.no_grad()
def generative_picture(accelerator, model, epoch,config):
    model.eval()
    img=model.samplePicture()
    img=accelerator.gather_for_metrics(img)
    img = torchvision.utils.make_grid(
            img, value_range=(-1, 1), padding=0, nrow=4
        )
    if accelerator.is_local_main_process:
        imagenet_save(
            img.cpu().detach(),
            config.DATA.video_save_path,
            epoch,
            f"Tinyimage",
        )


    x1_list = []
    x0_list = []
    label_list = []
    model.eval()
    for batch_data, batch_label in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        t_span = torch.linspace(0.0, 1.0, 20).to(config.TRAIN.device)
        x0= model.grlEncoder.diffusionModel.forward_sample(
                t_span=t_span, x=batch_data,with_grad=False
            ).detach()
        x1, x0, label = accelerator.gather_for_metrics(
                (batch_data, x0, batch_label)
            )
        x1_list.append(x1.cpu())
        label_list.append(label.cpu())
        x0_list.append(x0.cpu())
    x1_ = torch.cat(x1_list, dim=0)
    x0_ = torch.cat(x0_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    data_to_save = {"x1": x1_, "x0": x0_, "label": label_}
    if accelerator.is_main_process:
        path=config.DATA.checkpoint_path
        torch.save(
            data_to_save,
            f"{path}/{prefix}_new_data_{config.PROJECT_NAME}.pt",
        )

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
            outputs = model(images)
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
    
    if config.TRAIN.method == "Pretrain":
        ### Load the data
        data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        model = build_model(config)
        optimizer = build_optimizer(config, model)
        
        ### Load the model
        if (
                hasattr(config.DATA, "checkpoint_path")
                and config.DATA.checkpoint_path is not None
            ):
            diffusion_model_train_epoch = load_model(
                    path=config.DATA.checkpoint_path,
                    model=model.grlEncoder.diffusionModel.model,
                    optimizer=None,
                    prefix="DiffusionModel_Pretrain",
                )
        else:
                diffusion_model_train_epoch=0
                
        ### Prepare the model
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
        
        ### Train the model
        for epoch in range(config.TRAIN.iteration):
            if diffusion_model_train_epoch >= epoch:
                continue
            
                
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                if accelerator.is_local_main_process:
                    save_model(
                        config.DATA.checkpoint_path,
                        model.grlEncoder.diffusionModel.model,
                        optimizer,
                        epoch,
                        "DiffusionModel_Pretrain",
                    )
                    
            if hasattr (config.TEST,"analyse_freq") and (epoch+1) % config.TEST.analyse_freq == 0:
                train_log=analysis_logp(accelerator, model, data_loader_train, epoch)
                eval_log=analysis_logp(accelerator, model, data_loader_val, epoch)
                
                if accelerator.is_main_process:
                    wandb.log(
                    {"analyse/train_logp": train_log, "analyse/epoch": epoch,"eanalyse/eval_logp": eval_log},
                    commit=False,
                    )
                
        
            if (epoch + 1) % config.TEST.generative_freq == 0:
                generative_picture(accelerator, model,  epoch,config)
                
            if config.MODEL.model_type in ["ICFM","OT"]:
                train_icfm_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            elif config.MDOEL.model_type=="Diff":
                train_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            else:
                raise NotImplementedError("Model type not implemented")
            
        
    if config.TRAIN.method == "Finetune" or config.TRAIN.method == "Pretrain":
        data_loader_train, data_loader_val, mixup_fn = build_loader(config)
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
            data_loader_val,
            optimizer,
        ) = accelerator.prepare(
            model.grlEncoder.diffusionModel.model,
            model.grlHead,
            data_loader_train,
            data_loader_val,
            optimizer,
        )
        lr_scheduler= build_scheduler(config, optimizer,len(data_loader_train))
        for epoch in range(config.TRAIN.iteration):
            if (epoch) % config.TEST.eval_freq == 0:
                acc_1=validate(accelerator, model, data_loader_val, criterion, epoch,mixup_fn)
            train_epoch(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
            
            if hasattr (config.TEST,"analyse_freq") and (epoch+1) % config.TEST.analyse_freq == 0:
                train_log=analysis_logp(accelerator, model, data_loader_train, epoch)
                # eval_log=analysis_logp(accelerator, model, data_loader_val, epoch)
                # analysis_straightness(accelerator, model,config)
                
                if accelerator.is_main_process:
                    wandb.log(
                    {"analyse/train_logp": train_log, "analyse/epoch": epoch},
                    commit=False,
                    )
                
                    
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                if accelerator.is_local_main_process:
                    save_model(
                        config.DATA.checkpoint_path,
                        model,
                        optimizer,
                        epoch,
                        "GenerativeClassify",
                    )
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