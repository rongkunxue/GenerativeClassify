import os
from easydict import EasyDict
import torch
from torch.utils.data import Dataset

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
        data_loader, disable=not accelerator.is_local_main_process
    ):
        outputs = model(samples)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        idx+=1
        # max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(model)
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch},
            commit=True,
        )
    return 0

def train_epoch_withflowmaching(accelerator, model, criterion, data_loader, optimizer,lr_scheduler, epoch):
    model.train()
    num_steps=len(data_loader)
    idx=0
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        outputs = model(samples)
        loss1 = criterion(outputs, targets)
        x0 = model.grlEncoder.diffusionModel.gaussian_generator(samples.shape[0]).to(samples.device)
        loss2 = model.matchingLoss(x0=x0,x1=samples)
        loss = loss1+0.1*loss2
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        idx+=1
        # max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(model)
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch,"train/loss1": loss1,"train/loss2": loss2},
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
        wandb.log(
            {"train/loss": loss, "train/epoch": epoch},
            commit=True,
        )
    return 0

@torch.no_grad()
def analysis_logp(accelerator, model, data_loader, epoch):
    model.eval()
    from grl.generative_models.metric import compute_likelihood
    #only count 10 iteration in dataset
    count = 0
    log_p = []
    for samples, targets in track(
        data_loader, disable=not accelerator.is_local_main_process
    ):
        logp=compute_likelihood(
                model=model.grlEncoder.diffusionModel,
                x=samples,
                t=torch.linspace(0.0, 1.0, 100).to(samples.device),
                using_Hutchinson_trace_estimator=True,
            )
        log_p.append(logp)
        if count == 10:
            break
    mean_log_p = torch.stack(log_p).mean()
    return mean_log_p

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

@torch.no_grad()
def collect_new_dataloader(accelerator,data_loader, model, config,prefix):
    if prefix in ["train","test"]:
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
    
    elif prefix == "noise":
        batch_size = 200
        total_samples = 50000
        num_batches = total_samples // batch_size // accelerator.num_processes
        x1_list = []
        x0_list = []
        label_list = []
        model.eval()
        for _ in range(num_batches):
            t_span = torch.linspace(0.0, 1.0, 20).to(config.TRAIN.device)
            x0 = model.grlEncoder.diffusionModel.gaussian_generator(batch_size).to(config.TRAIN.device)
            x1 = model.grlEncoder.diffusionModel.sample_forward_process(t_span=t_span, x_0=x0).to(config.TRAIN.device)[-1] 
            x0,x1= accelerator.gather_for_metrics((x0, x1))
            x1_list.append(x1.cpu())
            x0_list.append(x0.cpu())
        x1_ = torch.cat(x1_list, dim=0)
        x0_ = torch.cat(x0_list, dim=0)
        if accelerator.is_main_process:
            print(x1_.shape[0]/50000)
        data_to_save = {"x1": x1_, "x0": x0_}
        if accelerator.is_main_process:
            path=config.DATA.checkpoint_path
            torch.save(
                data_to_save,
                f"{path}/{prefix}_new_data_{config.PROJECT_NAME}.pt",
        )
    
    else :
        raise NotImplementedError("prefix not implemented")
        

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


def train(config, accelerator):
    
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
                
                wandb.log(
                {"eval/train_logp": train_log, "eval/epoch": epoch,"eval/eval_logp": eval_log},
                commit=False,
                )
                
        
            if (epoch + 1) % config.TEST.generative_freq == 0:
                generative_picture(accelerator, model,  epoch,config)
                
            if config.model_type=="ICFM":
                train_icfm_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            elif config.model_type=="Diff":
                train_flow_matching(accelerator, model, data_loader_train, optimizer, epoch)
            else:
                raise NotImplementedError("Model type not implemented")
    
    if config.TRAIN.method == "Recitified_collect":
        data_loader_train, data_loader_val, mixup_fn = build_loader(config,True)
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
        
        # collect_new_dataloader(accelerator,data_loader_train, model, config,"train")
        # collect_new_dataloader(accelerator,data_loader_val, model, config,"test")
        collect_new_dataloader(accelerator,data_loader_train, model, config,"noise")


    if config.TRAIN.method == "Recitified_collect" or config.TRAIN.method == "Recitified":   
        data_loader_train, data_loader_val, mixup_fn = build_loader(config)
        train_data=torch.load(f"{config.DATA.checkpoint_path}/train_new_data_{config.PROJECT_NAME}.pt")
        test_data=torch.load(f"{config.DATA.checkpoint_path}/test_new_data_{config.PROJECT_NAME}.pt")
        x0=torch.cat([train_data["x0"],test_data["x0"]],dim=0)
        x1=torch.cat([train_data["x1"],test_data["x1"]],dim=0)
        
        dataset = torch.utils.data.TensorDataset(x0,x1)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.DATA.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        model = build_model(config)
        if (
            hasattr(config.DATA, "checkpoint_path")
            and config.DATA.checkpoint_path is not None
        ):
            load_model(
                    path=config.DATA.checkpoint_path,
                    model=model,
                    optimizer=None,
                    prefix="GenerativeClassify",
            )
        else:
            raise NotImplementedError("Please provide the checkpoint path")
        
        import copy
        recitified_model=copy.deepcopy(model)
        optimizer = torch.optim.Adam(
            recitified_model.grlEncoder.diffusionModel.model.parameters(),
            lr=config.TRAIN.lr,
        )    
        
        (
        recitified_model.grlEncoder.diffusionModel.model,
        recitified_model.grlHead,
        dataloader,
        data_loader_val,
        optimizer,
        ) = accelerator.prepare(
            recitified_model.grlEncoder.diffusionModel.model,
            recitified_model.grlHead,
            dataloader,
            data_loader_val,
            optimizer,
        )
        if config.TRAIN.loss_function == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif config.TRAIN.loss_function == "LabelSmoothingCrossEntropy":
            criterion = LabelSmoothingCrossEntropy(smoothing=config.TRAIN.label_smoothing)
        elif config.TRAIN.loss_function == "SoftTargetCrossEntropy":
            criterion = SoftTargetCrossEntropy()
        else:
            raise NotImplementedError
        for epoch in range(config.TRAIN.iteration):
            if (epoch) % config.TEST.eval_freq == 0:
                validate(accelerator, recitified_model, data_loader_val, criterion, epoch,mixup_fn)
            train_recitified_flow_matching(accelerator, recitified_model,dataloader, optimizer, epoch)
            # if hasattr(config.Train,"flow_matching") and config.Train.flow_matching:
            #     if model.type=="ICFM":
            #         train_epoch_withflowmaching(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
            #     elif model.type=="Diff":
            #         raise NotImplementedError("will write the method")                     
            #     else:
            #         raise NotImplementedError("Model type not implemented")
            # else :
            #     train_epoch(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                if accelerator.is_local_main_process:
                    save_model(
                        config.DATA.checkpoint_path,
                        model,
                        optimizer,
                        epoch,
                        "GenerativeClassify_recitified",
                    )
        
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
            if (epoch+1) % config.TEST.eval_freq == 0:
                validate(accelerator, model, data_loader_val, criterion, epoch,mixup_fn)
            if hasattr(config.Train,"flow_matching") and config.Train.flow_matching:
                if model.type=="ICFM":
                    train_epoch_withflowmaching(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
                elif model.type=="Diff":
                    raise NotImplementedError("will write the method")                     
                else:
                    raise NotImplementedError("Model type not implemented")
            else :
                train_epoch(accelerator, model, criterion, data_loader_train, optimizer, lr_scheduler,epoch)
            if (epoch + 1) % config.TEST.checkpoint_freq == 0:
                if accelerator.is_local_main_process:
                    save_model(
                        config.DATA.checkpoint_path,
                        model,
                        optimizer,
                        epoch,
                        "GenerativeClassify",
                    )
                    
    