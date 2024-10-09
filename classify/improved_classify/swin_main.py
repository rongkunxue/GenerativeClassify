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
from utils import train_one_epoch, validate
import ipdb


def main(config, accelerator):
    data_loader_train, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    if config.AUG.MIXUP > 0.0:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    last_iteration = -1
    model, data_loader_train, data_loader_val, optimizer = accelerator.prepare(
        model, data_loader_train, data_loader_val, optimizer
    )
    for epoch in range(config.TRAIN.EPOCHS):
        if epoch <= last_iteration:
            continue
        train_one_epoch(
            accelerator,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
        )
        validate(accelerator, data_loader_val, model)


def make_config(device):
    classes = 200
    config = EasyDict(
        dict(
            METHOD="SWIN",
            DEVICE=device,
            DATA=dict(
                BATCH_SIZE=128,
                DATA_PATH="/mnt/nfs3/zhangjinouwen/dataset/tiny-imagenet-200",
                DATASET="tiny_imagenet",
                IMG_SIZE=224,
                INTERPOLATION="bicubic",
                ZIP_MODE=False,
                PIN_MEMORY=True,
                NUM_WORKERS=8,
                MASK_PATCH_SIZE=32,
                MASK_RADIO=0.6,
            ),
            AUG=dict(
                COLOR_JITTER=0.4,
                AUTO_AUGMENT="rand-m9-mstd0.5-inc1",
                REPROB=0.25,
                REMODE="pixel",
                RECOUNT=1,
                MIXUP=0.8,
                CUTMIX=1.0,
                CUTMIX_MINMAX=None,
                MIXUP_PROB=1.0,
                MIXUP_SWITCH_PROB=0.5,
                MIXUP_MODE="batch",
            ),
            MODEL=dict(
                TYPE="swin",
                NAME="swin_small_patch4_window7_224",
                DROP_RATE=0.0,
                DROP_PATH_RATE=0.3,
                LABEL_SMOOTHING=0.1,
                NUM_CLASSES=classes,
                SWIN=dict(
                    PATCH_SIZE=4,
                    IN_CHANS=3,
                    EMBED_DIM=96,
                    DEPTHS=[2, 2, 18, 2],
                    NUM_HEADS=[3, 6, 12, 24],
                    WINDOW_SIZE=7,
                    MLP_RATIO=4.0,
                    QKV_BIAS=True,
                    QK_SCALE=None,
                    APE=False,
                    PATCH_NORM=True,
                ),
            ),
            TRAIN=dict(
                EPOCHS=300,
                WARMUP_EPOCHS=20,
                WEIGHT_DECAY=0.05,
                BATCH_SIZE=128,
                ACCUMULATION_STEPS=1,
                BASE_LR=0.000125,
                WARMUP_LR=1.25e-07,
                CLIP_GRAD=5.0,
                AUTO_RESUME=True,
                MIN_LR=1.25e-06,
                USE_CHECKPOINT=False,
                LR_SCHEDULER=dict(
                    NAME="cosine",
                    DECAY_EPOCHS=30,
                    DECAY_RATE=0.1,
                    WARMUP_PREFIX=True,
                    GAMMA=0.1,
                    MULTISTEPS=[],
                ),
                OPTIMIZER=dict(
                    NAME="adamw",
                    EPS=1e-08,
                    BETAS=(0.9, 0.999),
                    MOMENTUM=0.9,
                ),
                MOE=dict(
                    SAVE_MASTER=False,
                ),
                LAYER_DECAY=1.0,
            ),
            TEST=dict(
                SEED=0,
                CROP=True,
            ),
        )
    )
    return config


if __name__ == "__main__":
    accelerator = Accelerator()
    config = make_config(accelerator.device)
    seed = config.TEST.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    import wandb

    wandb.init(
        project=f"Classify_{config.METHOD}",
        config=config,
        mode="offline",
    )
    if config.METHOD == "SWIN":
        linear_scaled_lr = (
            config.TRAIN.BASE_LR
            * config.DATA.BATCH_SIZE
            * accelerator.num_processes
            / 512.0
        )
        linear_scaled_warmup_lr = (
            config.TRAIN.WARMUP_LR
            * config.DATA.BATCH_SIZE
            * accelerator.num_processes
            / 512.0
        )
        linear_scaled_min_lr = (
            config.TRAIN.MIN_LR
            * config.DATA.BATCH_SIZE
            * accelerator.num_processes
            / 512.0
        )
        config.TRAIN.BASE_LR = linear_scaled_lr
        config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
        config.TRAIN.MIN_LR = linear_scaled_min_lr
    main(config, accelerator)
