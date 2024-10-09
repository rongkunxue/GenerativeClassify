import os
import dmc2gym
from typing import List, Optional, Tuple, Union
import signal
import sys
from torch.utils.data import Dataset, DataLoader
import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
import cv2
from improved_utilities import resize, SampleData, ReplayMemoryDataset
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn import functional as F

import treetensor

from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from easydict import EasyDict
from matplotlib import animation

from accelerate import Accelerator
import ipdb
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
from grl.utils.log import log
from grl.neural_network import register_module
from grl.neural_network.encoders import GaussianFourierProjectionEncoder

from functools import partial
from improved_diffusion.unet import UNetModel
import wandb
from improved_utilities import (
    save_state,
    save_pt,
    load_pt,
    load_state,
    img_save,
    find_max_param_and_grad,
    ReplayMemoryDataset,
)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([4, 8]),
            dropout=0,
            channel_mult=(1, 2, 3, 4),
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


GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous()
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        should_proj = in_channels != out_channels
        self.proj = Conv1x1(in_channels, out_channels) if should_proj else nn.Identity()
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        x = x + r
        x = self.attn(x)
        return x

class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_ch, out_ch, cond_channels, attn)
                for (in_ch, out_ch) in zip(list_in_channels, list_out_channels)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, to_cat: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            x = x if to_cat is None else torch.cat((x, to_cat[i]), dim=1)
            x = resblock(x, cond)
            outputs.append(x)
        return x, outputs

class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            u_blocks.append(
                ResBlocks(
                    list_in_channels=[2 * c2] * n + [c1 + c2],
                    list_out_channels=[c2] * n + [c1],
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        upsamples = [nn.Identity()] + [Upsample(c) for c in reversed(channels[:-1])]
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, cond)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, cond)

        u_outputs = []
        for block, up, skip in zip(self.u_blocks, self.upsamples, reversed(d_outputs)):
            x_up = up(x)
            x, block_outputs = block(x_up, cond, skip[::-1])
            u_outputs.append((x_up, *block_outputs))

        return x, d_outputs, u_outputs

class DynamicModule(nn.Module):

    def __init__(self):
        super(DynamicModule, self).__init__()
        self.unet = UNet(
            cond_channels=256, depths=[2,2,2,2], channels=[64,64,64,64], attn_depths=[0,0,0,0],
        )

        num_actions = 9
        cond_channels = 256
        num_steps_conditioning = 1

        self.conv_in = Conv3x3((num_steps_conditioning+1)*3, 64)
        self.norm_out = GroupNorm(64)

        self.conv_out = Conv3x3(64, 3)


        # # action type being discrete
        # self.act_emb = nn.Sequential(
        #     nn.Embedding(num_actions, cond_channels // num_steps_conditioning // 2),
        #     nn.Flatten(),  # b t e -> b (t e)
        # )

        # action type being continuous
        self.act_emb = GaussianFourierProjectionEncoder(embed_dim=cond_channels//2, x_shape=[1], flatten=True)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        if condition is not None:
            # action_emb = self.act_emb(condition['action'].squeeze(1).to(torch.long))
            action_emb = self.act_emb(condition['action'])
            cond = torch.cat((action_emb, t), dim=1)
            
            # concatenate condition and t
            
            x = torch.cat((x, condition['state']), dim=1)
        else:
            cond = t
        x = self.conv_in(x)
        x, d_outputs, u_outputs = self.unet(cond=cond, x=x)
        x = self.conv_out(F.silu(self.norm_out(x)))

        return x

register_module(DynamicModule, "DynamicModule")


def make_config(device):
    x_size = (3, 64, 64)
    t_embedding_dim = 128
    t_encoder = dict(
        type="GaussianFourierProjectionTimeEncoder",
        args=dict(
            embed_dim=t_embedding_dim,
            scale=30.0,
        ),
    )
    config = EasyDict(
        dict(
            device=device,
            encoder=dict(
                diffusion_model=dict(
                    device=device,
                    x_size=x_size,
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq",
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
            ),
            dynamic=dict(
                diffusion_model=dict(
                    device=device,
                    x_size=x_size,
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq",
                        ),
                    ),
                    path=dict(
                        type="gvp",
                    ),
                    model=dict(
                        type="velocity_function",
                        args=dict(
                            t_encoder=t_encoder,
                            backbone=dict(
                                type="DynamicModule",
                                args={},
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=1e-4,
                iterations=200000,
                encoder_iterations=0,
                dynamic_iterations=200,
                batch_size=50,
                eval_freq=10,
                checkpoint_freq=50,
                dataset_path="/root/generativeencoder/exp/toy/cartpole/data",
                encoder_checkpoint_path="/mnt/nfs3/zhangjinouwen/checkpoint/cartpole/encoder",
                dynamic_checkpoint_path="/mnt/nfs3/zhangjinouwen/checkpoint/cartpole/dynamic",
                accelerate_checkpoint_path="/root/generativeencoder/exp/toy/cartpole/data/checkpoint",
                video_save_path="/root/generativeencoder/exp/toy/cartpole/video",
                device=device,
            ),
        )
    )

    return config


def render_video(data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    fig = plt.figure(figsize=(12, 12))

    ims = []

    for i, data in enumerate(data_list):

        grid = make_grid(
            data[:,:,:,:].contiguous().clip(-1, 1), value_range=(-1, 1), padding=0, nrow=4
        )/2+0.5
        img = ToPILImage()(grid)
        im = plt.imshow(img)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
    ani.save(
        os.path.join(video_save_path, f"{prefix}_{iteration}.mp4"),
        fps=fps,
        dpi=dpi,
    )
    # clean up
    plt.close(fig)
    plt.clf()

def pipeline():

    accelerator = Accelerator()
    device = accelerator.device
    config = make_config(device)

    # if accelerator.is_local_main_process:
    #     transform = transforms.Compose(
    #         [
    #             transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
    #             transforms.ToPILImage(),
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #         ]
    #     )
    #     import os
    #     os.environ["MUJOCO_GL"] = "egl"
    #     env = dmc2gym.make(domain_name='cartpole', task_name='balance', seed=1,from_pixels=True,visualize_reward=False)
    #     myData=SampleData(0,'/mnt/nfs/xuerongkun/dataset/Cartpole100',100,1050,(3,64,64),(1,))
    #     myData.start_sample_game(env,transform)
    # accelerator.wait_for_everyone()
    # print("finish")
    # ipdb.set_trace()

    logging.info(f"device{device}, config: \n{config}")
    wandb.init(
        project="cartpole",
        config=config,
        group="DDP",
        mode="offline",
    )
    diffusion_model_encoder = DiffusionModel(config=config.encoder.diffusion_model)
    diffusion_model_dynamic = DiffusionModel(config=config.dynamic.diffusion_model)
    # last_iteration_encoder=load_pt(config.parameter.encoder_checkpoint_path,accelerator,diffusion_model_encoder.model)
    # last_iteration_dynamic=load_pt(config.parameter.dynamic_checkpoint_path,accelerator,diffusion_model_dynamic.model)

    with accelerator.main_process_first():
        dataset = ReplayMemoryDataset(0, config.parameter.dataset_path, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer_encoder = torch.optim.Adam(
        diffusion_model_encoder.model.parameters(),
        lr=config.parameter.lr,
    )

    diffusion_model_encoder.model, optimizer_encoder, dataloader = accelerator.prepare(
        diffusion_model_encoder.model, optimizer_encoder, dataloader
    )

    optimizer_dynamic = torch.optim.Adam(
        diffusion_model_encoder.model.parameters(),
        lr=config.parameter.lr,
    )

    diffusion_model_dynamic.model, optimizer_dynamic = accelerator.prepare(
        diffusion_model_dynamic.model, optimizer_dynamic
    )

    last_iteration = load_state(
        config.parameter.accelerate_checkpoint_path, accelerator
    )
    last_iteration_encoder = last_iteration
    last_iteration_dynamic = last_iteration

    counter = 0
    history_iteration = [-1]

    for iteration in range(config.parameter.encoder_iterations):

        if iteration <= last_iteration_encoder:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            diffusion_model_encoder.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)
            x_t = diffusion_model_encoder.sample_forward_process(t_span=t_span, batch_size=4)[
                -1, ...
            ]
            x_t = accelerator.gather_for_metrics(x_t)
            x_t = torchvision.utils.make_grid(
                x_t, value_range=(-1, 1), padding=0, nrow=4
            )
            if accelerator.is_local_main_process:
                img_save(
                    x_t.cpu().detach(),
                    config.parameter.video_save_path,
                    iteration,
                    f"cartpole",
                )

        diffusion_model_encoder.train()
        for batch in track(dataloader, description=f"Epoch {iteration}", disable=not accelerator.is_local_main_process):
            batch_data = batch["state"].squeeze(1)
            img_save(batch_data[0])
            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model_encoder.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model_encoder.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            optimizer_encoder.zero_grad()
            accelerator.backward(loss)
            counter += 1
            optimizer_encoder.step()
            max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(
                diffusion_model_encoder.model
            )
            if accelerator.is_local_main_process:
                log.info(f"iteration {iteration}, step {counter}, loss {loss.item()}")
            wandb.log(
                {
                    "encoder/iteration": iteration,
                    "encoder/step": counter,
                    "encoder/loss": loss.item(),
                    "encoder/max_param_val": max_param_val,
                }
            )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_state(
                config.parameter.accelerate_checkpoint_path, accelerator, iteration
            )
        if (iteration + 1) % 500 == 0:
            if accelerator.is_local_main_process:
                save_pt(
                    config.parameter.checkpoint_path,
                    diffusion_model_encoder.model,
                    optimizer_dynamic,
                    iteration,
                )

    ## get latent
    diffusion_model_encoder.eval()

    transformed_data = {}
    for batch in track(dataloader, description=f"Transforming states.", disable=not accelerator.is_local_main_process):

        with torch.no_grad():
            t_span = torch.linspace(0.0, 1.0, 1000)
            batch_state = batch["state"].squeeze(1)
            batch_state_transformed = diffusion_model_encoder.forward_sample(
                t_span=t_span, x=batch_state
            ).detach()
            batch["latent_state"]=batch_state_transformed
            batch_next_state = batch["next_state"].squeeze(1)
            batch_next_state_transformed = diffusion_model_encoder.forward_sample(
                t_span=t_span, x=batch_next_state
            ).detach()
            batch["latent_next_state"]=batch_next_state_transformed
            batch = accelerator.gather_for_metrics(batch)

            for key in batch:
                if key not in transformed_data:
                    transformed_data[key] = []
                transformed_data[key].append(batch[key])

    for key in transformed_data:
        transformed_data[key] = torch.cat(transformed_data[key], dim=0)

    # save transformed data
    torch.save(transformed_data, os.path.join(config.parameter.dataset_path, "transformed_data.pt"))

    # load transformed data
    transformed_data = torch.load(os.path.join(config.parameter.dataset_path, "transformed_data.pt"))

    from torch.utils.data import TensorDataset
    # keys = 'state', 'action', 'reward', 'done', 'next_state', 'next_action', 'latent_state', 'latent_next_state'
    transformed_dataset = TensorDataset(
        transformed_data["state"].cpu(),
        transformed_data["action"].cpu(),
        transformed_data["reward"].cpu(),
        transformed_data["done"].cpu(),
        transformed_data["next_state"].cpu(),
        transformed_data["next_action"].cpu(),
        transformed_data["latent_state"].cpu(),
        transformed_data["latent_next_state"].cpu(),
    )
    

    dataloader_transformed = DataLoader(
        transformed_dataset,
        batch_size=config.parameter.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    dataloader_transformed = accelerator.prepare(dataloader_transformed)

    for iteration in range(config.parameter.dynamic_iterations):

        if iteration <= last_iteration_dynamic:
            continue

        if iteration >= 0 and iteration % config.parameter.eval_freq == 0:
            diffusion_model_dynamic.eval()
            random_idx = torch.randint(0, len(transformed_dataset), (4,))
            latent_state_list = []
            action_list = []
            for idx in random_idx:
                state, action, reward, done, next_state, next_action, latent_state, latent_next_state = transformed_dataset[idx]
                latent_state_list.append(latent_state.to(accelerator.device))
                action_list.append(action.to(accelerator.device))
            latent_state = torch.stack(latent_state_list, dim=0)
            action = torch.stack(action_list, dim=0)
            t_span = torch.linspace(0.0, 1.0, 1000)
            latent_frame = []
            latent_state_temp=latent_state.to(accelerator.device)
            latent_frame.append(latent_state_temp)

            for idx in range(20):
                condition = treetensor.torch.tensor(dict(action=action.squeeze(1), state=latent_state_temp), device=accelerator.device)
                x_0 = torch.rand_like(latent_state_temp, device=accelerator.device)
                x_t = diffusion_model_dynamic.sample_forward_process(t_span=t_span, x_0=x_0, condition=condition)
                latent_state_temp = x_t[-1]
                latent_frame.append(latent_state_temp)
            
            latent_frame = torch.stack(latent_frame, dim=0)

            latent_frame_flatten = latent_frame.reshape(-1, *latent_frame.shape[2:])
            frames_flatten = diffusion_model_encoder.sample(t_span=t_span, x_0=latent_frame_flatten)
            frames = frames_flatten.reshape(*latent_frame.shape)

            frames = accelerator.gather_for_metrics(frames)
            frames_list = torch.split(frames, 1, dim=0)
            frames_list = [frame.squeeze(0) for frame in frames_list]
            if accelerator.is_local_main_process:
                render_video(frames_list, config.parameter.video_save_path, iteration, prefix="dynamic")
            # wait for all processes to finish
            accelerator.wait_for_everyone()


        diffusion_model_dynamic.train()
        for batch in track(dataloader_transformed, description=f"Epoch {iteration}", disable=not accelerator.is_local_main_process):
            batch_state, batch_action, batch_reward, batch_done, batch_next_state, batch_next_action, batch_latent_state, batch_latent_next_state = batch

            condition = treetensor.torch.tensor(dict(action=batch_action.squeeze(1), state=batch_state.squeeze(1)), device=accelerator.device)

            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model_dynamic.flow_matching_loss(x=batch_next_state.squeeze(1), condition=condition)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model_dynamic.score_matching_loss(x=batch_next_state.squeeze(1), condition=condition)
            else:
                raise NotImplementedError("Unknown loss type")
            optimizer_dynamic.zero_grad()
            accelerator.backward(loss)

            optimizer_dynamic.step()
            max_param_val, max_grad_val, min_grad_val = find_max_param_and_grad(
                diffusion_model_dynamic.model
            )
            if accelerator.is_local_main_process:
                log.info(f"iteration {iteration}, loss {loss.item()}")
            wandb.log(
                {
                    "dynamic/iteration": iteration,
                    "dynamic/loss": loss.item(),
                    "dynamic/max_param_val": max_param_val,
                }
            )

def main():
    pipeline()


if __name__ == "__main__":
    main()
