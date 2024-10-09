import os
import signal
import sys
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track

matplotlib.use("Agg")
import math
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.neural_network import register_module
from grl.utils.log import log
from matplotlib import animation
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

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

class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNet(
            cond_channels=256, depths=[2,2,2], channels=[64,64,64], attn_depths=[0,0,0],
        )
        self.conv_in = Conv3x3(3, 64)
        self.conv_out = Conv3x3(64, 3)

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.conv_in(x)
        x, d_outputs, u_outputs = self.unet(cond=t, x=x)
        x = self.conv_out(x)
        return x

register_module(MyModule, "MyModule")

def make_config(device):
    x_size = (3, 100, 100)
    data_num=100000
    t_embedding_dim = 256
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
            dataset=dict(
                data_num=data_num,
                fig_path="/workspace/test_words/figs/",
                data_info_path="/workspace/test_words/dataset.pth",
            ),
            diffusion_model=dict(
                device=device,
                x_size=x_size,
                alpha=1.0,
                solver=dict(
                    type="ODESolver",
                    args=dict(
                        library="torchdyn",
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
                            type="MyModule",
                            args={},
                        ),
                    ),
                ),
            ),
            parameter=dict(
                training_loss_type="flow_matching",
                lr=5e-4,
                data_num=data_num,
                iterations=200000,
                batch_size=200*8,
                eval_freq=50,
                checkpoint_freq=50,
                checkpoint_path="/root/generative_encoder_words_11_unet_ddp/checkpoint-word-encoder",
                video_save_path="/root/generative_encoder_words_11_unet_ddp/video-word-encoder",
                device=device,
            ),
        )
    )

    return config

class TextImageDataset(Dataset):
    def __init__(self, dataset_file, fig_path, transform=None):
        # Load the dataset information from the file
        self.dataset_info = torch.load(dataset_file)
        self.fig_path = fig_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset_info)

    def __getitem__(self, idx):
        # Get the image filename and associated metadata
        data = self.dataset_info[idx]
        image_filename = data["filename"]
        texts = data["texts"]
        one_hot_vectors = data["one_hot_vectors"]
        
        # Load the image
        image_path = os.path.join(self.fig_path, image_filename)
        image = Image.open(image_path).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert one_hot_vectors to tensor
        one_hot_vectors = torch.tensor(one_hot_vectors, dtype=torch.float32)
        
        return image, one_hot_vectors

# Define transformations (if any)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


def pipeline(config):

    # seed=set_seed(seed_value=torch.distributed.get_rank()+1)

    diffusion_model = DiffusionModel(config=config.diffusion_model).to(
        config.diffusion_model.device
    )
    diffusion_model = torch.compile(diffusion_model)

    diffusion_model.model = nn.parallel.DistributedDataParallel(diffusion_model.model, device_ids=[torch.distributed.get_rank()])

    # Create the dataset
    dataset = TextImageDataset(config.dataset.data_info_path, config.dataset.fig_path, transform=transform)

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,
    )
    assert (
        config.parameter.batch_size
        % torch.distributed.get_world_size()
        == 0
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(
            config.parameter.batch_size
            // torch.distributed.get_world_size()
        ),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    #
    optimizer = torch.optim.Adam(
        diffusion_model.model.parameters(),
        lr=config.parameter.lr,
    )

    if config.parameter.checkpoint_path is not None:

        if (
            not os.path.exists(config.parameter.checkpoint_path)
            or len(os.listdir(config.parameter.checkpoint_path)) == 0
        ):
            log.warning(
                f"Checkpoint path {config.parameter.checkpoint_path} does not exist"
            )
            last_iteration = -1
        else:
            checkpoint_files = [
                f
                for f in os.listdir(config.parameter.checkpoint_path)
                if f.endswith(".pt")
            ]
            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            checkpoint = torch.load(
                os.path.join(config.parameter.checkpoint_path, checkpoint_files[-1]),
                map_location="cpu",
            )

            from collections import OrderedDict

            #checkpoint_sorted = OrderedDict()
            #for key, value in checkpoint["model"].items():
            #    name = key.replace("module.", "")
            #    checkpoint_sorted[name] = value

            diffusion_model.load_state_dict(checkpoint["model"])
            # optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    counter = 0
    iteration = 0

    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(12, 12))

        ims = []

        for i, data in enumerate(data_list):

            grid = make_grid(
                data.view([-1, 3, 100, 100]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=2
            )
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True)
        ani.save(
            os.path.join(video_save_path, f"{prefix}_{iteration}_{torch.distributed.get_rank()}.mp4"),
            fps=fps,
            dpi=dpi,
        )
        # clean up
        plt.close(fig)
        plt.clf()

    def save_checkpoint(model, optimizer, iteration):
        if torch.distributed.get_rank() == 0:
            if not os.path.exists(config.parameter.checkpoint_path):
                os.makedirs(config.parameter.checkpoint_path)
            torch.save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    iteration=iteration,
                ),
                f=os.path.join(
                    config.parameter.checkpoint_path, f"checkpoint_{iteration}.pt"
                ),
            )

    history_iteration = [-1]

    def save_checkpoint_on_exit(model, optimizer, iterations):
        if torch.distributed.get_rank() == 0:
            def exit_handler(signal, frame):
                log.info("Saving checkpoint when exit...")
                save_checkpoint(model, optimizer, iteration=iterations[-1])
                log.info("Done.")
                sys.exit(0)

            signal.signal(signal.SIGINT, exit_handler)

    save_checkpoint_on_exit(diffusion_model, optimizer, history_iteration)

    mp_list=[]
    optimizer.zero_grad()

    for iteration in range(config.parameter.iterations):

        if iteration <= last_iteration:
            continue

        if iteration > 0 and iteration % config.parameter.eval_freq == 0:
            diffusion_model.eval()
            t_span = torch.linspace(0.0, 1.0, 1000)

            x_t = (
                diffusion_model.sample_forward_process(t_span=t_span, batch_size=4)
                .cpu()
                .detach()
            )
            #x_t = x_t.reshape(x_t.shape[0], 10, 10, 1, 28, 28)
            x_t = [
                x.squeeze(0) for x in torch.split(x_t, split_size_or_sections=1, dim=0)
            ]
            render_video(x_t, config.parameter.video_save_path, iteration, fps=100, dpi=100, prefix="generate")
            torch.distributed.barrier()
            #p = mp.Process(target=render_video, args=(x_t, config.parameter.video_save_path, iteration, 100, 100, "generate"))
            #p.start()
            #mp_list.append(p)

        diffusion_model.train()

        sampler.set_epoch(iteration)

        rank = torch.distributed.get_rank()

        for batch_data, batched_value in track(data_loader, description=f"Epoch {iteration}"):
            batch_data = batch_data.to(config.device)
            batched_value = batched_value.to(config.device)

            if config.parameter.training_loss_type == "flow_matching":
                loss = diffusion_model.flow_matching_loss(batch_data)
            elif config.parameter.training_loss_type == "score_matching":
                loss = diffusion_model.score_matching_loss(batch_data)
            else:
                raise NotImplementedError("Unknown loss type")
            loss.backward()
            counter += 1
            optimizer.step()
            optimizer.zero_grad()
            

            log.info(
                f"Ranks {rank}, iteration {iteration}, step {counter}, loss {loss.item()}"
            )

        history_iteration.append(iteration)

        if (iteration + 1) % config.parameter.checkpoint_freq == 0:
            save_checkpoint(diffusion_model, optimizer, iteration)
            torch.distributed.barrier()

    for p in mp_list:
        p.join()


def main(world_size=8):
    torch.distributed.init_process_group("nccl", world_size=world_size)
    device = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    config = make_config(device=device)
    log.info(
        f"Starting rank={torch.distributed.get_rank()}, world_size={torch.distributed.get_world_size()}."
    )

    log.info("config: \n{}".format(config))
    pipeline(config)

if __name__ == "__main__":
    main()
