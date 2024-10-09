import multiprocessing as mp
import os
import signal
import sys

import matplotlib
import numpy as np
from easydict import EasyDict
from rich.progress import track
from sklearn.datasets import make_swiss_roll

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from easydict import EasyDict
from grl.generative_models.diffusion_model.diffusion_model import \
    DiffusionModel
from grl.generative_models.metric import compute_likelihood
from grl.neural_network import TemporalSpatialResidualNet, register_module
from grl.utils import set_seed
from grl.utils.log import log
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torchcfm.models.unet import UNetModel
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid


class MyModule(nn.Module):

    def __init__(self):
        super(MyModule, self).__init__()
        self.unet = UNetModel(
            dim=(1, 28, 28), num_channels=32, num_res_blocks=1, class_cond=False
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.unet(t, x, condition)
        
register_module(MyModule, "MyModule")

x_size = (1,28,28)
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
t_embedding_dim = 32
t_encoder = dict(
    type="GaussianFourierProjectionTimeEncoder",
    args=dict(
        embed_dim=t_embedding_dim,
        scale=30.0,
    ),
)
data_num=100000
config = EasyDict(
    dict(
        device=device,
        dataset=dict(
            data_num=data_num,
            noise=0.1,
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
            batch_size=4000,
            eval_freq=200,
            checkpoint_freq=200,
            checkpoint_path="/root/generative_encoder_linear_prob_2000/checkpoint-mnist-encoder",
            video_save_path="/root/generative_encoder_linear_prob_2000/video-mnist-encoder",
            device=device,
        ),
    )
)

if __name__ == "__main__":
    seed_value = set_seed()
    log.info(f"start exp with seed value {seed_value}.")
    diffusion_model = DiffusionModel(config=config.diffusion_model).to(
        config.diffusion_model.device
    )
    diffusion_model = torch.compile(diffusion_model)

    trainset = datasets.MNIST(
        "/root/generative_encoder_linear_prob_2000/mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
    )

    data_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.parameter.batch_size, shuffle=False, drop_last=False
    )

    #
    optimizer = torch.optim.Adam(
        diffusion_model.parameters(),
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
            diffusion_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            last_iteration = checkpoint["iteration"]
    else:
        last_iteration = -1

    def get_train_data(dataloader):
        while True:
            yield from dataloader

    loss_sum = 0.0
    counter = 0
    iteration = 0

    def render_video(data_list, video_save_path, iteration, fps=100, dpi=100, prefix=""):
        if not os.path.exists(video_save_path):
            os.makedirs(video_save_path)
        fig = plt.figure(figsize=(6, 6))

        ims = []

        for i, data in enumerate(data_list):

            grid = make_grid(
                data.view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
            )
            img = ToPILImage()(grid)
            im = plt.imshow(img)
            title = plt.text(0.5, 1.05, f't={i/len(data_list):.2f}', ha='center', va='bottom', transform=plt.gca().transAxes)
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

    def save_checkpoint(model, optimizer, iteration):
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
        def exit_handler(signal, frame):
            log.info("Saving checkpoint when exit...")
            save_checkpoint(model, optimizer, iteration=iterations[-1])
            log.info("Done.")
            sys.exit(0)

        signal.signal(signal.SIGINT, exit_handler)

    save_checkpoint_on_exit(diffusion_model, optimizer, history_iteration)

    t_span = torch.linspace(0.0, 1.0, 1000)

    data_list = []
    data_transform_list = []
    value_list = []

    for batch_data, batch_value in track(data_loader, description="Data Transforming"):

        with torch.no_grad():
            batch_data = batch_data.to(config.device)
            batch_value = batch_value.to(config.device)

            batch_data_transformed = diffusion_model.forward_sample(t_span=t_span, x=batch_data).detach()
            data_list.append(batch_data.cpu())
            value_list.append(batch_value.cpu())
            data_transform_list.append(batch_data_transformed.cpu())
    
    data_ = torch.cat(data_list, dim=0)
    data_transform = torch.cat(data_transform_list, dim=0)
    value_ = torch.cat(value_list, dim=0)

    # save tensor data to disk
    torch.save(data_, "/root/generative_encoder_linear_prob_2000/data.pt")
    torch.save(data_transform, "/root/generative_encoder_linear_prob_2000/data_transform.pt")
    torch.save(value_, "/root/generative_encoder_linear_prob_2000/value.pt")
