import torch
from grl.neural_network.unet import unet_2D
from grl.neural_network import register_module
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
import torch.nn as nn
from improved_utilities import (
    find_max_param_and_grad,
    img_save,
    load_pt,
    load_state,
    save_pt,
    save_state,
    imagenet_save,
)


class Unet_64(nn.Module):
    def __init__(self):
        super(Unet_64, self).__init__()
        self.unet = unet_2D(
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


class Unet_32(nn.Module):
    def __init__(self):
        super(Unet_32, self).__init__()
        self.unet = unet_2D(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=tuple([2, 4]),
            dropout=0,
            channel_mult=(1, 2, 2, 2),
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


class classifyHead(nn.Module):
    def __init__(self, config):
        super(classifyHead, self).__init__()
        self.flatten=nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(3 * config.image_size * config.image_size),
            nn.Linear(3 * config.image_size * config.image_size, 3 * config.image_size * config.image_size),
            nn.Tanh(),
            nn.Linear(3 * config.image_size * config.image_size,  config.classes, bias=False),
        )
        self.flatten = nn.Flatten()
        self.config = config

    def forward(self, x):
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class generativeEncoder(nn.Module):
    def __init__(self, config):
        super(generativeEncoder, self).__init__()
        if config.image_size == 64:
            register_module(Unet_64, "GenerativeClassifyUNet")
        elif config.image_size == 32:
            register_module(Unet_32, "GenerativeClassifyUNet")
        self.diffusionModel = DiffusionModel(config.diffusion_model)
        self.config = config

    def sample_forward_process(self):
        self.diffusionModel.model.eval()
        import torchvision
        t_span = torch.linspace(0.0, 1.0, 32).to(self.config.diffusion_model.device)
        x_t = self.diffusionModel.sample_forward_process(t_span=t_span, batch_size=4)[
            -1, ...
        ]
        return x_t

    def sample_backward_process(self, x, with_grad=False):
        t_span = torch.linspace(0.0, 1.0, self.config.t_span).to(x.device)
        x_t = self.diffusionModel.forward_sample(
            x=x, t_span=t_span, with_grad=with_grad
        )
        return x_t


class generativeClassify(nn.Module):
    def __init__(self, config):
        super(generativeClassify, self).__init__()
        self.grlEncoder = generativeEncoder(config)
        self.grlHead = classifyHead(config)
        self.config = config

    def forward(self, x):
        images = self.grlEncoder.sample_backward_process(x=x, with_grad=True)
        output = self.grlHead(images)
        return output

    def matchingLoss(self, x):
        return self.grlEncoder.diffusionModel.flow_matching_loss(x)

    def samplePicture(self, iteration=0, prefix="forwardImage"):
        return self.grlEncoder.sample_forward_process()
