import torch
from grl.neural_network.transformers.dit import DiT
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
class generativeEncoder(nn.Module):
    def __init__(self, config):
        super(generativeEncoder, self).__init__()
        if config.image_size == 64:
            register_module(Unet_64, "Unet")
        elif config.image_size == 32:
            register_module(Unet_32, "Unet")
        self.diffusionModel = DiffusionModel(config.diffusion_model)
        self.config = config

    def sample_forward_process(self):
        import torchvision

        t_span = torch.linspace(0.0, 1.0, 32).to(self.config.diffusion_model.device)
        x_t = self.diffusionModel.sample_forward_process(t_span=t_span, batch_size=4)[
            -1, ...
        ]
        x_t = torchvision.utils.make_grid(x_t, value_range=(-1, 1), padding=0, nrow=4)
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

    def samplePictue(self, iteration=0, prefix="forwardImage"):
        return self.grlEncoder.sample_forward_process()