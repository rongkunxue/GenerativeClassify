import torch
from grl.neural_network.transformers.dit import DiT
from grl.neural_network import register_module
from grl.generative_models.diffusion_model.diffusion_model import DiffusionModel
import torch.nn as nn


class DiT_32(nn.Module):
    def __init__(self):
        super(DiT_32, self).__init__()
        self.dit = DiT(
            input_size=32,
            patch_size=2,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            learn_sigma=False,
            condition=False,
        )
        
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.dit(t, x, None)
        return x
    
class DiT_64(nn.Module):
    def __init__(self):
        super(DiT_64, self).__init__()
        self.dit = DiT(
            input_size=64,
            patch_size=4,
            in_channels=3,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            learn_sigma=False,
            condition=False,
        )
        
    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.dit(t, x, None)
        return x
    
class classifyHead(nn.Module):
    def __init__(self, config):
        super(classifyHead, self).__init__()
        self.mypool=nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)) ,
            nn.Flatten(),
        )
        self.classifier=nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Linear(768,  config.classes, bias=False),
        )
        self.config = config

    def forward(self, x):
        x = self.mypool(x)
        x = self.classifier(x)
        return x


class generativeEncoder(nn.Module):
    def __init__(self, config):
        super(generativeEncoder, self).__init__()
        if config.image_size == 64:
            register_module(DiT_64, "GenerativeClassifyDiT_Diff")
        elif config.image_size == 32:
            register_module(DiT_32, "GenerativeClassifyDiT_Diff")
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
        if hasattr(self.config, "t_cutoff"):
            t_span = torch.linspace(0.0, 1.0, self.config.t_span, device=x.device)[:self.config.t_cutoff]
        else :
            t_span = torch.linspace(0.0, 1.0, self.config.t_span, device=x.device)
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

    def forward(self, x,with_grad=True):
        images = self.grlEncoder.sample_backward_process(x=x, with_grad=with_grad)
        output = self.grlHead(images)
        return output

    def matchingLoss(self, x):
        return self.grlEncoder.diffusionModel.flow_matching_loss(x)

    def samplePicture(self):
        return self.grlEncoder.sample_forward_process()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
