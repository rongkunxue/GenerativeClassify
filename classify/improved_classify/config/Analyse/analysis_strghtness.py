import os
from easydict import EasyDict
from accelerate import Accelerator
from picture_analysis import picture_analysis



if __name__ == "__main__":
    accelerator = Accelerator()
    import wandb

    wandb.init(
        project=config.PROJECT_NAME,
        config=config,
    )
    picture_analysis(config, accelerator)