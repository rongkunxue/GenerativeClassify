import wandb
from easydict import EasyDict
from accelerate import Accelerator
from train import train

def make_config(device):
    method="Finetune"
    type="GenerativeClassifyUNet_ICFM"
    classes = 10
    image_size = 32
    project_name = "Classify_CIFAR-10"
    config = EasyDict(
        dict(
            PROJECT_NAME=project_name,
            DEVICE=device,
            DATA=dict(
                batch_size=180,
                classes=classes,
                img_size=image_size,
                dataset_path="/home/xrk/EXP/data/CIFAR-10",
                checkpoint_path=f"./{project_name}/checkpoint",
                video_save_path=f"./{project_name}/video",
                dataset="CIFAR-10",
                AUG=dict(
                    interpolation="bicubic",
                    color_jitter=0.4,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    reprob=0.25,
                    remode="pixel",
                    recount=1,
                ),
            ),
            MODEL=dict(
                method=method,
                type=type,
                t_span=20,
                image_size=image_size,
                classes=classes,
                diffusion_model=dict(
                    device=device,
                    x_size=(3, image_size, image_size),
                    alpha=1.0,
                    solver=dict(
                        type="ODESolver",
                        args=dict(
                            library="torchdiffeq_adjoint",
                        ),
                    ),
                    path=dict(
                        sigma=0.0,
                    ),
                    model=dict(
                        type="velocity_function",
                        args=dict(
                            backbone=dict(
                                type=type,
                                args={},
                            ),
                        ),
                    ),
                ),
            ),
            TRAIN=dict(
                method=method,
                loss_function="LabelSmoothingCrossEntropy", #LabelSmoothingCrossEntropy or SoftTargetCrossEntropy
                label_smoothing=0.1,
                training_loss_type="flow_matching",
                optimizer_type="adam",
                lr=1e-4,
                iteration=2000,
                device=device,
            ),
            TEST=dict(
                seed=0,
                crop=True,
                eval_freq=100,
                generative_freq=10,
                checkpoint_freq=10,
            ),
        )
    )
    return config


if __name__ == "__main__":
    accelerator = Accelerator()
    config = make_config(accelerator.device)

    wandb.init(
        project=config.PROJECT_NAME,
        config=config,
    )
    train(config, accelerator)