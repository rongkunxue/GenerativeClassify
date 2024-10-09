import wandb
from easydict import EasyDict
from accelerate import Accelerator
from train import train

def make_config(device):
    method="Finetune"
    type="GenerativeClassifyUNet"
    classes = 200
    image_size = 64
    project_name = "Classify_Tiny_imagent_mixup"
    config = EasyDict(
        dict(
            PROJECT_NAME=project_name,
            DEVICE=device,
            DATA=dict(
                batch_size=180,
                classes=classes,
                img_size=image_size,
                dataset_path="",
                checkpoint_path=f"./{project_name}/checkpoint",
                video_save_path=f"./{project_name}/video",
                dataset="Tinyimagenet",
                AUG=dict(
                    interpolation="bicubic",
                    color_jitter=0.4,
                    auto_augment="rand-m9-mstd0.5-inc1",
                    reprob=0.25,
                    remode="pixel",
                    recount=1,
                ),
                MIXUP=dict(
                    mixup=0.8,
                    cutmix=1.0,
                    cutminmax=None,
                    mixup_prob=1.0,
                    mixup_switch_prob=0.5,
                    mixup_mode="batch",
                )
            ),
            MODEL=dict(
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
                        type="gvp",
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
                loss_function="SoftTargetCrossEntropy", #LabelSmoothingCrossEntropy or SoftTargetCrossEntropy
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
                eval_freq=5,
                generative_freq=50,
                checkpoint_freq=10,
            ),
        )
    )
    return config


if __name__ == "__main__":
    accelerator = Accelerator()
    config = make_config(accelerator.device)
    import wandb

    wandb.init(
        project=config.PROJECT_NAME,
        config=config,
    )
    train(config, accelerator)