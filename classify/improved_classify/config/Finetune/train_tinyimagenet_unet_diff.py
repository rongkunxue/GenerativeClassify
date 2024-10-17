import wandb
from easydict import EasyDict
from accelerate import Accelerator
from train import train

def make_config(device):
    method="Finetune"
    type="GenerativeClassifyUNet"
    classes = 200
    image_size = 64
    project_name = "Classify_Tiny_imagent_label_smooth"
    config = EasyDict(
        dict(
            PROJECT_NAME=project_name,
            DEVICE=device,
            DATA=dict(
                batch_size=180,
                classes=classes,
                img_size=image_size,
                dataset_path="/root/data/tiny-imagenet-200",
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
                loss_function="LabelSmoothingCrossEntropy", #LabelSmoothingCrossEntropy or SoftTargetCrossEntropy
                label_smoothing=0.1,
                training_loss_type="flow_matching",
                
                optimizer_type="adamw",
                lr=1.25e-4,
                warmup_lr=1.25e-07,
                min_lr=1.25e-6,
                
                iteration=2000,
                warmup_iteration=5,
                decay_iteration=5,
                
                
                
                device=device,
                OPTIMIZER=dict(
                    eps=1e-08,
                    betas=(0.9, 0.999),
                    momentum=0.9,
                ),
                LR_SCHEDULER=dict(
                    name="cosine",
                    decay_rate=0.1,
                    warmup_prefix=True,
                    gamma=0.1,
                    multisteps=[],
                ),
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
    
    num_processes = accelerator.num_processes
    config.TRAIN.lr=config.TRAIN.lr*num_processes*config.DATA.batch_size/512
    config.TRAIN.warmup_lr=config.TRAIN.warmup_lr*num_processes*config.DATA.batch_size/512
    config.TRAIN.min_lr=config.TRAIN.min_lr*num_processes*config.DATA.batch_size/512
    
    wandb.init(
        project=config.PROJECT_NAME,
        config=config,
    )
    train(config, accelerator)