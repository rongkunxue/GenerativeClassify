import torch
import os
import logging
from torch.utils.data import DataLoader


def save_state(path, accelerator, iteration):
    if accelerator.is_main_process:
        if not os.path.exists(path):
            os.makedirs(path)
    accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(path, f"checkpoint_{iteration}_save"))


def save_pt(path, model, optimizer, iteration, accelerator=None,prefix=""):
    if not os.path.exists(path):
        os.makedirs(path)
    if  accelerator is not None:
        model= accelerator.unwrap_model(model)
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            iteration=iteration,
        ),
        f=os.path.join(path, f"checkpoint_{iteration}{prefix}.pt"),
    )


def load_pt(path, accelerator=None, model=None):
    last_iteraion = -1
    checkpoint_path = path
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path) or not os.listdir(checkpoint_path):
            logging.warning(
                f"Checkpoint path {checkpoint_path} does not exist or is empty"
            )
            return last_iteraion

        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")],
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
        if not checkpoint_files:
            logging.warning(f"No checkpoint files found in {checkpoint_path}")
            return last_iteraion

        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1])
        if accelerator is not None:
            with accelerator.main_process_first():
                checkpoint = torch.load(checkpoint_file, map_location="cpu")
                last_iteraion = checkpoint.get("iteration", -1)
                ori_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
                ori_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ori_state_dict.items()}
                model.load_state_dict(ori_state_dict)
                logging.warning(f"{last_iteraion}_checkpoint files has been loaded")
        else:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
            last_iteraion = checkpoint.get("iteration", -1)
            ori_state_dict = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            ori_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ori_state_dict.items()}
            model.load_state_dict(ori_state_dict)
            logging.warning(f"{last_iteraion}_checkpoint files has been loaded")
        return last_iteraion
    return last_iteraion


def load_state(path, accelerator):
    last_iteraion = -1
    if path is not None:
        checkpoint_path = path

        if not os.path.exists(checkpoint_path) or len(os.listdir(checkpoint_path)) == 0:
            logging.warning(
                f"Checkpoint path {checkpoint_path} does not exist or is empty"
            )
            return last_iteraion
        else:
            checkpoint_files = [
                f for f in os.listdir(checkpoint_path) if f.endswith("_save")
            ]
            if not checkpoint_files:
                logging.warning(f"No checkpoint files found in {checkpoint_path}")
                return last_iteraion

            checkpoint_files = sorted(
                checkpoint_files, key=lambda x: int(os.path.basename(x).split("_")[1])
            )
            accelerator.load_state(os.path.join(checkpoint_path, checkpoint_files[-1]))
            return int(os.path.basename(checkpoint_files[-1]).split("_")[1]) + 1
    else:
        logging.warning("No checkpoint path specified in the configuration")
        return last_iteraion


def find_max_param_and_grad(model):
    max_param_val = -float("inf")
    max_grad_val = -float("inf")
    min_grad_val = float("inf")
    for name, param in model.named_parameters():
        max_param_val = max(max_param_val, param.data.abs().max().item())
        if param.grad is not None:
            max_grad_val = max(max_grad_val, param.grad.abs().max().item())
            min_grad_val = min(min_grad_val, param.grad.abs().min().item())
    if max_grad_val == -float("inf"):
        max_grad_val = None
    if min_grad_val == float("inf"):
        min_grad_val = None
    return max_param_val, max_grad_val, min_grad_val



def create_data_loader(dataset, batch_size, shuffle, num_workers=4, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
