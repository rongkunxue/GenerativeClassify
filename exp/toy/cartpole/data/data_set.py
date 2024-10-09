import cv2
import dmc2gym
import numpy as np
import torch
from improved_utilities import ReplayMemoryDataset, SampleData, resize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)
import os

os.environ["MUJOCO_GL"] = "egl"
env = dmc2gym.make(
    domain_name="cartpole",
    task_name="balance",
    seed=1,
    from_pixels=True,
    visualize_reward=False,
)
myData = SampleData(
    0, "/mnt/nfs/xuerongkun/dataset/Carpole", 10, 1050, (3, 64, 64), (1,)
)
myData.start_sample_game(env, transform)
