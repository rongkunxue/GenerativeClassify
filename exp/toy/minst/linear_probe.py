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
from grl.neural_network import MultiLayerPerceptron
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

x_size = (1,28,28)
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")

config = EasyDict(
    dict(
        device=device,
        model=dict(
            hidden_sizes=[28*28,],
            output_size=10,
            activation="relu",
        ),
        model_combine=dict(
            hidden_sizes=[2*28*28,],
            output_size=10,
            activation="relu",
        ),
        parameter=dict(
            training_loss_type="flow_matching",
            lr=5e-4,
            train_data_num=50000,
            batch_size=3000,
            eval_freq=200,
            checkpoint_freq=200,
            checkpoint_path="/root/generative_encoder_linear_prob_2000/checkpoint-mnist-probe",
            video_save_path="/root/generative_encoder_linear_prob_2000/video-mnist-probe",
            dataset_path="/root/generative_encoder_linear_prob_2000/dataset",
            device=device,
        ),
    )
)

if __name__ == "__main__":
    
    # torch.save(data_, "/root/generative_encoder_linear_prob_2000/data.pt")
    # torch.save(data_transform, "/root/generative_encoder_linear_prob_2000/data_transform.pt")
    # torch.save(value_, "/root/generative_encoder_linear_prob_2000/value.pt")

    # load dataset from path
    data_ = torch.load("/root/generative_encoder_linear_prob_2000/data.pt")
    data_transform = torch.load("/root/generative_encoder_linear_prob_2000/data_transform.pt")
    value_ = torch.load("/root/generative_encoder_linear_prob_2000/value.pt")

    #split data_ into train and test by half
    train_data = data_[:config.parameter.train_data_num]
    test_data = data_[config.parameter.train_data_num:]
    train_data_transform = data_transform[:config.parameter.train_data_num]
    test_data_transform = data_transform[config.parameter.train_data_num:]
    train_value = value_[:config.parameter.train_data_num]
    test_value = value_[config.parameter.train_data_num:]

    train_combine_data = torch.cat((train_data, train_data_transform), dim=1)
    test_combine_data = torch.cat((test_data, test_data_transform), dim=1)


    # Define model, criterion, and optimizer
    linear_probe_1 = MultiLayerPerceptron(
        **config.model,
    ).to(config.device)

    linear_probe_2 = MultiLayerPerceptron(
        **config.model,
    ).to(config.device)

    linear_probe_3 = MultiLayerPerceptron(
        **config.model_combine,
    ).to(config.device)


    criterion = nn.CrossEntropyLoss()

    optimizer_1 = torch.optim.Adam(
        linear_probe_1.parameters(),
        lr=config.parameter.lr,
    )

    optimizer_2 = torch.optim.Adam(
        linear_probe_2.parameters(),
        lr=config.parameter.lr,
    )

    optimizer_3 = torch.optim.Adam(
        linear_probe_3.parameters(),
        lr=config.parameter.lr,
    )

    # Create data loaders
    train_loader = DataLoader(TensorDataset(train_data, train_value), batch_size=config.parameter.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(test_data, test_value), batch_size=config.parameter.batch_size, shuffle=False, drop_last=True)
    train_loader_transform = DataLoader(TensorDataset(train_data_transform, train_value), batch_size=config.parameter.batch_size, shuffle=True, drop_last=True)
    test_loader_transform = DataLoader(TensorDataset(test_data_transform, test_value), batch_size=config.parameter.batch_size, shuffle=False, drop_last=True)

    train_loader_combine = DataLoader(TensorDataset(train_combine_data, train_value), batch_size=config.parameter.batch_size, shuffle=True, drop_last=True)
    test_loader_combine = DataLoader(TensorDataset(test_combine_data, test_value), batch_size=config.parameter.batch_size, shuffle=False, drop_last=True)


    # Training function
    def train_model(model, optimizer, train_loader, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                optimizer.zero_grad()
                outputs = model(inputs.view(inputs.size(0), -1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return loss.item()

    # Evaluation function
    def evaluate_model(model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs.view(inputs.size(0), -1))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy * 100

    import wandb
    with wandb.init(project="generative_encoder_linear_prob_2000") as run:

        for i in range(10000):
            # Train and evaluate the first model on original MNIST data
            loss_origin = train_model(linear_probe_1, optimizer_1, train_loader, num_epochs=1)
            acc_origin = evaluate_model(linear_probe_1, test_loader)

            # Train and evaluate the second model on transformed MNIST data
            loss_transformed = train_model(linear_probe_2, optimizer_2, train_loader_transform, num_epochs=1)
            acc_transformed = evaluate_model(linear_probe_2, test_loader_transform)

            # Train and evaluate the third model on combined data
            loss_combine = train_model(linear_probe_3, optimizer_3, train_loader_combine, num_epochs=1)
            acc_combine = evaluate_model(linear_probe_3, test_loader_combine)

            print(f"Epoch {i+1}, Loss (original): {loss_origin:.4f}, Accuracy (original): {acc_origin:.2f}%, Loss (transformed): {loss_transformed:.4f}, Accuracy (transformed): {acc_transformed:.2f}%, Loss (combined): {loss_combine:.4f}, Accuracy (combined): {acc_combine:.2f}%")

            wandb.log(
                data=dict(
                    epoch=i,
                    accuracy_original=acc_origin,
                    accuracy_transformed=acc_transformed,
                    acc_combine=acc_combine,
                    loss_original=loss_origin,
                    loss_transformed=loss_transformed,
                    loss_combine=loss_combine,
                ),
                commit=True,
            )

    wandb.finish()

