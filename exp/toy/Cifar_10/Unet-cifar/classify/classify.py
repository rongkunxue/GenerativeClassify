import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm
import wandb
from accelerate import Accelerator
from improved_utilities import create_data_loader, img_save_batch, save_pt
from rich.progress import BarColumn, Progress, TextColumn
from torch.utils.data import Dataset

# Initialize the accelerator
accelerator = Accelerator()
# Initialize wandb
wandb.init(project="cifar10_classification_4")


class CombinedDataset(Dataset):
    def __init__(self, path, transform=None):
        data = torch.load(path)
        self.data = data["data"]
        self.data_transform = data["data_transform"]
        self.value = data["value"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data_transform = self.data_transform[idx]

        if self.transform:
            data = self.transform(data)
            # data_transform = self.transform(data_transform)

        combined_data = torch.cat((data, data_transform), dim=0)
        return combined_data, self.value[idx]


class OrginDataset(Dataset):
    def __init__(self, path, transform=None):
        data = torch.load(path)
        self.data = data["data"]
        self.value = data["value"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.value[idx]


class TransformDataset(Dataset):
    def __init__(self, path, transform=None):
        data = torch.load(path)
        self.data = data["data_transform"]
        self.value = data["value"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data, self.value[idx]


# Define the data transformations
transform_origin = transforms.Compose(
    [
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
        transforms.Lambda(lambda x: x * 255),
        transforms.Lambda(lambda x: x.clamp(0, 255)),
        transforms.Lambda(lambda x: x.byte()),
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class SimpleModel(nn.Module):
    def __init__(self, size):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * size * size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x


# import torch.nn.functional as F
# class SimpleModel(nn.Module):
#     def __init__(self, size):
#         super(SimpleModel, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(3 * size * size, 512)  # 第一个全连接层
#         self.fc2 = nn.Linear(512, 256)  # 第二个全连接层
#         self.fc3 = nn.Linear(256, 128)  # 第三个全连接层
#         self.fc4 = nn.Linear(128, 10)  # 输出层

#     def forward(self, x):
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))  # ReLU 激活函数
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x


def initialize_model(name, size=224):
    if name == "Resnet50":
        model = torchvision.models.resnet50(pretrained=False)
        inchannel = model.fc.in_features
        model.fc = nn.Linear(inchannel, 10)
        return model
    elif name == "linear":
        model = SimpleModel(size)
        return model
    else:
        return -1


def train_and_evaluate(
    model, train_loader, test_loader, optimizer, loss_fn, epoch, name
):
    model.train()
    training_loss = 0.0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task("[red]Training", total=len(train_loader))

        for images, labels in train_loader:
            predict_label = model(images)
            loss = loss_fn(predict_label, labels)
            optimizer.zero_grad()
            loss.backward()  # Corrected from accelerator.backward to loss.backward
            optimizer.step()
            training_loss += loss.item()
            progress.advance(task)

    wandb.log(
        {
            f"{name}/epoch": epoch + 1,
            f"{name}/training_loss": training_loss / len(train_loader),
        },
        commit=False,
    )
    train_accuracy = evaluate(model, train_loader)
    test_accuracy = evaluate(model, test_loader)
    wandb.log(
        {
            f"{name}/epoch": epoch + 1,
            f"{name}/test_accuracy": test_accuracy,
            f"{name}/train_accuracy": train_accuracy,
        },
        commit=True,
    )
    return 0


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model(images)
            outputs, targets = accelerator.gather_for_metrics((outputs, targets))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total * 100
    return accuracy


batch_size = 715

TEST_PATH = (
    "/root/generativeencoder/exp/toy/Cifar_10/dataset/cifar_transformed_test_tensor.pt"
)
TRAIN_PATH = (
    "/root/generativeencoder/exp/toy/Cifar_10/dataset/cifar_transformed_train_tensor.pt"
)
train_origin_dataset = OrginDataset(TRAIN_PATH, transform=transform_origin)
test_origin_dataset = OrginDataset(TEST_PATH, transform=transform_origin)
train_combine_dataset = CombinedDataset(TRAIN_PATH, transform_origin)
test_combine_dataset = CombinedDataset(TEST_PATH, transform_origin)
train_transform_dataset = TransformDataset(TRAIN_PATH)
test_transform_dataset = TransformDataset(TEST_PATH)

# Data loaders using the function
train_origin_loader = create_data_loader(train_origin_dataset, batch_size, shuffle=True)
test_origin_loader = create_data_loader(test_origin_dataset, batch_size, shuffle=False)
train_combine_loader = create_data_loader(
    train_combine_dataset, batch_size, shuffle=True
)
test_combine_loader = create_data_loader(
    test_combine_dataset, batch_size, shuffle=False
)
train_transform_loader = create_data_loader(
    train_transform_dataset, batch_size, shuffle=True
)
test_transform_loader = create_data_loader(
    test_transform_dataset, batch_size, shuffle=False
)

# Initialize models, optimizers and loss functions
models = {
    "origin": initialize_model("Resnet50"),
    # "combine": initialize_model(),
    "transform": initialize_model("linear", 32),
}

optimizers = {
    "origin": torch.optim.Adam(
        models["origin"].parameters(),
        lr=5e-4,
    ),
    "transform": torch.optim.Adam(
        models["transform"].parameters(),
        lr=5e-4,
    ),
}
loss_fn = torch.nn.CrossEntropyLoss()

train_loaders = {
    "origin": train_origin_loader,
    # "combine": train_combine_loader,
    "transform": train_transform_loader,
}

test_loaders = {
    "origin": test_origin_loader,
    # "combine": test_combine_loader,
    "transform": test_transform_loader,
}

# Prepare models and optimizers with accelerator outside the loop
for name in models.keys():
    models[name], optimizers[name], train_loaders[name], test_loaders[name] = (
        accelerator.prepare(
            models[name], optimizers[name], train_loaders[name], test_loaders[name]
        )
    )

epochs = 500000
for epoch in range(epochs):
    for name in models.keys():
        train_and_evaluate(
            models[name],
            train_loaders[name],
            test_loaders[name],
            optimizers[name],
            loss_fn,
            epoch,
            name,
        )

    if (epoch + 1) % 10 == 0:  # Increase the frequency to every 100 epochs
        if accelerator.is_local_main_process:
            for name in models.keys():
                save_pt(
                    f"/root/generativeencoder/exp/toy/Cifar_10/Unet-cifar/classify/classify_{name}",
                    models[name],
                    optimizers[name],
                    epoch,
                )

# Finish the wandb run
wandb.finish()
