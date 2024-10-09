import matplotlib
import matplotlib.pyplot as plt
import wandb
import numpy as np
from sklearn.manifold import TSNE


def apply_tsne_and_plot(features, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i in range(10):
        indices = labels == i
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            label=f"Class {i}",
            alpha=0.5,
        )
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.savefig(f"{title}.png")
    plt.show()
    wandb.log({title: wandb.Image(f"{title}.png")})


def extract_features(model, data_loader, layer_name):
    features = []
    labels = []

    def hook(module, input, output):
        features.append(output.cpu().detach().numpy())

    layer = dict([*model.named_modules()])[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for images, target in data_loader:
            model(images.to(accelerator.device))
            labels.extend(target.cpu().numpy())

    hook_handle.remove()
    features = np.concatenate(features, axis=0)
    return features, np.array(labels)
