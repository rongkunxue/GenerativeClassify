import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from improved_utilities import img_save
def visualize_pca_cifar10(images, labels, save_path, n_components=1000, plot_components=10, dpi=300):
    """
    Visualizes the PCA of CIFAR-10 dataset.
    
    Parameters:
    - images: torch.Tensor, the images from CIFAR-10 dataset.
    - labels: torch.Tensor, the labels corresponding to the images.
    - save_path: str, the path to save the resulting plot.
    - n_components: int, the number of PCA components to reduce to.
    - plot_components: int, the number of PCA components to plot.
    - dpi: int, the resolution of the saved plot.
    """
    # Flatten the image data
    images_flattened = images.view(images.size(0), -1)

    # Convert the data to a numpy array
    images_flattened_np = images_flattened.numpy()

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    images_pca = pca.fit_transform(images_flattened_np)

    # Get PCA explained variance
    explained_variance = pca.explained_variance_
    print("PCA explained variance:", explained_variance)

    # Convert PCA results to a DataFrame and add labels
    df_pca = pd.DataFrame(data=images_pca[:, :plot_components], columns=[f'PC{i+1}' for i in range(plot_components)])
    df_pca['label'] = labels

    # Visualize the first plot_components dimensions of PCA results
    sns.pairplot(df_pca, hue='label', palette='tab10', plot_kws={'alpha': 0.5})
    plt.suptitle(f'Pairplot of PCA (First {plot_components} Components) of CIFAR-10', y=1.02)

    # Save the image with the specified DPI
    plt.savefig(save_path, dpi=dpi)

    # Display the plot
    plt.show()



# Define the transform (normalize the dataset)
data = torch.load("/root/dataset/cifar_transformed_train.pt")
images_ori,images, labels=data["data"],data["data_transform"],data["value"]

# Load the CIFAR-10 dataset
train_loader = DataLoader(TensorDataset(images_ori,images, labels), batch_size=10000, shuffle=True, drop_last=True)
# Select one batch of data
dataiter = iter(train_loader)
images_ori,images, labels = next(dataiter)

img_save(images_ori[0],"/root/Github/generativeencoder/exp/toy/Cifar_10/Unet-cifar/pca",prefix="ori")
img_save(images[0],"/root/Github/generativeencoder/exp/toy/Cifar_10/Unet-cifar/pca",prefix="tran")
visualize_pca_cifar10(
    images=images_ori,
    labels=labels,
    save_path='/root/Github/generativeencoder/exp/toy/Cifar_10/Unet-cifar/pca/pca_cifar10_ori.png',
    n_components=1000,
    plot_components=10,
    dpi=300
)
visualize_pca_cifar10(
    images=images,
    labels=labels,
    save_path='/root/Github/generativeencoder/exp/toy/Cifar_10/Unet-cifar/pca/pca_cifar10_tran.png',
    n_components=1000,
    plot_components=10,
    dpi=300
)