import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define the transform (normalize the dataset)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='/mnt/nfs/xuerongkun/dataset/Cifar', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=2)

# Select one batch of data
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Flatten the image data
images_flattened = images.view(images.size(0), -1)

# Convert the data to a numpy array
images_flattened_np = images_flattened.numpy()

# Apply PCA for dimensionality reduction
pca = PCA(n_components=1000)
images_pca = pca.fit_transform(images_flattened_np)

# Get PCA explained variance
explained_variance = pca.explained_variance_
print("PCA explained variance:", explained_variance)

# Convert PCA results to a DataFrame and add labels
df_pca = pd.DataFrame(data=images_pca[:, :10], columns=[f'PC{i+1}' for i in range(10)])
df_pca['label'] = labels

# Visualize the first 10 dimensions of PCA results
sns.pairplot(df_pca, hue='label', palette='tab10', plot_kws={'alpha': 0.5})
plt.suptitle('Pairplot of PCA (First 10 Components) of CIFAR-10', y=1.02)

# Save the image with DPI=300
plt.savefig('/root/Github/generativeencoder/exp/toy/Cifar_10/Unet-cifar/pca/pairplot_pca_cifar10.png', dpi=300)

# Display the plot
plt.show()