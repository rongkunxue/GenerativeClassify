import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Define the text to be added
text = "But"

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Create a list to store the dataset information
dataset_images = []
dataset_labels = []
dataset_centers = []

# Loop to create one image
image = Image.new("RGB", (64, 64), color="white")

# Initialize the drawing context
draw = ImageDraw.Draw(image)

# Use a truetype font if you have one available
font = ImageFont.truetype(
    "/root/generativeencoder/exp/toy/Best_word/TimesNewRoman.ttf", 10
)  # Reduced font size
# Define the margin
margin = 1

# Calculate the bounding box of the text
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Ensure the text fits within the image
if text_width + 2 * margin < 64 and text_height + 2 * margin < 64:
    # Position the text at the top left corner
    # position = (margin, margin)
    position = (64 - text_width - margin-4, 64 - text_height - margin-4)
    draw.text(position, text, fill="black", font=font)
    # Calculate the center of the text bounding box
    center_x = (position[0] + position[0] + text_width) / 2
    center_y = (position[1] + position[1] + text_height) / 2
    centers = [(center_x, center_y)]
    labels = [0]
else:
    print(f"Warning: Text '{text}' is too large to fit within the image.")
    labels = [101]
    centers = [(65, 65)]

# Convert image to numpy array in the format [64, 64, 3]
img_np = np.array(image)
image.save("/root/generativeencoder/exp/toy/Best_word/data/image5.png")

# Add the image, label, and center information to the dataset
# dataset_images.append(img_np)
# dataset_labels.append(labels)
# dataset_centers.append(centers)

# # Convert lists to numpy arrays
# dataset_images = np.array(dataset_images)
# dataset_labels = np.array(dataset_labels, dtype=np.float32)
# dataset_centers = np.array(dataset_centers, dtype=np.float32)

# # Save the dataset using PyTorch with pickle protocol 4
# with open("/root/generativeencoder/exp/toy/Best_word/data/words_dataset_64x64.pt", "wb") as f:
#     torch.save(
#         {
#             "images": dataset_images,
#             "labels": dataset_labels,
#             "centers": dataset_centers,
#         },
#         f,
#         pickle_protocol=4,
#     )
