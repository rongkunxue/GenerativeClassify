import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Define the text to be added
texts = [
    "Amazing",
    "Great job",
    "Well done",
    "Keep it up",
    "Fantastic",
    "Good luck",
    "All the best",
    "Cheers",
    "Way to go",
    "Bravo",
    "Outstanding",
    "Marvelous",
    "Incredible",
    "Splendid",
    "Awesome",
    "Excellent",
    "But",
    "Wonderful",
    "Impressive",
    "Remarkable",
    "Sensational",
    "Stunning",
    "Terrific",
    "Phenomenal",
    "Extraordinary",
    "Magnificent",
    "Superb",
    "Brilliant",
    "Spectacular",
    "Wonderful",
    "Cool",
    "Nice",
    "Sweet",
    "Rad",
    "Neat",
    "Groovy",
    "Nifty",
    "Bus",
    "Wow",
    "Yay",
    "Hooray",
    "Woohoo",
    "Bravo",
    "Huzzah",
    "Yippee",
    "Unbelievable",
    "Astounding",
    "Breathtaking",
    "Stupendous",
    "Dazzling",
    "Jaw-dropping",
    "Mind-blowing",
    "Epic",
    "Legendary",
    "Stellar",
    "Astonishing",
    "Awe-inspiring",
    "Superb",
    "Fascinating",
    "Marvelous",
    "Wonderful",
    "Incredible",
    "Magnificent",
    "Remarkable",
    "Stunning",
    "Fabulous",
    "Awesome",
    "Outstanding",
    "Amazing",
    "Fantastic",
    "Wonderful",
    "Marvelous",
    "Phenomenal",
    "Terrific",
    "Superb",
    "Excellent",
    "Spectacular",
    "Stellar",
    "Amazing",
    "Fantastic",
    "Marvelous",
    "Great work",
    "Splendid effort",
    "Bravo",
    "Exceptional",
    "Incredible",
    "Fantastic",
    "Excellent",
    "Superb",
    "Outstanding",
    "Remarkable",
    "Awesome",
    "Brilliant",
    "Stellar",
    "Splendid",
]
texts = sorted(texts)

# Create a mapping from text to index for direct indexing
text_to_index = {text: i for i, text in enumerate(texts)}


# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


# Function to check if two bounding boxes overlap
def overlap(bbox1, bbox2):
    return not (
        bbox1[2] < bbox2[0]
        or bbox1[0] > bbox2[2]
        or bbox1[3] < bbox2[1]
        or bbox1[1] > bbox2[3]
    )


# Create a list to store the dataset information
dataset_images = []
dataset_labels = []
dataset_centers = []

# Loop to create multiple images
for i in range(20000):  # Outer loop for 10000 sets of 6 samples
    num_sentences = 2
    # Randomly select the sentences without repetition
    selected_texts = random.sample(texts, num_sentences)
    for j in range(3):  # Inner loop for 6 samples each
        # Create a new 64x64 image with the selected background color
        image = Image.new("RGB", (64, 64), color="white")

        # Initialize the drawing context
        draw = ImageDraw.Draw(image)

        # Use a truetype font if you have one available
        font = ImageFont.truetype(
            "/root/generativeencoder/exp/toy/Best_word/TimesNewRoman.ttf", 10
        )  # Reduced font size
        # Define the margin
        margin = 1

        # Add the sentences to the image
        text_positions = []
        centers = []
        labels = []
        for text in selected_texts:
            # Calculate the bounding box of the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Ensure the text fits within the image
            if text_width + 2 * margin >= 64 or text_height + 2 * margin >= 64:
                print(f"Warning: Text '{text}' is too large to fit within the image.")
                labels.append(101)
                centers.append((65, 65))
                continue

            # Try to place the text in a random position without overlapping existing text
            max_attempts = 100
            placed = False
            for _ in range(max_attempts):
                position = (
                    random.randint(margin, 64 - text_width - margin),
                    random.randint(margin, 64 - text_height - margin),
                )
                new_bbox = (
                    position[0],
                    position[1],
                    position[0] + text_width,
                    position[1] + text_height,
                )

                # Check for overlap with existing text
                if all(
                    not overlap(new_bbox, existing_bbox)
                    for existing_bbox in text_positions
                ):
                    draw.text(position, text, fill="black", font=font)
                    text_positions.append(new_bbox)
                    # Calculate the center of the text bounding box
                    center_x = (new_bbox[0] + new_bbox[2]) / 2
                    center_y = (new_bbox[1] + new_bbox[3]) / 2
                    centers.append((center_x, center_y))
                    labels.append(text_to_index[text])
                    placed = True
                    break
            if not placed:
                print(f"Warning: Could not place text '{text}' without overlap.")
                labels.append(101)
                centers.append((65, 65))

        # Sort labels and centers by label index
        sorted_indices = np.argsort(labels)
        labels = np.array(labels)[sorted_indices].tolist()
        centers = np.array(centers)[sorted_indices].tolist()

        # Convert image to numpy array in the format [64, 64, 3]
        img_np = np.array(image)
        image.save("/root/generativeencoder/exp/toy/Best_word/data/image.png")
        # Add the image, label, and center information to the dataset
        dataset_images.append(img_np)
        dataset_labels.append(labels)
        dataset_centers.append(centers)

# Convert lists to numpy arrays
dataset_images = np.array(dataset_images)
dataset_labels = np.array(dataset_labels, dtype=np.float32)
dataset_centers = np.array(dataset_centers, dtype=np.float32)

# Save the dataset using PyTorch with pickle protocol 4
with open("/mnt/nfs/xuerongkun/dataset/Words/words_dataset_64x64.pt", "wb") as f:
    torch.save(
        {
            "images": dataset_images,
            "labels": dataset_labels,
            "centers": dataset_centers,
        },
        f,
        pickle_protocol=4,
    )
