import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Define the text to be added
texts = [
    "Hello, world!", "What's up?", "How are you?", "Goodbye!", "See you later!",
    "Good morning!", "Good night!", "Have a nice day!", "Happy birthday!", "Congratulations!",
    "Welcome!", "Thank you!", "Sorry!", "Excuse me!", "No problem!",
    "Yes, please!", "No, thanks!", "Help!", "Stop!", "Go!",
    "Start!", "End!", "Open!", "Close!", "Run!",
    "Walk!", "Jump!", "Sit!", "Stand!", "Look!",
    "Listen!", "Speak!", "Read!", "Write!", "Think!",
    "Dream!", "Smile!", "Laugh!", "Cry!", "Sing!",
    "Dance!", "Play!", "Work!", "Rest!", "Sleep!",
    "Wake up!", "Eat!", "Drink!", "Cook!", "Clean!",
    "Wash!", "Dry!", "Iron!", "Fold!", "Pack!",
    "Unpack!", "Buy!", "Sell!", "Pay!", "Borrow!",
    "Lend!", "Find!", "Lose!", "Catch!", "Throw!",
    "Push!", "Pull!", "Hold!", "Drop!", "Give!",
    "Take!", "Send!", "Receive!", "Open the door!", "Close the window!",
    "Turn on the light!", "Turn off the light!", "Lock the door!", "Unlock the door!", "Turn up the volume!",
    "Turn down the volume!", "Increase speed!", "Decrease speed!", "Start the engine!", "Stop the engine!",
    "Charge the battery!", "Change the channel!", "Adjust the seat!", "Fasten your seatbelt!", "Unfasten your seatbelt!",
    "Put on your shoes!", "Take off your shoes!", "Wear a hat!", "Remove your hat!", "Save the file!",
    "Delete the file!", "Copy the text!", "Paste the text!", "Print the document!", "Scan the document!"
]

# Create a mapping from text to index for one-hot encoding
text_to_index = {text: i for i, text in enumerate(texts)}

# Function to generate a random color
def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))

# Function to encode sentences as one-hot vectors
def encode_sentences(selected_texts):
    one_hot_vectors = np.zeros(len(texts))
    for text in selected_texts:
        one_hot_vectors[text_to_index[text]] = 1
    return one_hot_vectors

# Create a list to store the dataset information
dataset_info = []

# Loop to create multiple images
for i in range(100):
    # Randomly select the number of sentences (1 to 4)
    num_sentences = random.randint(1, 4)
    # Randomly select the sentences without repetition
    selected_texts = random.sample(texts, num_sentences)
    # Generate a random background color
    color = random_color()
    
    # Create a new 100x100 image with the selected background color
    image = Image.new('RGB', (100, 100), color=color)
    
    # Initialize the drawing context
    draw = ImageDraw.Draw(image)
    
    try:
        # Use a truetype font if you have one available
        font = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        # Fallback to default PIL font if truetype font is not available
        font = ImageFont.load_default()
    
    # Add the sentences to the image
    y_offset = 0
    text_positions = []
    for text in selected_texts:
        # Calculate the bounding box of the text
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Ensure that the text fits within the image
        while text_width > 100:
            # If the text width exceeds the image width, choose another position
            text = text[:max(1, len(text) - 1)]
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
        
        # Calculate the maximum y position to avoid overflow
        max_y = 100 - y_offset - text_height
        
        if max_y < 0:
            raise ValueError("Text height exceeds available space in the image")
        
        # Randomize the position at which to draw the text within the remaining space
        position = (random.randint(0, 100 - text_width), y_offset)
        
        # Draw the text on the image
        draw.text(position, text, fill="black", font=font)
        
        # Update y_offset to avoid overlap
        y_offset += text_height + 5  # Add some padding between lines
        
        # Store text position
        text_positions.append(position)
    
    # Save the image as a PNG file
    image_filename = f"text_image_{i}.png"
    image_path='./figs/' + image_filename
    if not os.path.exists('./figs/'):
        os.makedirs('./figs/')
    image.save(image_path)
    
    # Encode the selected sentences as one-hot vectors
    one_hot_vectors = encode_sentences(selected_texts)
    
    # Add the image information to the dataset
    dataset_info.append({
        "filename": image_filename,
        "texts": selected_texts,
        "one_hot_vectors": one_hot_vectors.tolist(),  # Convert to list for serialization
        "color": color,
        "positions": text_positions
    })

# Save the dataset information using PyTorch
torch.save(dataset_info, './dataset.pth')

