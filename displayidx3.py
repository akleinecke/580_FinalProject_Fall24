import struct
import random
import numpy as np
import matplotlib.pyplot as plt

# Function to read the MNIST images from file
def read_images(file_path):
    with open(file_path, 'rb') as f:
        # Read magic number and number of images
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
    return images

# Function to read the MNIST labels from file
def read_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        
        # Read the labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the training images and labels
images = read_images('./train_data/Group5.Digits.Images.idx3-ubyte')
labels = read_labels('./train_data/Group5.Digits.Labels.idx1-ubyte')

num_images = len(images)
num_cols = 10  # You can change this to adjust the number of columns in the grid
num_rows = (num_images // num_cols) + 1  # Calculate the number of rows needed

# Create the figure with subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 1.1))
axes = axes.flatten()  # Flatten the axes array for easy iteration

# Loop through each image and display it
for i in range(num_images):
    ax = axes[i]  # Get the subplot axis
    image = images[i].reshape(28, 28)  # Reshape image to 28x28 pixels
    ax.imshow(image, cmap='gray')  # Display the image in grayscale
    ax.set_title(f"Label: {labels[i]}")  # Set the label as title
    ax.axis('off')  # Turn off axis labels

# Turn off axes for remaining empty subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Adjust layout to make it neat
plt.tight_layout()
plt.show()
