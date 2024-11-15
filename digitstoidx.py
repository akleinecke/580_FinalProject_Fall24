import os
import numpy as np
from PIL import Image
import struct

# Function to convert image to numpy array (grayscale or RGB)
def image_to_array(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale (L mode)
    return np.array(image)

#write to idx3 for images and idx1 for labels
def write_idx(images, labels, output_image_file, output_label_file):
    # Write the images in IDX3 format
    with open(output_image_file, 'wb') as f:
        f.write(struct.pack(">I", 2051))  # Magic number for images (2051)
        # Number of images, rows, cols
        f.write(struct.pack(">I", len(images)))  # Number of images
        f.write(struct.pack(">I", 28))  # Rows and columns (28X28 image)
        f.write(struct.pack(">I", 28))  
        for image in images:
            # Write each image as a flat byte array
            f.write(image.tobytes())

    # Write the labels in IDX3 format
    with open(output_label_file, 'wb') as f:
        # Magic number for labels
        f.write(struct.pack(">I", 2049))  # Magic number for labels (2049)
        # Number of labels
        f.write(struct.pack(">I", len(labels)))  # Number of labels
        for label in labels:
            # Write the label as a single byte
            f.write(struct.pack("B", label))

def convert(input_folder, output_image_file, output_label_file):
    images = []
    labels = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            # get label from Group5.Digits.<label char>_##.png
            label = filename[14]
            # opena nd convert to greyscale
            image = Image.open(os.path.join(input_folder, filename)).convert('L') 
            # Add image as array and add the label
            images.append(np.array(image))
            labels.append(int(label))
    #convert to numpy array
    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(len(images), -1)  # Flatten images
    write_idx(images, labels, output_image_file, output_label_file)

# Usage
input_folder = "./image_files/digits/"
output_folder = "./train_data/"
output_image_file = os.path.join(output_folder, "Group5.Digits.Images.idx3-ubyte")
output_label_file = os.path.join(output_folder, "Group5.Digits.Labels.idx1-ubyte")
convert(input_folder, output_image_file, output_label_file)
