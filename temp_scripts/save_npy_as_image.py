import os
import numpy as np
from PIL import Image

def save_as_jpeg(npy_file, jpeg_file):
    # Load the .npy file
    data = np.load(npy_file)

    # Normalize data if necessary to fit into uint8 range (0-255)
    if data.dtype != np.uint8:
        data = (data - data.min()) / (data.max() - data.min()) * 255
        data = data.astype(np.uint8)

    # Convert numpy array to Pillow image
    img = Image.fromarray(data)

    # Save as JPEG
    img.save(jpeg_file)

# Example usage

folder = "freehand"
# modes = ["gt", "pred"]
modes = ["pred"]
for mode in modes:
    npy_file = os.path.join(folder, "files/0000000019_{}.npy".format(mode))
    jpeg_file = "0000000019_{}.jpeg".format(mode)
    save_as_jpeg(npy_file, os.path.join(folder, jpeg_file))