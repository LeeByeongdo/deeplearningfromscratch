from PIL import Image
import PIL.ImageOps
import numpy as np


def image_to_bytes(img_path):
    img = Image.open(img_path).convert("L")
    resized = img.resize((28, 28))
    inverted = PIL.ImageOps.invert(resized)
    inverted.save("inverted.png")

    return np.array(inverted).reshape(-1, 784)
