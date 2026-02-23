import numpy as np
from PIL import Image, ImageOps


def pil_to_mnist_tensor(pil_img, invert=True, do_transpose=True):
    # 1) grayscale
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    # 2) invert (white bg -> black fg)
    if invert:
        pil_img = ImageOps.invert(pil_img)

    # 3) resize to 28x28
    pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 4) to numpy / normalize -> (28,28) float32 0..1
    x = np.array(pil_img).astype("float32") / 255.0

    # 5) transpose to match your EMNIST training (you used transpose there)
    if do_transpose:
        x = x.T

    # 6) add batch + channel -> (1,28,28,1)
    x = x[None, ..., None]
    return x
