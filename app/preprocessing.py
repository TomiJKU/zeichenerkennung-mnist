import numpy as np
from PIL import Image, ImageOps

def pil_to_mnist_tensor(pil_img, invert=True, do_transpose=True):
    # 1) grayscale
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    # 2) invert (white background -> black background consistency)
    if invert:
        pil_img = ImageOps.invert(pil_img)

    # 3) resize to 28x28 (THIS fixes your 119x119)
    pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 4) to numpy / normalize
    x = np.array(pil_img).astype("float32") / 255.0  # (28,28)

    # 5) transpose to match EMNIST training (if enabled there)
    if do_transpose:
        x = x.T

    # 6) add batch + channel -> (1,28,28,1)
    x = x[None, ..., None]
    return x
