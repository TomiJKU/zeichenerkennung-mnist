import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Test.MLP.app.preprocessing_mlp import pil_to_emnist_mlp_vector
from Test.app.labels import label_to_char

TARGET_CHAR = "3"
TRAIN_TRANSPOSE = True  # match training setting


def char_to_label(ch: str) -> int:
    if "0" <= ch <= "9":
        return ord(ch) - ord("0")
    if "A" <= ch <= "Z":
        return 10 + ord(ch) - ord("A")
    if "a" <= ch <= "z":
        return 36 + ord(ch) - ord("a")
    raise ValueError("Invalid character")


def main():
    target_label = char_to_label(TARGET_CHAR)
    ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

    img_raw = None
    found_label = None
    for x, y in tfds.as_numpy(ds):
        if int(y) == target_label:
            img_raw = np.squeeze(x)  # (28,28)
            found_label = int(y)
            break

    if img_raw is None:
        raise RuntimeError("No matching sample found.")

    pil_img = Image.fromarray(img_raw, mode="L")

    # IMPORTANT: EMNIST sample -> invert=False
    x_vec = pil_to_emnist_mlp_vector(pil_img, invert=False, do_transpose=TRAIN_TRANSPOSE)
    img_out = x_vec.reshape(28, 28)

    fig, axs = plt.subplots(1, 2, figsize=(7, 3))

    axs[0].imshow(img_raw, cmap="gray")
    axs[0].set_title(f"INPUT raw EMNIST\nlabel={found_label} ({label_to_char(found_label)})")
    axs[0].axis("off")

    axs[1].imshow(img_out, cmap="gray")
    axs[1].set_title("OUTPUT after preprocessing_mlp\n(reshaped from 784)")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
