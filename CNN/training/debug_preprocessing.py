import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Test.CNN.app.preprocessing_cnn import pil_to_emnist_cnn_tensor
from Test.app.labels import label_to_char


# ======= Wähle hier ein Zeichen =======
TARGET_CHAR = "3"    # z.B. "3", "E", "M", "w"
DO_TRANSPOSE = True  # muss zu deinem Training passen


def char_to_label(ch: str) -> int:
    if "0" <= ch <= "9":
        return ord(ch) - ord("0")
    if "A" <= ch <= "Z":
        return 10 + (ord(ch) - ord("A"))
    if "a" <= ch <= "z":
        return 36 + (ord(ch) - ord("a"))
    raise ValueError("Invalid character (use 0-9, A-Z, a-z)")


def main():
    target_label = char_to_label(TARGET_CHAR)

    print("=" * 60)
    print("DEBUG EMNIST PREPROCESSING")
    print(f"Requested char : '{TARGET_CHAR}'")
    print(f"Target label   : {target_label}")
    print("=" * 60)

    ds = tfds.load("emnist/byclass", split="train", as_supervised=True)

    # find first sample with the label
    img_raw = None
    found_label = None

    for x, y in tfds.as_numpy(ds):
        if int(y) == target_label:
            img_raw = np.squeeze(x)  # (28,28) uint8
            found_label = int(y)
            break

    if img_raw is None:
        raise RuntimeError("No matching sample found.")

    print(f"Found label    : {found_label}")
    print(f"Label->char    : {label_to_char(found_label)}")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))

    # 1) Original EMNIST
    axs[0].imshow(img_raw, cmap="gray")
    axs[0].set_title(f"Original EMNIST\nlabel={found_label} ({label_to_char(found_label)})")
    axs[0].axis("off")

    # 2) After transpose (training-side)
    img_norm = img_raw.astype("float32") / 255.0
    if DO_TRANSPOSE:
        img_norm = img_norm.T

    axs[1].imshow(img_norm, cmap="gray")
    axs[1].set_title(f"After transpose\nlabel={found_label} ({label_to_char(found_label)})")
    axs[1].axis("off")

    # 3) After preprocessing_cnn (IMPORTANT: invert=False for EMNIST samples)
    pil_img = Image.fromarray(img_raw, mode="L")
    x_gui = pil_to_emnist_cnn_tensor(
        pil_img,
        invert=False,             # <-- IMPORTANT for EMNIST samples
        do_transpose=DO_TRANSPOSE
    )
    img_gui = np.squeeze(x_gui)

    axs[2].imshow(img_gui, cmap="gray")
    axs[2].set_title(f"After preprocessing\nlabel={found_label} ({label_to_char(found_label)})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
