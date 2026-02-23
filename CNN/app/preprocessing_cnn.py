import numpy as np
from PIL import Image, ImageOps


def _center_of_mass_shift(img28: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(img28 > 0.05)
    if len(xs) == 0:
        return img28

    cx = xs.mean()
    cy = ys.mean()
    shift_x = int(round(13.5 - cx))
    shift_y = int(round(13.5 - cy))

    out = np.zeros_like(img28)
    y0 = max(0, shift_y); y1 = min(28, 28 + shift_y)
    x0 = max(0, shift_x); x1 = min(28, 28 + shift_x)

    out[y0:y1, x0:x1] = img28[y0 - shift_y:y1 - shift_y, x0 - shift_x:x1 - shift_x]
    return out


def pil_to_emnist_cnn_tensor(pil_img, invert=True, do_transpose=True):
    # grayscale
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    # invert (white bg -> black bg consistency)
    if invert:
        pil_img = ImageOps.invert(pil_img)

    arr = np.array(pil_img).astype(np.float32) / 255.0  # (H,W)

    # threshold for stable bbox
    arr_bin = (arr > 0.10).astype(np.float32)
    ys, xs = np.nonzero(arr_bin)

    if len(xs) == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # margin
    margin = 10
    x0 = max(0, x0 - margin); x1 = min(arr.shape[1] - 1, x1 + margin)
    y0 = max(0, y0 - margin); y1 = min(arr.shape[0] - 1, y1 + margin)

    cropped = arr[y0:y1 + 1, x0:x1 + 1]
    crop_img = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")

    # resize content to fit 20x20 keeping aspect ratio
    w, h = crop_img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))

    crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # paste into 28x28 canvas
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(crop_img, (left, top))

    x28 = np.array(canvas).astype(np.float32) / 255.0
    x28 = _center_of_mass_shift(x28)

    if do_transpose:
        x28 = x28.T

    return x28[None, ..., None].astype(np.float32)
