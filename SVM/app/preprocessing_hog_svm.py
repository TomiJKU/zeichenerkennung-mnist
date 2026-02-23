import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog


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


def _to_28x28_canvas(arr01: np.ndarray) -> np.ndarray:
    """
    Input: arr01 float32 0..1 arbitrary size
    Output: 28x28 float32 0..1, crop/center/pad + COM shift
    """
    arr_bin = (arr01 > 0.10).astype(np.float32)
    ys, xs = np.nonzero(arr_bin)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    margin = 10
    x0 = max(0, x0 - margin); x1 = min(arr01.shape[1] - 1, x1 + margin)
    y0 = max(0, y0 - margin); y1 = min(arr01.shape[0] - 1, y1 + margin)

    cropped = arr01[y0:y1 + 1, x0:x1 + 1]
    crop_img = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")

    w, h = crop_img.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))
    else:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))

    crop_img = crop_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - new_w) // 2
    top = (28 - new_h) // 2
    canvas.paste(crop_img, (left, top))

    x28 = np.array(canvas).astype(np.float32) / 255.0
    x28 = _center_of_mass_shift(x28)
    return x28


def hog_features_from_pil(
    pil_img,
    invert: bool = True,
    do_transpose: bool = False,
    orientations: int = 9,
    pixels_per_cell=(4, 4),
    cells_per_block=(2, 2),
):
    """
    Returns HOG feature vector (float32) for a PIL input (e.g., canvas drawing).

    - invert=True for GUI drawings (black on white)
    - do_transpose=False for GUI drawings (you draw already upright)
    """
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    if invert:
        pil_img = ImageOps.invert(pil_img)

    arr01 = (np.array(pil_img).astype(np.float32) / 255.0)
    x28 = _to_28x28_canvas(arr01)

    if do_transpose:
        x28 = x28.T

    feat = hog(
        x28,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)

    return feat
