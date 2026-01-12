from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

def pil_to_mnist_tensor(
    pil_img: Image.Image,
    *,
    invert: bool = True,
    center_crop: bool = True,
    out_size: int = 28,
) -> np.ndarray:
    """
    Konvertiert eine PIL-Graustufenfläche (z.B. 280x280, Hintergrund weiß=255, Strich schwarz=0)
    in ein MNIST-kompatibles Tensor-Format: (1, out_size, out_size, 1), float32, Werte 0..1.

    invert=True:
        Invertiert (weiß->schwarz, schwarz->weiß), damit der Strich "hell" ist wie bei MNIST.
        (MNIST hat typischerweise helle Ziffer auf dunklem Hintergrund)
    center_crop=True:
        Schneidet den Bereich um den gezeichneten Inhalt aus und zentriert ihn.
        (hilft stark bei GUI-Eingaben)
    """
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    img = pil_img

    # 1) Optional invertieren (GUI: schwarzer Strich auf weißem Hintergrund)
    if invert:
        img = ImageOps.invert(img)

    # 2) Optional: Content finden und crop + in die Mitte setzen
    if center_crop:
        # Wir suchen die "nicht-schwarzen" Pixel nach Invertierung:
        # nach invert ist Hintergrund meist 0 (schwarz) und Strich >0 (hell)
        arr = np.array(img)
        ys, xs = np.where(arr > 0)

        if len(xs) > 0 and len(ys) > 0:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()

            # kleiner Rand um das Bounding-Box-Fenster
            pad = 10
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(arr.shape[1] - 1, x1 + pad)
            y1 = min(arr.shape[0] - 1, y1 + pad)

            cropped = img.crop((x0, y0, x1 + 1, y1 + 1))

            # Auf quadratisch bringen (damit nicht verzerrt wird)
            w, h = cropped.size
            side = max(w, h)
            square = Image.new("L", (side, side), color=0)  # Hintergrund schwarz (nach invert)
            square.paste(cropped, ((side - w) // 2, (side - h) // 2))

            img = square
        # wenn nichts gezeichnet: img bleibt wie es ist

    # 3) Resize auf 28x28
    img_small = img.resize((out_size, out_size), Image.Resampling.LANCZOS)

    # 4) Normalisieren und Shape anpassen
    x = np.array(img_small).astype("float32") / 255.0      # (28,28)
    x = x[None, ..., None]                                  # (1,28,28,1)
    return x


def debug_to_uint8_image(x: np.ndarray) -> Image.Image:
    """
    Hilfsfunktion: Tensor (1,28,28,1) zurück zu PIL-Bild (0..255),
    um das Preprocessing visuell zu prüfen.
    """
    if x.ndim != 4:
        raise ValueError("Expected 4D tensor (1,H,W,1)")
    arr = (np.clip(x[0, :, :, 0], 0, 1) * 255.0).astype("uint8")
    return Image.fromarray(arr, mode="L")
