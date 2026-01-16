from __future__ import annotations

import os
import numpy as np
import tensorflow as tf


class KerasDigitClassifier:
    """
    Lädt ein Keras-Modell (SavedModel/.keras/.h5) und macht Vorhersagen.
    Erwartet Input-Shape (1, 28, 28, 1), float32, Werte 0..1.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, x: np.ndarray) -> tuple[int, float, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (1,H,W,C), got {x.shape}")

        expected = self.model.input_shape  # (None,H,W,C)
        if expected is None or len(expected) != 4:
            raise ValueError(f"Unexpected model input_shape: {expected}")

        if x.shape[1:] != expected[1:]:
            raise ValueError(f"Expected input shape (1,{expected[1]},{expected[2]},{expected[3]}), got {x.shape}")

        probs = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        return pred, conf, probs

