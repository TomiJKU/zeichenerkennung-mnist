from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
import joblib


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

    def is_loaded(self) -> bool:
        return self.model is not None

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
            raise ValueError(
                f"Expected input shape (1,{expected[1]},{expected[2]},{expected[3]}), got {x.shape}"
            )

        probs = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        return pred, conf, probs


class SklearnPipelineClassifier:
    """
    Lädt ein scikit-learn Modell/Pipeline aus .joblib und macht Vorhersagen.
    Erwartet Input wie Keras: (1,28,28,1) float32 0..1,
    wird intern geflattet zu (1,784).

    Typisch: Pipeline(PCA -> RandomForest).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipeline = None

    def load(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def predict(self, x: np.ndarray) -> tuple[int, float, np.ndarray | None]:
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        x = np.asarray(x, dtype=np.float32)

        # Erwartet (1,28,28,1) -> flatten zu (1,784)
        if x.ndim == 4:
            x_flat = x.reshape((x.shape[0], -1))
        elif x.ndim == 2:
            x_flat = x.reshape(1, -1)
        else:
            x_flat = x.reshape(1, -1)

        # Klassenreihenfolge in sklearn kann theoretisch abweichen -> absichern
        classes = getattr(self.pipeline, "classes_", None)

        if hasattr(self.pipeline, "predict_proba"):
            probs_raw = self.pipeline.predict_proba(x_flat)[0]

            if classes is None:
                # Fallback: assume classes are 0..N-1
                probs = probs_raw
            else:
                # Map auf 0..61 (62 Klassen). Falls alle Klassen vorhanden: identity.
                probs = np.zeros(62, dtype=np.float32)
                for idx, cls in enumerate(classes):
                    cls_int = int(cls)
                    if 0 <= cls_int < 62:
                        probs[cls_int] = probs_raw[idx]

            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            return pred, conf, probs
        else:
            pred = int(self.pipeline.predict(x_flat)[0])
            return pred, 1.0, None
