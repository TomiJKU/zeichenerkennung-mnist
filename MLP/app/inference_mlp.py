import os
import numpy as np
import tensorflow as tf


class MLPDigitClassifier:
    """
    Loads a Keras MLP model expecting input shape (None, 784)
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)

    def is_loaded(self):
        return self.model is not None

    def predict(self, x: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 784:
            raise ValueError(f"Expected (N,784), got {x.shape}")

        probs = self.model.predict(x, verbose=0)[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
        return pred, conf, probs
