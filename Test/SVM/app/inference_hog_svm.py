import os
import numpy as np
import joblib


class HOGLinearSVMClassifier:
    """
    Loads a joblib model trained on HOG features.
    We store a CalibratedClassifierCV so predict_proba exists.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def is_loaded(self):
        return self.model is not None

    def predict_from_hog(self, hog_vec: np.ndarray):
        """
        hog_vec: shape (F,) or (1,F)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        x = np.asarray(hog_vec, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]  # (62,)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            return pred, conf, probs

        pred = int(self.model.predict(x)[0])
        return pred, 1.0, None
