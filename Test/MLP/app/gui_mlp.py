import os
import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageDraw

from .preprocessing_mlp import pil_to_emnist_mlp_vector
from .inference_mlp import MLPDigitClassifier

from Test.app.labels import label_to_char


CANVAS_SIZE = 280
DRAW_RADIUS = 10


class MLPTestApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MLP Test – EMNIST")

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, padx=12, pady=12)

        btns = ttk.Frame(root)
        btns.grid(row=1, column=0, sticky="ew", padx=12)

        ttk.Button(btns, text="Predict", command=self.predict).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Clear", command=self.clear).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        self.pred_text = tk.StringVar(value="-")
        ttk.Label(root, textvariable=self.pred_text, font=("Arial", 18)).grid(
            row=2, column=0, padx=12, pady=(8, 12), sticky="w"
        )

        self._init_pil_surface()
        self.canvas.bind("<Button-1>", self._draw)
        self.canvas.bind("<B1-Motion>", self._draw)

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        self.model_path = os.path.join(project_root, "Test", "MLP", "models", "emnist_mlp_best.keras")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.classifier = MLPDigitClassifier(self.model_path)
        try:
            self.classifier.load()
            self.pred_text.set("Model loaded ✓  – draw and Predict")
        except Exception as e:
            self.pred_text.set(f"Model load failed: {e}")

    def _init_pil_surface(self):
        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

    def _draw(self, e):
        r = DRAW_RADIUS
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="black", outline="black")
        self.pil_draw.ellipse((e.x - r, e.y - r, e.x + r, e.y + r), fill=0)

    def clear(self):
        self.canvas.delete("all")
        self._init_pil_surface()
        self.pred_text.set("-")

    def is_canvas_empty(self):
        return np.array(self.pil_img).min() == 255

    def predict(self):
        if self.is_canvas_empty():
            self.pred_text.set("Bitte zuerst zeichnen")
            return
        if not self.classifier.is_loaded():
            self.pred_text.set("Kein Modell geladen")
            return

        # Canvas drawings: invert=True, do_transpose=False (wie bei dir funktionierend)
        x = pil_to_emnist_mlp_vector(self.pil_img, invert=True, do_transpose=False)

        pred, conf, _ = self.classifier.predict(x)
        self.pred_text.set(f"{label_to_char(pred)}  (p={conf:.2f})")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    MLPTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
