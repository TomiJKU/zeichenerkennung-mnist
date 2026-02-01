import os
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont
import numpy as np
from PIL import Image, ImageDraw

# --- CNN ---
from Test.CNN.app.preprocessing_cnn import pil_to_emnist_cnn_tensor
from Test.CNN.app.inference_cnn import CNNDigitClassifier

# --- MLP ---
from Test.MLP.app.preprocessing_mlp import pil_to_emnist_mlp_vector
from Test.MLP.app.inference_mlp import MLPDigitClassifier

# --- SVM (HOG) ---
from Test.SVM.app.preprocessing_hog_svm import hog_features_from_pil
from Test.SVM.app.inference_hog_svm import HOGLinearSVMClassifier

# --- Label mapping (du hast es unter Test/app/labels.py) ---
from Test.app.labels import label_to_char


CANVAS_SIZE = 280
DRAW_RADIUS = 10
NUM_CLASSES = 62

# Soft-voting weights
W_CNN = 0.60
W_SVM = 0.40


def repo_root_from_this_file(file_path: str) -> str:
    """
    file_path = .../zeichenerkennung-mnist/Test/app/gui_complete.py
    repo_root = .../zeichenerkennung-mnist
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(file_path))))


def topk_from_probs(probs: np.ndarray, k: int = 3):
    probs = np.asarray(probs, dtype=np.float32).reshape(-1)
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


class ModelBlock:
    """
    Card-style block with:
    - title
    - status line
    - top-3 predictions
    - horizontal probability bars (top1 green, others red)
    """
    def __init__(self, parent, title: str, font_normal, font_bold):
        self.font_normal = font_normal
        self.font_bold = font_bold

        # outer card
        self.frame = tk.Frame(
            parent,
            bg="white",
            highlightbackground="#c9c9c9",
            highlightthickness=1,
            bd=0,
        )

        # inner padding
        inner = tk.Frame(self.frame, bg="white")
        inner.grid(row=0, column=0, sticky="nsew", padx=10, pady=8)
        self.frame.grid_columnconfigure(0, weight=1)
        inner.grid_columnconfigure(1, weight=1)

        # title
        self.title_label = tk.Label(inner, text=title, bg="white", fg="#111111", font=font_bold)
        self.title_label.grid(row=0, column=0, columnspan=2, sticky="w")

        # separator
        tk.Frame(inner, bg="#e6e6e6", height=1).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 8))

        # status
        self.status_var = tk.StringVar(value="initializing...")
        self.status = tk.Label(
            inner,
            textvariable=self.status_var,
            bg="white",
            fg="#444444",
            font=font_normal,
            justify="left",
            wraplength=360
        )
        self.status.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        # rows for top3: left label, right bar
        self.line_vars = [tk.StringVar(value="-") for _ in range(3)]
        self.lines = []
        self.bars = []

        # bar settings
        self.bar_width = 220   # px (adjust if you want)
        self.bar_height = 12   # px

        for i in range(3):
            # prediction text
            lbl_font = font_bold if i == 0 else font_normal
            lbl = tk.Label(inner, textvariable=self.line_vars[i], bg="white", fg="#111111", font=lbl_font)
            lbl.grid(row=3 + i, column=0, sticky="w", pady=2)

            # bar canvas
            c = tk.Canvas(inner, width=self.bar_width, height=self.bar_height,
                          bg="white", highlightthickness=0)
            c.grid(row=3 + i, column=1, sticky="e", pady=2)

            # draw background bar (light gray)
            c.create_rectangle(0, 0, self.bar_width, self.bar_height, fill="#eeeeee", outline="#dddddd")
            # draw foreground bar (colored) — we update its coords
            color = "#2ecc71" if i == 0 else "#e74c3c"  # green / red
            bar_id = c.create_rectangle(0, 0, 0, self.bar_height, fill=color, outline=color)

            self.lines.append(lbl)
            self.bars.append((c, bar_id, color))

    def grid(self, **kwargs):
        self.frame.grid(**kwargs)

    def set_status(self, text: str):
        self.status_var.set(text)

    def _set_bar(self, idx: int, prob: float):
        prob = float(prob)
        prob = 0.0 if prob < 0 else 1.0 if prob > 1 else prob

        c, bar_id, color = self.bars[idx]
        w = int(self.bar_width * prob)

        # update bar length
        c.coords(bar_id, 0, 0, w, self.bar_height)

        # keep correct color (top1 green, others red)
        c.itemconfig(bar_id, fill=color, outline=color)

    def _clear_bar(self, idx: int):
        c, bar_id, color = self.bars[idx]
        c.coords(bar_id, 0, 0, 0, self.bar_height)

    def set_top3(self, items):
        """
        items: list of tuples (label_int, prob_float)
        """
        if not items:
            for i in range(3):
                self.line_vars[i].set("-")
                self._clear_bar(i)
            return

        for i in range(3):
            if i < len(items):
                lab, p = items[i]
                ch = label_to_char(int(lab))
                self.line_vars[i].set(f"{ch}   (p={float(p):.2f})")
                self._set_bar(i, float(p))
            else:
                self.line_vars[i].set("-")
                self._clear_bar(i)


class AllModelsTop3GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Zeichenerkennung – CNN / MLP / SVM / Ensemble (Top-3)")

        self.font_normal = tkfont.Font(family="Arial", size=12)
        self.font_bold = tkfont.Font(family="Arial", size=12, weight="bold")

        # Left canvas
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=20, padx=12, pady=12)

        # Right panel
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        # Buttons
        btns = ttk.Frame(right)
        btns.grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Predict (alle)", command=self.predict_all).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Clear", command=self.clear).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)

        ttk.Separator(right, orient="horizontal").grid(row=1, column=0, sticky="ew", pady=10)

        # Model blocks
        self.block_cnn = ModelBlock(right, "CNN (Top-3)", self.font_normal, self.font_bold)
        self.block_mlp = ModelBlock(right, "MLP (Top-3)", self.font_normal, self.font_bold)
        self.block_svm = ModelBlock(right, "Linear SVM (Top-3)", self.font_normal, self.font_bold)
        self.block_ens = ModelBlock(
            right,
            f"Soft Voting (CNN {W_CNN:.2f} + SVM {W_SVM:.2f}) (Top-3)",
            self.font_normal,
            self.font_bold
        )

        right.grid_columnconfigure(0, weight=1)

        self.block_cnn.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.block_mlp.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.block_svm.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.block_ens.grid(row=5, column=0, sticky="ew")

        # Drawing backend
        self._init_pil_surface()
        self.canvas.bind("<Button-1>", self._draw)
        self.canvas.bind("<B1-Motion>", self._draw)

        # Load models
        self.cnn = None
        self.mlp = None
        self.svm = None
        self._load_models()

    def _load_models(self):
        repo_root = repo_root_from_this_file(__file__)

        # --- paths inside Test/ ---
        cnn_best = os.path.join(repo_root, "Test", "CNN", "models", "emnist_cnn_best.keras")
        cnn_fallback = os.path.join(repo_root, "Test", "CNN", "models", "emnist_cnn_aug.keras")
        cnn_path = cnn_best if os.path.exists(cnn_best) else cnn_fallback

        mlp_best = os.path.join(repo_root, "Test", "MLP", "models", "emnist_mlp_best.keras")
        mlp_fallback = os.path.join(repo_root, "Test", "MLP", "models", "emnist_mlp.keras")
        mlp_path = mlp_best if os.path.exists(mlp_best) else mlp_fallback

        svm_path = os.path.join(repo_root, "Test", "SVM", "models", "emnist_hog_linear_svm.joblib")

        # Load CNN
        try:
            self.cnn = CNNDigitClassifier(cnn_path)
            self.cnn.load()
            self.block_cnn.set_status(f"loaded: {os.path.relpath(cnn_path, repo_root)}")
        except Exception as e:
            self.cnn = None
            self.block_cnn.set_status(f"not loaded: {e}")

        # Load MLP
        try:
            self.mlp = MLPDigitClassifier(mlp_path)
            self.mlp.load()
            self.block_mlp.set_status(f"loaded: {os.path.relpath(mlp_path, repo_root)}")
        except Exception as e:
            self.mlp = None
            self.block_mlp.set_status(f"not loaded: {e}")

        # Load SVM
        try:
            self.svm = HOGLinearSVMClassifier(svm_path)
            self.svm.load()
            self.block_svm.set_status(f"loaded: {os.path.relpath(svm_path, repo_root)}")
        except Exception as e:
            self.svm = None
            self.block_svm.set_status(f"not loaded: {e}")

        # Ensemble
        if self.cnn is not None and self.svm is not None:
            self.block_ens.set_status("ready (uses CNN + SVM)")
        else:
            self.block_ens.set_status("not ready (needs CNN + SVM)")

    # -------------------------
    # Drawing
    # -------------------------
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
        self.block_cnn.set_top3(None)
        self.block_mlp.set_top3(None)
        self.block_svm.set_top3(None)
        self.block_ens.set_top3(None)

    def is_canvas_empty(self):
        return np.array(self.pil_img).min() == 255

    # -------------------------
    # Prediction (keeps per-model preprocessing)
    # -------------------------
    def predict_all(self):
        if self.is_canvas_empty():
            self.block_cnn.set_top3(None)
            self.block_mlp.set_top3(None)
            self.block_svm.set_top3(None)
            self.block_ens.set_top3(None)
            return

        cnn_probs = None
        svm_probs = None

        # CNN
        if self.cnn is not None and self.cnn.is_loaded():
            x_cnn = pil_to_emnist_cnn_tensor(self.pil_img, invert=True, do_transpose=False)
            _, _, probs = self.cnn.predict(x_cnn)
            cnn_probs = np.asarray(probs, dtype=np.float32)
            self.block_cnn.set_top3(topk_from_probs(cnn_probs, 3))
        else:
            self.block_cnn.set_top3(None)

        # MLP
        if self.mlp is not None and self.mlp.is_loaded():
            x_mlp = pil_to_emnist_mlp_vector(self.pil_img, invert=True, do_transpose=False)
            _, _, probs = self.mlp.predict(x_mlp)
            mlp_probs = np.asarray(probs, dtype=np.float32)
            self.block_mlp.set_top3(topk_from_probs(mlp_probs, 3))
        else:
            self.block_mlp.set_top3(None)

        # SVM
        if self.svm is not None and self.svm.is_loaded():
            feat = hog_features_from_pil(self.pil_img, invert=True, do_transpose=False)
            pred, conf, probs = self.svm.predict_from_hog(feat)
            if probs is not None:
                svm_probs = np.asarray(probs, dtype=np.float32)
                self.block_svm.set_top3(topk_from_probs(svm_probs, 3))
            else:
                # fallback (no proba)
                self.block_svm.set_top3([(pred, conf)])
        else:
            self.block_svm.set_top3(None)

        # Ensemble (CNN + SVM soft voting)
        if cnn_probs is not None and svm_probs is not None:
            p_final = W_CNN * cnn_probs + W_SVM * svm_probs
            self.block_ens.set_top3(topk_from_probs(p_final, 3))
        else:
            self.block_ens.set_top3(None)


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    AllModelsTop3GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
