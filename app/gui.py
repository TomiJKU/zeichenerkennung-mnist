import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
from app.preprocessing import pil_to_mnist_tensor
from app.inference import KerasDigitClassifier
from app.storage import append_feedback, load_feedback, DEFAULT_FEEDBACK_FILE
from app.metrics import confusion_matrix, format_confusion_matrix
import os



CANVAS_SIZE = 280          # 10x größer als 28, später downsampling
DRAW_RADIUS = 10           # "Pinselgröße"

class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Zeichenerkennung")
        self.model_status = "initializing..."


        # --- Layout: links Canvas, rechts Controls ---
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=10, padx=12, pady=12)

        controls = ttk.Frame(root)
        controls.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        ttk.Button(controls, text="Clear", command=self.clear).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Predict", command=self.predict).grid(row=1, column=0, sticky="ew")

        ttk.Separator(controls, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Vorhersage:", font=("Arial", 11)).grid(row=3, column=0, sticky="w")
        self.pred_text = tk.StringVar(value="-")
        ttk.Label(controls, textvariable=self.pred_text, font=("Arial", 18)).grid(row=4, column=0, sticky="w")

        ttk.Label(controls, text="Model status:").grid(row=5, column=0, sticky="w", pady=(10,0))
        self.model_status_text = tk.StringVar(value=self.model_status)
        ttk.Label(controls, textvariable=self.model_status_text, wraplength=250).grid(row=6, column=0, sticky="w")

        # --- Feedback UI ---
        ttk.Separator(controls, orient="horizontal").grid(row=7, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Feedback:", font=("Arial", 11)).grid(row=8, column=0, sticky="w")

        ttk.Label(controls, text="True Label (0-9):").grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.true_entry = ttk.Entry(controls, width=10)
        self.true_entry.grid(row=10, column=0, sticky="w")

        self.feedback_info = tk.StringVar(value=f"Speicher: {DEFAULT_FEEDBACK_FILE}")
        ttk.Label(controls, textvariable=self.feedback_info, wraplength=260).grid(row=11, column=0, sticky="w", pady=(6, 0))

        ttk.Button(controls, text="Richtig", command=self.feedback_correct).grid(row=12, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(controls, text="Falsch", command=self.feedback_wrong).grid(row=13, column=0, sticky="ew")

        ttk.Separator(controls, orient="horizontal").grid(row=14, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Confusion Matrix:").grid(row=15, column=0, sticky="w")
        self.cm_box = tk.Text(controls, width=38, height=12, font=("Courier", 10))
        self.cm_box.grid(row=16, column=0, pady=(6, 0))

        self.num_classes = 10
        self.refresh_confusion_matrix()


        # --- Zeichenfläche: PIL Image im Hintergrund (für späteres Preprocessing) ---
        self._init_pil_surface()

        # --- Maus-Events fürs Zeichnen
    
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)

        # --- Model setup ---
        # Modellpfad: ../models/best_model.keras (vom app/ Ordner aus gesehen)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "best_model.keras")

        self.classifier = KerasDigitClassifier(model_path)
        try:
            self.classifier.load()
            self.model_status = "loaded"
        except Exception as e:
            # App soll trotzdem laufen
            self.model_status = f"not loaded: {e}"

        self.model_status_text.set(self.model_status)

    def _init_pil_surface(self):
        # Weißer Hintergrund (L = 8-bit grayscale). Weiß = 255, Schwarz = 0
        self.pil_img = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=255)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

    def _start_stroke(self, event):
        # optional: könnte man später für "letzte Position" nutzen
        self._draw_at(event.x, event.y)

    def _draw_stroke(self, event):
        self._draw_at(event.x, event.y)

    def _draw_at(self, x: int, y: int):
        r = DRAW_RADIUS

        # Auf Tkinter-Canvas zeichnen (sichtbar)
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")

        # Parallel ins PIL-Image zeichnen (für ML)
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=0)  # schwarz

    def clear(self):
        self.canvas.delete("all")
        self._init_pil_surface()
        self.pred_text.set("-")

    def refresh_confusion_matrix(self):
        pairs = load_feedback()
        cm = confusion_matrix(pairs, num_classes=self.num_classes)
        text = format_confusion_matrix(cm)

        self.cm_box.delete("1.0", tk.END)
        self.cm_box.insert(tk.END, text)

    def feedback_correct(self):
        if self.last_pred is None:
            self.pred_text.set("Erst Predict ausführen")
            return

        append_feedback(self.last_pred, self.last_pred)
        self.refresh_confusion_matrix()

    def feedback_wrong(self):
        if self.last_pred is None:
            self.pred_text.set("Erst Predict ausführen")
            return

        t = self.true_entry.get().strip()
        if not t.isdigit():
            self.pred_text.set("True Label eingeben (0-9)")
            return

        t = int(t)
        if not (0 <= t < self.num_classes):
            self.pred_text.set("True Label muss 0-9 sein")
            return

        append_feedback(t, self.last_pred)
        self.refresh_confusion_matrix()


    def predict(self):
        x = pil_to_mnist_tensor(self.pil_img, invert=True)

        # Falls Modell nicht geladen: klarer Hinweis
        if not hasattr(self, "classifier") or self.classifier.model is None:
            self.pred_text.set("Kein Modell geladen")
            return

        pred, conf, _ = self.classifier.predict(x)
        self.pred_text.set(f"{pred}  (p={conf:.2f})")
        self.last_pred = pred



def main():
    root = tk.Tk()
    # ttk theme ist optional, sieht aber meist besser aus
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass

    DigitApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
