import tkinter as tk
import numpy as np
import os
from tkinter import ttk
from PIL import Image, ImageDraw
from app.preprocessing import pil_to_mnist_tensor
from app.inference import KerasDigitClassifier
from app.metrics import confusion_matrix
from app.storage import append_feedback, load_feedback, DEFAULT_FEEDBACK_FILE, reset_feedback
from app.labels import label_to_char, char_to_label

AVAILABLE_MODELS = [
    ("EMNIST CNN",    "emnist_cnn.keras"),
    ("EMNIST MLP",    "emnist_mlp.keras"),
    ("EMNIST LogReg", "emnist_logreg.keras"),
]




CANVAS_SIZE = 280          # 10x größer als 28, später downsampling
DRAW_RADIUS = 10           # "Pinselgröße"

class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Zeichenerkennung")
        self.model_status = "initializing..."
        self.last_pred = None



        # --- Layout: links Canvas, rechts Controls ---
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=10, padx=12, pady=12)

        controls = ttk.Frame(root)
        controls.grid(row=0, column=1, sticky="n", padx=12, pady=12)

        ttk.Label(controls, text="Modell auswählen:", font=("Arial", 10)).grid(row=0, column=0, sticky="w")
        self.model_choice = tk.StringVar(value=AVAILABLE_MODELS[0][0])

        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_choice,
            values=[name for name, _ in AVAILABLE_MODELS],
            state="readonly",
            width=22
        )
        self.model_combo.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)


        ttk.Button(controls, text="Clear", command=self.clear).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.predict_btn = ttk.Button(controls, text="Predict", command=self.predict)
        self.predict_btn.grid(row=3, column=0, sticky="ew")


        ttk.Separator(controls, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Prediction (probability):", font=("Arial", 11)).grid(row=5, column=0, sticky="w")
        self.pred_text = tk.StringVar(value="-")
        ttk.Label(controls, textvariable=self.pred_text, font=("Arial", 18)).grid(row=6, column=0, sticky="w")

        ttk.Label(controls, text="Model status:").grid(row=7, column=0, sticky="w", pady=(10,0))
        self.model_status_text = tk.StringVar(value=self.model_status)
        ttk.Label(controls, textvariable=self.model_status_text, wraplength=250).grid(row=8, column=0, sticky="w")

        # --- Feedback UI ---
        ttk.Separator(controls, orient="horizontal").grid(row=9, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Feedback:", font=("Arial", 11)).grid(row=10, column=0, sticky="w")

        ttk.Label(controls, text="True Label (0-9, A-Z, a-z):").grid(row=11, column=0, sticky="w", pady=(6, 0))
        self.true_entry = ttk.Entry(controls, width=10)
        self.true_entry.grid(row=12, column=0, sticky="w")

        self.feedback_info = tk.StringVar(value=f"{DEFAULT_FEEDBACK_FILE}")
        ttk.Label(controls, textvariable=self.feedback_info, wraplength=260).grid(row=13, column=0, sticky="w", pady=(6, 0))

        ttk.Button(controls, text="RIGHT", command=self.feedback_correct).grid(row=14, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(controls, text="WRONG", command=self.feedback_wrong).grid(row=15, column=0, sticky="ew")
        ttk.Button(controls, text="Reset Feedback", command=self.reset_feedback_ui).grid(row=16, column=0, sticky="ew", pady=(8, 0))

        ttk.Separator(controls, orient="horizontal").grid(row=17, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Confusion Matrix:").grid(row=18, column=0, sticky="w")

        cm_frame = ttk.Frame(controls)
        cm_frame.grid(row=19, column=0, pady=(6, 0), sticky="nsew")

        xscroll = ttk.Scrollbar(cm_frame, orient="horizontal")
        yscroll = ttk.Scrollbar(cm_frame, orient="vertical")

        self.cm_box = tk.Text(
            cm_frame,
            width=70,
            height=18,
            font=("Courier", 11),
            wrap="none",
            xscrollcommand=xscroll.set,
            yscrollcommand=yscroll.set,
        )
        xscroll.config(command=self.cm_box.xview)
        yscroll.config(command=self.cm_box.yview)

        self.cm_box.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")

        cm_frame.grid_rowconfigure(0, weight=1)
        cm_frame.grid_columnconfigure(0, weight=1)

        self.cm_box.tag_configure("diag", font=("Courier", 11, "bold"), foreground="blue")
        self.cm_box.tag_configure("err", font=("Courier", 11, "bold"), foreground="red")

        self.num_classes = 62
        self.refresh_confusion_matrix()


        # --- Zeichenfläche: PIL Image im Hintergrund (für späteres Preprocessing) ---
        self._init_pil_surface()

        # --- Maus-Events fürs Zeichnen
    
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)

        # --- Model setup ---
        
        self.load_selected_model()
        if self.classifier is None or self.classifier.model is None:
            self.predict_btn.state(["disabled"])
        else:
            self.predict_btn.state(["!disabled"])

    def on_model_change(self, event=None):
        self.last_pred = None
        self.pred_text.set("-")
        self.load_selected_model()

        if self.classifier is None or self.classifier.model is None:
            self.predict_btn.state(["disabled"])
        else:
            self.predict_btn.state(["!disabled"])

    def load_selected_model(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        selected_name = self.model_choice.get()
        filename = None
        for name, fn in AVAILABLE_MODELS:
            if name == selected_name:
                filename = fn
                break

        if filename is None:
            self.model_status = "not loaded: unknown selection"
            self.model_status_text.set(self.model_status)
            self.classifier = None
            return

        model_path = os.path.join(project_root, "models", filename)
        self.classifier = KerasDigitClassifier(model_path)

        try:
            self.classifier.load()
            self.model_status = f"loaded: {filename}"
        except Exception as e:
            self.model_status = f"not loaded: {e}"

        self.model_status_text.set(self.model_status)


    def is_canvas_empty(self) -> bool:
        arr = np.array(self.pil_img)  # Hintergrund weiß=255, Strich schwarz=0
        # Wenn alles weiß ist, wurde nichts gezeichnet
        return arr.min() == 255

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
        self.last_pred = None

    def refresh_confusion_matrix(self):
        pairs = load_feedback(DEFAULT_FEEDBACK_FILE)
        cm = confusion_matrix(pairs, num_classes=self.num_classes)

        self.cm_box.delete("1.0", tk.END)

        # Header: Zeichen anzeigen (0-9, A-Z, a-z)
        self.cm_box.insert(tk.END, "    ")
        for j in range(self.num_classes):
            ch = str(label_to_char(j))
            self.cm_box.insert(tk.END, f"{ch:>4s}")
        self.cm_box.insert(tk.END, "\n")

        self.cm_box.insert(tk.END, "    " + "-" * (4 * self.num_classes) + "\n")

        # Rows
        for i in range(self.num_classes):
            row_ch = str(label_to_char(i))
            self.cm_box.insert(tk.END, f"{row_ch:>2s}: ")

            for j in range(self.num_classes):
                val = int(cm[i, j])
                cell = f"{val:4d}"

                if val == 0:
                    self.cm_box.insert(tk.END, cell)
                else:
                    tag = "diag" if i == j else "err"
                    self.cm_box.insert(tk.END, cell, tag)

            self.cm_box.insert(tk.END, "\n")




    def reset_feedback_ui(self):
        reset_feedback(DEFAULT_FEEDBACK_FILE)
        self.refresh_confusion_matrix()
        self.pred_text.set("Feedback zurückgesetzt")


    def feedback_correct(self):
        if self.last_pred is None:
            self.pred_text.set("Erst Predict ausführen")
            return

        append_feedback(self.last_pred, self.last_pred, DEFAULT_FEEDBACK_FILE)

        self.refresh_confusion_matrix()
        self.true_entry.delete(0, tk.END)


    def feedback_wrong(self):
        if self.last_pred is None:
            self.pred_text.set("Erst Predict ausführen")
            return

        try:
            t = char_to_label(self.true_entry.get())
        except ValueError as e:
            self.pred_text.set(str(e))
            return

        append_feedback(t, self.last_pred, DEFAULT_FEEDBACK_FILE)

        self.refresh_confusion_matrix()
        self.true_entry.delete(0, tk.END)
        self.pred_text.set("Feedback gespeichert")



    def predict(self):
        if self.is_canvas_empty():
            self.pred_text.set("Bitte zuerst zeichnen")
            return

        x = pil_to_mnist_tensor(self.pil_img, invert=True, do_transpose=True)


        # Falls Modell nicht geladen: klarer Hinweis
        if not hasattr(self, "classifier") or self.classifier.model is None:
            self.pred_text.set("Kein Modell geladen")
            return

        pred, conf, _ = self.classifier.predict(x)
        self.pred_text.set(f"{label_to_char(pred)}  (p={conf:.2f})")
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
