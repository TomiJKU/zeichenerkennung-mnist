import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw

CANVAS_SIZE = 280          # 10x größer als 28, später downsampling
DRAW_RADIUS = 10           # "Pinselgröße"

class DigitApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Zeichenerkennung (A2)")

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

        # --- Zeichenfläche: PIL Image im Hintergrund (für späteres Preprocessing) ---
        self._init_pil_surface()

        # --- Maus-Events fürs Zeichnen ---
        self.canvas.bind("<Button-1>", self._start_stroke)
        self.canvas.bind("<B1-Motion>", self._draw_stroke)

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

    def predict(self):
        """
        A2: Dummy-Vorhersage.
        In A4 ersetzen wir das durch echtes Modell-Inferencing.
        """
        # Dummy: zeig einfach eine fixe Ausgabe
        self.pred_text.set("7  (dummy)")

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
