# Zeichenerkennung – EMNIST ByClass (62 Klassen) mit GUI + Multi-Model (CNN / MLP / SVM)

Dieses Projekt implementiert eine GUI zur Erkennung handschriftlicher Zeichen (EMNIST ByClass: **0–9, A–Z, a–z** = **62 Klassen**) und lädt mehrere Klassifikatoren (CNN, MLP, Linear SVM/HOG). Optional kann ein Ensemble (Soft Voting) verwendet werden.

## Features

- **GUI (Tkinter)**: Zeichnen am Canvas, Vorhersage per Klick
- **Mehrere Modelle**:
  - CNN (Keras)
  - MLP (Keras)
  - Linear SVM auf HOG-Features (joblib)
  - Optional: **Ensemble** (Soft Voting, z. B. CNN + SVM)
- **Top-k Predictions** (z. B. Top-3) inkl. Wahrscheinlichkeiten/Confidence
- Strukturierte Trennung von:
  - Preprocessing
  - Inference/Model-Loading
  - GUI

## Projektstruktur (Zielzustand nach Umstellung)

> Hinweis: In der aktuellen Historie existierte zeitweise ein Ordner `Test/`. Dieser wird/ wurde in die Projektwurzel „hochgezogen“, damit Imports und Startkommandos wieder konsistent sind. :contentReference[oaicite:2]{index=2}

finaler Root:
zeichenerkennung-mnist/
├─ app/

│ ├─ gui_complete.py # GUI Einstiegspunkt (python -m app.gui_complete)

│ ├─ labels.py # Label↔Char Mapping

│ └─ ...

├─ CNN/

│ ├─ app/ # preprocessing_cnn.py, inference_cnn.py, ...

│ └─ models/ # *.keras

├─ MLP/
│ ├─ app/ # preprocessing_mlp.py, inference_mlp.py, ...

│ └─ models/ # *.keras

├─ SVM/

│ ├─ app/ # preprocessing_hog_svm.py, inference_hog_svm.py, ...

│ └─ models/ # *.joblib

├─ training/ # Trainingsskripte (optional je nach Stand)

├─ report/ # Dokumentation/Abgabe

├─ requirements.txt

├─ README.md

└─deprecated_old_root/ # Legacy-Code, nicht mehr aktiv verwendet



## Setup

### Repository klonen

```bash
git clone https://github.com/TomiJKU/zeichenerkennung-mnist.git
```
```bash
cd zeichenerkennung-mnist
```

### Virtuelle Umgebung
```bash
python -m venv .venv
```
### Aktivieren:
- Windows (PowerShell):
```bash
.\.venv\Scripts\Activate.ps1
```
-macOS/Linux:
```bash
source .venv/bin/activate
```
### Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```
### GUI starten
Aus dem Repository-Root:
```bash
python -m app.gui_complete
```
### Modelle
Die GUI lädt die Modelle aus den jeweiligen Modellordnern:
- CNN/models/*.keras
- MLP/models/*.keras
- SVM/models/*.joblib
