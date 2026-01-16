# GUI / App – Zeichenerkennung

## Überblick
Die GUI ermöglicht das Zeichnen handschriftlicher Zeichen und deren Klassifikation durch ein trainiertes Modell.
Zusätzlich kann der Benutzer Feedback geben, welches für eine live aktualisierte Confusion Matrix verwendet wird.

## Start
```bash
python -m app.gui
```
## Funktionen

- Zeichen-Canvas
- Vorhersage mit Confidence
- Feedback (richtig / falsch)
- Unterstützung von 62 Klassen (0–9, A–Z, a–z)
- Confusion Matrix mit Scrollbars

## Module
- gui.py – Tkinter GUI
- preprocessing.py – Bildvorverarbeitung (Resize, Normalize, Transpose)
- inference.py – Modell laden & Vorhersage
- labels.py – Mapping Label ↔ Zeichen
- storage.py – Feedback speichern/laden
- metrics.py – Confusion Matrix Berechnung

## Preprocessing
- Grayscale
- Invert (optional)
- Resize auf 28×28
- Normalisierung [0,1]
- Transpose (EMNIST-Konsistenz)
- Tensorform (1,28,28,1)
