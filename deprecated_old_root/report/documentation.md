# Projektdokumentation

---

## 📄 `report/documentation.md`

```md
# Projektbericht – Zeichenerkennung mit MNIST & EMNIST

## 1. Zielsetzung
Ziel des Projekts ist die Entwicklung eines Modells zur Erkennung handschriftlicher Zeichen sowie dessen Integration in eine grafische Applikation.

Die Anwendung erlaubt:
- Zeichnen eines Zeichens
- Modellbasierte Vorhersage
- Benutzerfeedback
- Live Confusion Matrix

## 2. Datensätze
### MNIST
28×28 Graustufenbilder handschriftlicher Ziffern (0–9).

### EMNIST ByClass
Erweiterung auf 62 Klassen (0–9, A–Z, a–z).

## 3. Preprocessing
### Training
- Normalisierung
- Resize auf 28×28
- Transpose

### GUI
- Canvas → Bild
- Grayscale
- Invert
- Resize
- Normalisierung
- Transpose
- `(1,28,28,1)`

## 4. Modell
- CNN
- Input: `(1,28,28,1)`
- Output: Softmax (62 Klassen)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy

## 5. Evaluation
- Validation Accuracy ~85–87 %
- Confusion Matrix
- Typische Fehler bei ähnlichen Zeichen

## 6. Deployment
- Tkinter GUI
- Feedback richtig/falsch
- Speicherung in CSV
- Live Confusion Matrix

## 7. Teamarbeit
- Thomas: GUI, Inferenz, Feedback
- Florian: Training & Evaluation
- Lorenz: Dokumentation
- Andreas: Abgabe

## 8. Fazit
Das Projekt zeigt eine vollständige Pipeline von Training bis Deployment mit interaktivem Feedback.
