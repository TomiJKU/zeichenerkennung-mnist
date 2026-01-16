# Zeichenerkennung – MNIST & EMNIST mit GUI, Feedback und Confusion Matrix

## Ziel des Projekts
Ziel dieses Projekts ist die Entwicklung eines Modells zur Erkennung handschriftlicher Zeichen sowie dessen Einsatz in einer grafischen Applikation (GUI).

Das Projekt umfasst:
- Training eines neuronalen Netzes auf MNIST (Ziffern 0–9)
- Erweiterung auf EMNIST ByClass (62 Klassen: 0–9, A–Z, a–z)
- Deployment des Modells in einer GUI
- Benutzerfeedback (richtig/falsch)
- Laufend aktualisierte Confusion Matrix basierend auf gespeichertem Feedback

## Projektstruktur
zeichenerkennung-mnist/
├── app/
├── training/
├── models/
├── report/
├── requirements.txt
├── .gitignore
└── README.md

## Setup
### Repository klonen
```bash
git clone https://github.com/TomiJKU/zeichenerkennung-mnist.git
cd zeichenerkennung-mnist

```
## Virtuelle Umgebung
```bash
python -m venv .venv
```
## Aktivieren:

- Windows: .venv\Scripts\Activate.ps1
- macOS/Linux: source .venv/bin/activate

## Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

## GUI starten
python -m app.gui

## Training (EMNIST – 62 Klassen)
```bash
python -m training.train_emnist_byclass
```
Das Modell wird unter models/best_model.keras gespeichert.
