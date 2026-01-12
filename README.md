# Zeichenerkennung mit MNIST

## Ziel
Entwicklung eines Modells zur Erkennung handschriftlicher Zeichen (MNIST)
und Integration in eine GUI mit Nutzerfeedback und Confusion Matrix.

## Projektstruktur
- `training/` – Modelltraining & Evaluation
- `app/` – GUI & Deployment
- `models/` – Trainierte Modelle
- `report/` – Dokumentation & Screenshots

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
