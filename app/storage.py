from __future__ import annotations

import csv
import os
from typing import List, Tuple

DEFAULT_FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.csv")


def ensure_feedback_file(path: str = DEFAULT_FEEDBACK_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true", "pred"])


def append_feedback(true_label: int, pred_label: int, path: str = DEFAULT_FEEDBACK_FILE) -> None:
    ensure_feedback_file(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([int(true_label), int(pred_label)])


def load_feedback(path: str = DEFAULT_FEEDBACK_FILE) -> List[Tuple[int, int]]:
    if not os.path.exists(path):
        return []
    rows: List[Tuple[int, int]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                t = int(r["true"])
                p = int(r["pred"])
                rows.append((t, p))
            except Exception:
                # ignoriert kaputte Zeilen
                continue
    return rows

