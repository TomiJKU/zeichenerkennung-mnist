from __future__ import annotations

import numpy as np
from typing import Iterable, Tuple


def confusion_matrix(pairs: Iterable[Tuple[int, int]], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in pairs:
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def format_confusion_matrix(cm: np.ndarray) -> str:
    n = cm.shape[0]
    lines = []
    header = "     " + " ".join([f"{i:4d}" for i in range(n)])
    lines.append(header)
    for i in range(n):
        lines.append(f"{i:2d}: " + " ".join([f"{cm[i, j]:4d}" for j in range(n)]))
    return "\n".join(lines)

