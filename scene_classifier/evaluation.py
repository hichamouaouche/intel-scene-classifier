"""Calcul et lecture des métriques d'évaluation du modèle."""

import json
from pathlib import Path

import numpy as np

from .config import CLASSES, LOW_CONFIDENCE, METRICS_PATH


def compute_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = y_true == y_pred

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        confusion[truth, prediction] += 1

    low = confidence < LOW_CONFIDENCE
    return {
        "images": int(len(y_true)),
        "accuracy": float(correct.mean()),
        "per_class": {
            name: {
                "images": int(confusion[i].sum()),
                "recall": float(confusion[i, i] / max(confusion[i].sum(), 1)),
                "precision": float(confusion[i, i] / max(confusion[:, i].sum(), 1)),
            }
            for i, name in enumerate(CLASSES)
        },
        "confusion_matrix": {"labels": list(CLASSES), "rows": confusion.tolist()},
        "low_confidence": {
            "threshold": LOW_CONFIDENCE,
            "images": int(low.sum()),
            "accuracy": float(correct[low].mean()) if low.any() else None,
        },
    }


def load_metrics(path: Path = METRICS_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
