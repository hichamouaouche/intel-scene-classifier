"""Évalue le modèle sur seg_test et écrit reports/metrics.json.

Utilisation : python -m scripts.evaluate
"""

import json

import numpy as np
from PIL import Image

from scene_classifier.config import CLASSES, LABELS, METRICS_PATH, TEST_DATA_DIR
from scene_classifier.dataset import list_images, true_class
from scene_classifier.evaluation import compute_metrics
from scene_classifier.model import load_model, predict_proba

BATCH_SIZE = 64


def main():
    model = load_model()
    paths = list_images(TEST_DATA_DIR)
    if not paths:
        raise SystemExit(f"Aucune image trouvée dans {TEST_DATA_DIR}")

    y_true = np.array([CLASSES.index(true_class(path)) for path in paths])
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = [Image.open(TEST_DATA_DIR / path) for path in paths[start : start + BATCH_SIZE]]
        batches.append(predict_proba(model, images))
        print(f"\r{min(start + BATCH_SIZE, len(paths))}/{len(paths)} images", end="", flush=True)
    print()

    metrics = compute_metrics(y_true, np.concatenate(batches))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Précision globale : {metrics['accuracy']:.2%} sur {metrics['images']} images")
    for name, scores in metrics["per_class"].items():
        print(f"  {LABELS[name]:<10} rappel {scores['recall']:.1%}  précision {scores['precision']:.1%}")
    print(f"Métriques écrites dans {METRICS_PATH}")


if __name__ == "__main__":
    main()
