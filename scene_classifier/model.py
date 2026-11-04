"""Chargement du modèle et prédiction sur des images PIL."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import CLASSES, MODEL_DIR


def load_model(model_dir: Path = MODEL_DIR) -> tf.keras.Model:
    if not model_dir.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_dir}")
    # Le modèle est un dossier décompressé : load_model n'accepte que l'archive.
    model = tf.keras.models.model_from_json((model_dir / "config.json").read_text())
    model.load_weights(model_dir / "model.weights.h5")
    if model.output_shape[-1] != len(CLASSES):
        raise ValueError(
            f"Le modèle a {model.output_shape[-1]} sorties pour {len(CLASSES)} classes."
        )
    return model


def preprocess(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Convertit une image en tableau (hauteur, largeur, 3).

    Pas de normalisation : le modèle contient sa propre couche Rescaling.
    """
    return np.array(image.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)


def predict_proba(model: tf.keras.Model, images: list[Image.Image]) -> np.ndarray:
    """Renvoie les probabilités par classe, de forme (nombre d'images, classes)."""
    _, height, width, _ = model.input_shape
    batch = np.stack([preprocess(image, (width, height)) for image in images])
    return model(batch, training=False).numpy()
