import numpy as np
import pytest
from PIL import Image

from scene_classifier.config import CLASSES, IMAGE_DIR, SAMPLE_DATA_DIR, TEST_DATA_DIR
from scene_classifier.dataset import list_images, true_class
from scene_classifier.evaluation import compute_metrics
from scene_classifier.model import load_model, predict_proba, preprocess


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_model_matches_classes(model):
    assert model.input_shape == (None, 150, 150, 3)
    assert model.output_shape == (None, len(CLASSES))


def test_preprocess_keeps_pixel_range():
    array = preprocess(Image.new("L", (300, 200), color=255), (150, 150))
    assert array.shape == (150, 150, 3)
    assert array.dtype == np.float32
    assert array.max() == 255  # la normalisation est faite dans le modèle


def test_probabilities_sum_to_one(model):
    probabilities = predict_proba(model, [Image.new("RGB", (150, 150)), Image.new("RGB", (90, 60))])
    assert probabilities.shape == (2, len(CLASSES))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, rtol=1e-5)


@pytest.mark.skipif(not IMAGE_DIR.exists(), reason="aucune image disponible")
def test_known_image_is_classified(model):
    path = "buildings/20057.jpg"
    probabilities = predict_proba(model, [Image.open(IMAGE_DIR / path)])[0]
    assert CLASSES[probabilities.argmax()] == true_class(path)


@pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="dataset de test absent")
def test_dataset_has_every_class():
    assert {true_class(path) for path in list_images()} == set(CLASSES)


def test_compute_metrics():
    y_true = np.array([0, 0, 1])
    probabilities = np.zeros((3, len(CLASSES)))
    probabilities[0, 0] = probabilities[1, 1] = 1.0
    probabilities[2, 1] = 0.5
    probabilities[2, 2] = 0.5
    metrics = compute_metrics(y_true, probabilities)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"]["rows"][0][:2] == [1, 1]
    assert metrics["low_confidence"]["images"] == 1


def test_samples_cover_every_class():
    assert {true_class(path) for path in list_images(SAMPLE_DATA_DIR)} == set(CLASSES)
