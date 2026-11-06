"""Chemins et constantes partagés par l'application, les scripts et les tests."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models" / "cnn_intel_best.keras"
TEST_DATA_DIR = ROOT_DIR / "data" / "seg_test" / "seg_test"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
ASSETS_DIR = ROOT_DIR / "assets"

# Même ordre (alphabétique) que les sorties du modèle à l'entraînement.
CLASSES = ("buildings", "forest", "glacier", "mountain", "sea", "street")
LABELS = {
    "buildings": "Bâtiments",
    "forest": "Forêt",
    "glacier": "Glacier",
    "mountain": "Montagne",
    "sea": "Mer",
    "street": "Rue",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# En dessous de ce seuil, l'interface signale une prédiction peu sûre.
LOW_CONFIDENCE = 0.60

PROJECT = {
    "author": "Hicham OUAOUCHE",
    "supervisor": "Asmae OUHMIDA",
    "school": "ENSET Mohammedia",
    "university": "Université Hassan II de Casablanca",
    "module": "IA et Frameworks",
}
