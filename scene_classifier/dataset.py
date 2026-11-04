"""Accès aux images du dataset, rangées dans un dossier par classe."""

from pathlib import Path

from .config import IMAGE_EXTENSIONS, TEST_DATA_DIR


def list_images(directory: Path = TEST_DATA_DIR) -> list[str]:
    """Chemins relatifs « classe/fichier », triés."""
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.glob("*/*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def true_class(relative_path: str) -> str:
    return relative_path.split("/")[0]
