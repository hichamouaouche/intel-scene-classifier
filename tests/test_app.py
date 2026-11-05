import pytest
from streamlit.testing.v1 import AppTest

from scene_classifier.config import ROOT_DIR, TEST_DATA_DIR

pytestmark = pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="dataset de test absent")


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=120).run()


def test_first_image_is_predicted(app):
    assert not app.exception
    assert app.success[0].value == "Correct : la vraie scène est Bâtiments."


def test_class_filter(app):
    app.selectbox(key="class_filter").set_value("Mer").run()
    assert not app.exception
    assert app.selectbox(key="dataset_image").value.startswith("sea/")


def test_random_button(app):
    for _ in range(3):
        app.button[0].click().run()
        assert not app.exception
        assert app.success or app.error
