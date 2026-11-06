import pytest
from streamlit.testing.v1 import AppTest

from scene_classifier.config import IMAGE_DIR, METRICS_PATH, ROOT_DIR

pytestmark = pytest.mark.skipif(not IMAGE_DIR.exists(), reason="aucune image disponible")


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT_DIR / "app.py"), default_timeout=120).run()


def page_html(app) -> str:
    return "".join(element.value for element in app.markdown)


def test_first_image_is_predicted(app):
    assert not app.exception
    assert "Correct : la vraie scène est Bâtiments" in page_html(app)


def test_class_filter(app):
    app.selectbox(key="class_filter").set_value("Mer").run()
    assert not app.exception
    assert app.selectbox(key="dataset_image").value.startswith("sea/")


def test_random_button(app):
    for _ in range(3):
        app.button[0].click().run()
        assert not app.exception
        assert "la vraie scène est" in page_html(app)


@pytest.mark.skipif(not METRICS_PATH.exists(), reason="métriques non calculées")
def test_performance_tab(app):
    html = page_html(app)
    assert "Précision globale" in html
    assert 'class="heatmap"' in html
