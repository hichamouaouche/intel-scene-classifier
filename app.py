"""Interface Streamlit : streamlit run app.py"""

import random

import streamlit as st
from PIL import Image, UnidentifiedImageError

from scene_classifier import ui
from scene_classifier.config import (
    CLASSES,
    LABELS,
    LOW_CONFIDENCE,
    MODEL_DIR,
    TEST_DATA_DIR,
)
from scene_classifier.dataset import list_images, true_class
from scene_classifier.evaluation import load_metrics
from scene_classifier.model import load_model, predict_proba

ALL_CLASSES = "Toutes"
SOURCE_DATASET = "Dataset de test"
SOURCE_UPLOAD = "Mon image"

st.set_page_config(page_title="Classification de scènes", page_icon="🏞️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

get_model = st.cache_resource(load_model)
get_images = st.cache_data(list_images)


def pick_random_image():
    choice = random.choice(dataset_images)
    st.session_state.class_filter = LABELS[true_class(choice)]
    st.session_state.dataset_image = choice


def render_sidebar():
    metrics = load_metrics()
    with st.sidebar:
        st.header("Le modèle")
        if metrics:
            st.metric("Précision sur le jeu de test", ui.format_percent(metrics["accuracy"], 1))
            st.caption(f"Mesurée sur {metrics['images']:,} images de seg_test.".replace(",", " "))
        else:
            st.caption("Pas encore évalué : lancez python -m scripts.evaluate.")
        st.divider()
        st.markdown("**Scènes reconnues**")
        st.markdown("\n".join(f"- {LABELS[name]}" for name in CLASSES))
        st.divider()
        st.caption("CNN entraîné sur le dataset Intel, images en 150 × 150 pixels.")


def select_dataset_image():
    """Renvoie (chemin de l'image, vraie classe) ou (None, None)."""
    if not dataset_images:
        st.warning(f"Aucune image trouvée dans {TEST_DATA_DIR}.")
        return None, None
    class_label = st.selectbox(
        "Scène", [ALL_CLASSES] + [LABELS[name] for name in CLASSES], key="class_filter"
    )
    options = [
        path
        for path in dataset_images
        if class_label == ALL_CLASSES or LABELS[true_class(path)] == class_label
    ]
    if st.session_state.get("dataset_image") not in options:
        st.session_state.dataset_image = options[0]
    choice = st.selectbox("Image", options, key="dataset_image")
    st.button("Image au hasard", icon="🎲", on_click=pick_random_image, width="stretch")
    return TEST_DATA_DIR / choice, true_class(choice)


def render_prediction(probabilities, actual_class):
    predicted_class = CLASSES[int(probabilities.argmax())]
    confidence = float(probabilities.max())
    with st.container(border=True):
        st.markdown(ui.prediction_header(predicted_class, confidence), unsafe_allow_html=True)
        if actual_class == predicted_class:
            st.success(f"Correct : la vraie scène est {LABELS[actual_class]}.", icon="✅")
        elif actual_class is not None:
            st.error(f"Incorrect : la vraie scène est {LABELS[actual_class]}.", icon="❌")
        if confidence < LOW_CONFIDENCE:
            st.warning("Confiance faible : le modèle hésite entre plusieurs scènes.", icon="⚠️")
        st.markdown("**Probabilités par scène**")
        st.markdown(ui.probability_bars(probabilities), unsafe_allow_html=True)


try:
    model = get_model()
except (OSError, ValueError) as error:
    st.error(f"Impossible de charger le modèle {MODEL_DIR.name} : {error}")
    st.stop()

dataset_images = get_images()
render_sidebar()
st.markdown(ui.HEADER, unsafe_allow_html=True)

left, right = st.columns([1, 1.15], gap="large")
image_source = actual_class = None

with left:
    with st.container(border=True):
        source = st.segmented_control(
            "Source de l'image", [SOURCE_DATASET, SOURCE_UPLOAD], default=SOURCE_DATASET, required=True
        )
        if source == SOURCE_DATASET:
            image_source, actual_class = select_dataset_image()
        else:
            image_source = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])

    image = None
    if image_source is not None:
        try:
            image = Image.open(image_source).convert("RGB")
        except (OSError, UnidentifiedImageError):
            st.error("Ce fichier n'a pas pu être lu comme une image.")
    if image is not None:
        st.image(image, width="stretch")

with right:
    if image is not None:
        render_prediction(predict_proba(model, [image])[0], actual_class)
    elif source == SOURCE_UPLOAD:
        st.info("Importez une image pour lancer la prédiction.", icon="🖼️")
