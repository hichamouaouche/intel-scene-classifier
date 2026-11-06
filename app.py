"""Interface Streamlit : streamlit run app.py"""

import random

import streamlit as st
from PIL import Image, UnidentifiedImageError

from scene_classifier import ui
from scene_classifier.config import (
    ASSETS_DIR,
    CLASSES,
    LABELS,
    LOW_CONFIDENCE,
    MODEL_DIR,
    PROJECT,
    TEST_DATA_DIR,
)
from scene_classifier.dataset import list_images, true_class
from scene_classifier.evaluation import load_metrics
from scene_classifier.model import load_model, predict_proba

ALL_CLASSES = "Toutes les scènes"
SOURCE_DATASET = "Jeu de test"
SOURCE_UPLOAD = "Mon image"

st.set_page_config(page_title="Classification de scènes", page_icon="🏞️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

get_model = st.cache_resource(load_model)
get_images = st.cache_data(list_images)


def pick_random_image():
    choice = random.choice(dataset_images)
    st.session_state.class_filter = LABELS[true_class(choice)]
    st.session_state.dataset_image = choice


def select_dataset_image():
    """Renvoie (chemin de l'image, nom affiché, vraie classe) ou (None, None, None)."""
    if not dataset_images:
        st.warning(f"Aucune image trouvée dans {TEST_DATA_DIR}.")
        return None, None, None
    filter_col, image_col = st.columns(2)
    with filter_col:
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
    with image_col:
        choice = st.selectbox("Image", options, key="dataset_image")
    st.button("Image au hasard", icon=":material/shuffle:", on_click=pick_random_image, width="stretch")
    return TEST_DATA_DIR / choice, choice, true_class(choice)


def render_analysis_tab():
    left, right = st.columns([1, 1.1], gap="large")
    image = image_name = actual_class = None

    with left:
        with st.container(border=True):
            st.markdown(
                ui.card_title(1, "Choisir une image", "Depuis le jeu de test ou depuis votre ordinateur."),
                unsafe_allow_html=True,
            )
            source = st.segmented_control(
                "Source de l'image",
                [SOURCE_DATASET, SOURCE_UPLOAD],
                default=SOURCE_DATASET,
                required=True,
                label_visibility="collapsed",
            )
            if source == SOURCE_DATASET:
                image_source, image_name, actual_class = select_dataset_image()
            else:
                image_source = st.file_uploader("Importer une image", type=["png", "jpg", "jpeg"])
                image_name = image_source.name if image_source else None

            if image_source is not None:
                try:
                    image = Image.open(image_source).convert("RGB")
                except (OSError, UnidentifiedImageError):
                    st.error("Ce fichier n'a pas pu être lu comme une image.", icon=":material/error:")
            if image is not None:
                st.image(image, width="stretch")
                st.markdown(ui.image_meta(image_name, image.size), unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown(ui.card_title(2, "Résultat de l'analyse"), unsafe_allow_html=True)
            if image is None:
                st.markdown(
                    ui.empty_state("Aucune image à analyser", "Importez une image PNG ou JPG pour lancer la prédiction."),
                    unsafe_allow_html=True,
                )
                return
            probabilities = predict_proba(model, [image])[0]
            predicted_class = CLASSES[int(probabilities.argmax())]
            confidence = float(probabilities.max())
            st.markdown(
                ui.prediction_header(predicted_class, confidence)
                + ui.verdicts(predicted_class, actual_class, confidence < LOW_CONFIDENCE)
                + ui.probability_bars(probabilities),
                unsafe_allow_html=True,
            )


def render_performance_tab():
    if not metrics:
        st.info("Le modèle n'a pas encore été évalué : lancez python -m scripts.evaluate.")
        return
    st.markdown(ui.kpis(metrics), unsafe_allow_html=True)
    table_col, heatmap_col = st.columns([1, 1.25], gap="large")
    with table_col:
        with st.container(border=True):
            st.markdown(
                ui.card_title("A", "Résultats par scène", "Rappel : part des images de la scène retrouvées. "
                              "Précision : part des prédictions de la scène qui sont justes."),
                unsafe_allow_html=True,
            )
            st.markdown(ui.class_table(metrics), unsafe_allow_html=True)
    with heatmap_col:
        with st.container(border=True):
            st.markdown(
                ui.card_title("B", "Matrice de confusion", "Où le modèle se trompe : plus la case est foncée, "
                              "plus la confusion est fréquente."),
                unsafe_allow_html=True,
            )
            st.markdown(ui.confusion_heatmap(metrics), unsafe_allow_html=True)


def render_about_tab():
    model_col, credits_col = st.columns([1.4, 1], gap="large")
    with model_col:
        with st.container(border=True):
            st.markdown(ui.card_title("i", "Le modèle et les données"), unsafe_allow_html=True)
            st.markdown(
                f"""
<div class="about">

Le modèle est un **réseau de neurones convolutif** de trois blocs (32, 64 puis 128 filtres),
avec Batch Normalization, Dropout et Global Average Pooling. Il compte
**{ui.format_number(model.count_params())} paramètres** et reçoit des images de 150 × 150 pixels.

- **Dataset :** Intel Image Classification, six scènes, 14 034 images d'entraînement et 3 000 de test.
- **Prétraitement :** l'image est redimensionnée en 150 × 150 pixels. La normalisation est faite par le modèle.
- **Augmentation :** retournement, rotation, zoom et contraste aléatoires, actifs à l'entraînement seulement.

**Limites.** Le modèle choisit toujours l'une des six scènes, même pour une image qui n'y
correspond pas. Une confiance élevée ne garantit pas une bonne réponse.

</div>
""",
                unsafe_allow_html=True,
            )
    with credits_col:
        with st.container(border=True):
            st.markdown(ui.card_title("✦", "Projet académique"), unsafe_allow_html=True)
            logo_left, logo_right = st.columns([1, 1.6], vertical_alignment="center")
            logo_left.image(str(ASSETS_DIR / "logo-uh2c.png"), width=110)
            logo_right.image(str(ASSETS_DIR / "logo-enset.png"), width="stretch")
            st.markdown(
                f"""
<div class="credit-label">Réalisé par</div><div class="credit-name">{PROJECT['author']}</div>
<div class="credit-label">Encadrant</div><div class="credit-name">{PROJECT['supervisor']}</div>
<div class="credit-label">Module {PROJECT['module']} · {PROJECT['school']} · {PROJECT['university']}</div>
""",
                unsafe_allow_html=True,
            )


try:
    model = get_model()
except (OSError, ValueError) as error:
    st.error(f"Impossible de charger le modèle {MODEL_DIR.name} : {error}")
    st.stop()

dataset_images = get_images()
metrics = load_metrics()

st.markdown(ui.hero(metrics, model.count_params()), unsafe_allow_html=True)
analysis_tab, performance_tab, about_tab = st.tabs(
    [":material/image_search: Analyser une image", ":material/insights: Performances du modèle", ":material/info: À propos"]
)
with analysis_tab:
    render_analysis_tab()
with performance_tab:
    render_performance_tab()
with about_tab:
    render_about_tab()
