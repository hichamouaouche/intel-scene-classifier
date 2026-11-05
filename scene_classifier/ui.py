"""Styles et fragments HTML de l'interface Streamlit."""

from html import escape

from .config import CLASSES, LABELS

CSS = """
<style>
.block-container { padding-top: 2.5rem; max-width: 1200px; }
.hero-title { font-size: 2.1rem; font-weight: 700; line-height: 1.2; margin: 0; }
.hero-subtitle { color: #55625F; margin: .35rem 0 1.5rem; }
.overline { color: #55625F; font-size: .8rem; font-weight: 600;
            letter-spacing: .06em; text-transform: uppercase; }
.prediction { font-size: 2.6rem; font-weight: 700; line-height: 1.15; }
.confidence { color: #55625F; margin-bottom: 1.25rem; }
.bars { padding-bottom: .75rem; }
.bar-row { display: grid; grid-template-columns: 6rem 1fr 4.5rem; gap: .75rem;
           align-items: center; padding: .4rem 0; font-size: .95rem; }
.bar-row.top { font-weight: 600; }
.bar-track { background: #E8EDEB; border-radius: 4px; height: 10px; }
.bar-fill { background: #A9B5B2; border-radius: 4px; height: 100%; }
.top .bar-fill { background: #0F766E; }
.bar-value { text-align: right; font-variant-numeric: tabular-nums; }
</style>
"""

HEADER = (
    '<div class="hero-title">Classification de scènes naturelles</div>'
    '<div class="hero-subtitle">Choisissez une image pour découvrir ce que le modèle '
    "y reconnaît.</div>"
)


def format_percent(fraction: float, decimals: int = 2) -> str:
    return f"{fraction * 100:.{decimals}f} %".replace(".", ",")


def prediction_header(predicted_class: str, confidence: float) -> str:
    return (
        '<div class="overline">Scène prédite</div>'
        f'<div class="prediction">{escape(LABELS[predicted_class])}</div>'
        f'<div class="confidence">Confiance : {format_percent(confidence)}</div>'
    )


def probability_bars(probabilities) -> str:
    """Barres horizontales triées, la classe prédite mise en avant."""
    ranked = sorted(zip(CLASSES, probabilities), key=lambda item: -item[1])
    rows = []
    for rank, (name, probability) in enumerate(ranked):
        label, value = escape(LABELS[name]), format_percent(float(probability))
        rows.append(
            f'<div class="bar-row{" top" if rank == 0 else ""}" title="{label} : {value}">'
            f"<div>{label}</div>"
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{float(probability) * 100:.2f}%"></div></div>'
            f'<div class="bar-value">{value}</div></div>'
        )
    return f'<div class="bars">{"".join(rows)}</div>'
