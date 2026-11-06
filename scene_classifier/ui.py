"""Styles et fragments HTML de l'interface Streamlit."""

from html import escape

from .config import CLASSES, LABELS

TEAL = "#0F766E"
INK = "#16202A"
MUTED = "#5B6773"

CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }

/* Bandeau d'en-tête */
.hero { background: linear-gradient(120deg, #0F766E 0%, #134E6F 100%); color: #fff;
        border-radius: 16px; padding: 1.75rem 2rem; margin-bottom: 1.25rem; }
.hero-eyebrow { font-size: .78rem; font-weight: 600; letter-spacing: .08em;
                text-transform: uppercase; opacity: .8; }
.hero-title { font-size: 2rem; font-weight: 700; line-height: 1.2; margin: .3rem 0; }
.hero-subtitle { opacity: .88; max-width: 44rem; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.1rem; }
.chip { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
        border-radius: 999px; padding: .25rem .8rem; font-size: .85rem; }
.chip b { font-weight: 700; }

/* Titres de carte */
.card-title { display: flex; align-items: center; gap: .6rem; font-weight: 700;
              font-size: 1.05rem; margin-bottom: .35rem; }
.step { display: inline-flex; align-items: center; justify-content: center;
        width: 1.6rem; height: 1.6rem; border-radius: 999px; background: #E3F1EF;
        color: #0F766E; font-size: .85rem; font-weight: 700; }
.card-subtitle { color: #5B6773; font-size: .9rem; margin-bottom: .75rem; }
.image-meta { color: #5B6773; font-size: .85rem; margin-top: -.25rem; }
.image-meta code { background: #EEF2F4; color: #16202A; padding: .05rem .35rem; border-radius: 4px; }

/* Résultat */
.overline { color: #5B6773; font-size: .78rem; font-weight: 600;
            letter-spacing: .07em; text-transform: uppercase; }
.prediction { font-size: 2.6rem; font-weight: 700; line-height: 1.15; color: #16202A; }
.confidence { color: #5B6773; margin: .15rem 0 .9rem; }
.confidence b { color: #16202A; }
.verdicts { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1.1rem; }
.verdict { display: inline-flex; align-items: center; gap: .4rem; border-radius: 999px;
           padding: .3rem .8rem; font-size: .88rem; font-weight: 600; }
.verdict.ok { background: #E3F2E8; color: #1D6B3A; }
.verdict.ko { background: #FBE4E4; color: #A12A2A; }
.verdict.warn { background: #FBF1D6; color: #7A5600; }
.verdict.info { background: #EEF2F4; color: #3F4B57; }
.bars-title { font-weight: 600; margin-bottom: .35rem; }
.bars { padding-bottom: .5rem; }
.bar-row { display: grid; grid-template-columns: 6.5rem 1fr 4.5rem; gap: .75rem;
           align-items: center; padding: .38rem 0; font-size: .95rem; }
.bar-row.top { font-weight: 600; }
.bar-track { background: #E9EEF1; border-radius: 4px; height: 10px; }
.bar-fill { background: #A9B4BC; border-radius: 4px; height: 100%; }
.top .bar-fill { background: #0F766E; }
.bar-value { text-align: right; font-variant-numeric: tabular-nums; }

/* État vide */
.empty { border: 1.5px dashed #C9D2D9; border-radius: 12px; padding: 2.5rem 1.5rem;
         text-align: center; color: #5B6773; background: #FAFBFC; }
.empty-icon { font-size: 2rem; margin-bottom: .4rem; }
.empty b { color: #16202A; display: block; margin-bottom: .2rem; }

/* Indicateurs clés */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem; margin: .5rem 0 1.25rem; }
.kpi { background: #fff; border: 1px solid #DDE3E8; border-radius: 12px; padding: 1rem 1.2rem; }
.kpi-label { color: #5B6773; font-size: .85rem; }
.kpi-value { font-size: 1.9rem; font-weight: 700; line-height: 1.25; color: #16202A; }
.kpi-note { color: #5B6773; font-size: .82rem; }

/* Tableau par scène */
.perf { width: 100%; border-collapse: collapse; font-size: .95rem; table-layout: fixed; border: none; }
.perf th, .perf td { border-left: none !important; border-right: none !important; border-top: none !important; background: none; }
.perf col.name { width: 22%; } .perf col.count { width: 16%; } .perf col.metric { width: 31%; }
.perf th { white-space: nowrap; }
.perf th { text-align: left; color: #5B6773; font-weight: 600; font-size: .82rem;
           text-transform: uppercase; letter-spacing: .04em; padding: .5rem .5rem;
           border-bottom: 1px solid #DDE3E8; }
.perf td { padding: .55rem .5rem; border-bottom: 1px solid #EEF2F4; vertical-align: middle; }
.perf tr:last-child td { border-bottom: none; }
.perf .num { text-align: right; font-variant-numeric: tabular-nums; }
.metric-cell { display: grid; grid-template-columns: minmax(40px, 1fr) 3.6rem; gap: .6rem; align-items: center; }
.metric-cell .bar-track { height: 8px; }
.metric-cell .bar-fill { background: #0F766E; }

/* Matrice de confusion */
.heatmap { width: 100%; border-collapse: separate; border-spacing: 3px; font-size: .9rem; border: none; }
.heatmap th, .heatmap td { border: none !important; }
.heatmap th { background: none; }
.heatmap th { color: #5B6773; font-weight: 600; font-size: .8rem; padding: .3rem; }
.heatmap th.row { text-align: right; padding-right: .6rem; white-space: nowrap; }
.heatmap td { text-align: center; border-radius: 6px; padding: .55rem .2rem;
              font-variant-numeric: tabular-nums; }
.heatmap td small { display: block; font-size: .72rem; opacity: .8; }
.axis-note { color: #5B6773; font-size: .82rem; margin-top: .4rem; }
.legend { display: flex; align-items: center; gap: .5rem; color: #5B6773; font-size: .8rem; margin-top: .6rem; }
.legend-ramp { width: 160px; height: 8px; border-radius: 4px;
               background: linear-gradient(90deg, #F1F6F6, #0B4F4A); }

/* À propos */
.about p, .about li { color: #3F4B57; }
.credit-label { color: #5B6773; font-size: .82rem; }
.credit-name { font-weight: 700; font-size: 1.05rem; margin-bottom: .6rem; }

@media (max-width: 640px) {
  .hero { padding: 1.25rem; }
  .hero-title { font-size: 1.5rem; }
  .prediction { font-size: 2rem; }
}
</style>
"""


def format_percent(fraction: float, decimals: int = 2) -> str:
    return f"{fraction * 100:.{decimals}f} %".replace(".", ",")


def format_number(value: int) -> str:
    # Espace insécable : le nombre ne se coupe jamais en fin de ligne.
    return f"{value:,}".replace(",", "\u00a0")


def hero(metrics: dict | None, parameters: int) -> str:
    chips = [f"<b>{len(CLASSES)}</b> scènes", f"<b>{format_number(parameters)}</b> paramètres", "Entrée 150 × 150 px"]
    if metrics:
        chips.insert(0, f"Précision <b>{format_percent(metrics['accuracy'], 1)}</b>")
    return (
        '<div class="hero"><div class="hero-eyebrow">Deep learning · Vision par ordinateur</div>'
        '<div class="hero-title">Classification de scènes naturelles</div>'
        '<div class="hero-subtitle">Un réseau de neurones convolutif reconnaît le type de scène '
        "d'une photo : bâtiments, forêt, glacier, montagne, mer ou rue.</div>"
        f'<div class="chips">{"".join(f"<span class=chip>{c}</span>" for c in chips)}</div></div>'
    )


def card_title(step: int | str, title: str, subtitle: str = "") -> str:
    html = f'<div class="card-title"><span class="step">{step}</span>{escape(title)}</div>'
    if subtitle:
        html += f'<div class="card-subtitle">{escape(subtitle)}</div>'
    return html


def image_meta(name: str, size: tuple[int, int]) -> str:
    return f'<div class="image-meta"><code>{escape(name)}</code> · {size[0]} × {size[1]} px</div>'


def empty_state(title: str, text: str, icon: str = "🖼️") -> str:
    return f'<div class="empty"><div class="empty-icon">{icon}</div><b>{escape(title)}</b>{escape(text)}</div>'


def prediction_header(predicted_class: str, confidence: float) -> str:
    return (
        '<div class="overline">Scène prédite</div>'
        f'<div class="prediction">{escape(LABELS[predicted_class])}</div>'
        f'<div class="confidence">Confiance du modèle : <b>{format_percent(confidence)}</b></div>'
    )


def verdicts(predicted_class: str, actual_class: str | None, low_confidence: bool) -> str:
    """Pastilles de statut : chacune porte une icône et un texte, jamais la couleur seule."""
    items = []
    if actual_class == predicted_class:
        items.append(("ok", "✓", f"Correct : la vraie scène est {LABELS[actual_class]}"))
    elif actual_class is not None:
        items.append(("ko", "✕", f"Incorrect : la vraie scène est {LABELS[actual_class]}"))
    else:
        items.append(("info", "ⓘ", "Image importée : vraie scène inconnue"))
    if low_confidence:
        items.append(("warn", "!", "Confiance faible : le modèle hésite"))
    pills = "".join(
        f'<span class="verdict {kind}"><span aria-hidden="true">{icon}</span>{escape(text)}</span>'
        for kind, icon, text in items
    )
    return f'<div class="verdicts">{pills}</div>'


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
    return (
        '<div class="bars-title">Probabilités par scène</div>'
        f'<div class="bars">{"".join(rows)}</div>'
    )


def kpis(metrics: dict) -> str:
    per_class = metrics["per_class"]
    best = max(per_class, key=lambda name: per_class[name]["recall"])
    worst = min(per_class, key=lambda name: per_class[name]["recall"])
    low = metrics["low_confidence"]
    tiles = [
        ("Précision globale", format_percent(metrics["accuracy"], 1), "images bien classées"),
        ("Images évaluées", format_number(metrics["images"]), "jeu de test seg_test"),
        ("Scène la mieux reconnue", LABELS[best], f"rappel {format_percent(per_class[best]['recall'], 1)}"),
        ("Scène la plus difficile", LABELS[worst], f"rappel {format_percent(per_class[worst]['recall'], 1)}"),
        (
            "Prédictions peu sûres",
            format_percent(low["images"] / metrics["images"], 1),
            f"confiance sous {format_percent(low['threshold'], 0)}, "
            f"{format_percent(low['accuracy'], 1)} justes",
        ),
    ]
    return '<div class="kpis">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{escape(value)}</div><div class="kpi-note">{escape(note)}</div></div>'
        for label, value, note in tiles
    ) + "</div>"


def _metric_cell(value: float) -> str:
    return (
        f'<div class="metric-cell"><div class="bar-track"><div class="bar-fill" '
        f'style="width:{value * 100:.1f}%"></div></div>'
        f'<div class="num">{format_percent(value, 1)}</div></div>'
    )


def class_table(metrics: dict) -> str:
    rows = "".join(
        f"<tr><td><b>{LABELS[name]}</b></td><td class=num>{format_number(scores['images'])}</td>"
        f"<td>{_metric_cell(scores['recall'])}</td><td>{_metric_cell(scores['precision'])}</td></tr>"
        for name, scores in metrics["per_class"].items()
    )
    return (
        '<table class="perf"><colgroup><col class="name"><col class="count">'
        '<col class="metric"><col class="metric"></colgroup><thead><tr><th>Scène</th><th class=num>Images</th>'
        "<th>Rappel</th><th>Précision</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _ramp(fraction: float) -> tuple[str, str]:
    """Rampe séquentielle à une teinte : clair (0) vers sarcelle foncé (1)."""
    start, end = (0xF1, 0xF6, 0xF6), (0x0B, 0x4F, 0x4A)
    rgb = [round(a + (b - a) * fraction) for a, b in zip(start, end)]
    return "#{:02X}{:02X}{:02X}".format(*rgb), ("#FFFFFF" if fraction > 0.45 else INK)


def confusion_heatmap(metrics: dict) -> str:
    labels = [LABELS[name] for name in metrics["confusion_matrix"]["labels"]]
    rows = metrics["confusion_matrix"]["rows"]
    header = "<tr><th></th>" + "".join(f"<th>{label}</th>" for label in labels) + "</tr>"
    body = []
    for truth, counts in zip(labels, rows):
        total = sum(counts) or 1
        cells = []
        for predicted, count in zip(labels, counts):
            share = count / total
            background, color = _ramp(share)
            cells.append(
                f'<td style="background:{background};color:{color}" '
                f'title="Vraie scène {truth}, prédite {predicted} : {count} images ({format_percent(share, 1)})">'
                f"{count}<small>{format_percent(share, 0)}</small></td>"
            )
        body.append(f'<tr><th class="row">{truth}</th>{"".join(cells)}</tr>')
    return (
        f'<table class="heatmap"><thead>{header}</thead><tbody>{"".join(body)}</tbody></table>'
        '<div class="axis-note">Lignes : vraie scène · Colonnes : scène prédite · '
        "Pourcentages calculés par ligne.</div>"
        '<div class="legend">0 %<span class="legend-ramp"></span>100 % des images de la scène</div>'
    )
