# Classification de scènes naturelles

Application web Streamlit qui reconnaît le type de scène d'une photo à l'aide
d'un réseau de neurones convolutif (CNN) entraîné sur le dataset
[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

Six scènes sont reconnues : bâtiments, forêt, glacier, montagne, mer et rue.

**Démo en ligne : https://shareappio-cmmmwerrt4ve6cbgmdhjyq.streamlit.app/**

![Onglet « Analyser une image »](docs/screenshots/accueil.png)

## Fonctionnalités

L'interface est organisée en trois onglets.

**Analyser une image**

- Choix d'une image du jeu de test, filtrée par scène ou tirée au hasard, ou import de sa propre image (PNG ou JPG).
- Scène prédite, confiance du modèle et probabilités des six scènes.
- Pastilles de statut : Correct ou Incorrect pour une image du jeu de test, et Confiance faible sous 60 %.

| Prédiction incorrecte | Confiance faible |
|---|---|
| ![Prédiction incorrecte](docs/screenshots/prediction-incorrecte.png) | ![Avertissement de confiance faible](docs/screenshots/confiance-faible.png) |

| Image importée | Fichier invalide |
|---|---|
| ![Prédiction sur une image importée](docs/screenshots/import-image.png) | ![Message pour un fichier invalide](docs/screenshots/fichier-invalide.png) |

**Performances du modèle** : indicateurs clés, rappel et précision par scène, et matrice de confusion en carte de chaleur.

![Onglet « Performances du modèle »](docs/screenshots/performances.png)

**À propos** : description du modèle et des données, limites, et cadre académique du projet.

![Onglet « À propos »](docs/screenshots/a-propos.png)

L'interface s'adapte aussi aux écrans de téléphone.

<img src="docs/screenshots/mobile.png" alt="Interface sur mobile" width="260">

## Résultats

Sur les 3 000 images du jeu de test (`seg_test`), le modèle atteint **81,7 %** de
précision.

| Scène | Rappel | Précision |
|---|---|---|
| Bâtiments | 84,4 % | 78,7 % |
| Forêt | 98,3 % | 86,5 % |
| Glacier | 70,7 % | 82,8 % |
| Montagne | 82,9 % | 73,5 % |
| Mer | 87,1 % | 81,8 % |
| Rue | 69,3 % | 90,1 % |

Les confusions les plus fréquentes concernent des scènes proches : rue et
bâtiments, glacier et montagne.

## Le modèle

CNN de trois blocs convolutifs (32, 64 puis 128 filtres) avec Batch
Normalization, Dropout et Global Average Pooling, soit 323 366 paramètres.
Les images d'entrée font 150 × 150 pixels. La normalisation et l'augmentation
des données (retournement, rotation, zoom, contraste) sont intégrées au modèle.

## Structure du projet

```
.
├── app.py                    # Interface Streamlit (point d'entrée)
├── scene_classifier/         # Code métier, indépendant de l'interface
│   ├── config.py             # Chemins, classes et seuils
│   ├── model.py              # Chargement du modèle et prédiction
│   ├── dataset.py            # Accès aux images du dataset
│   ├── evaluation.py         # Calcul et lecture des métriques
│   └── ui.py                 # Styles et composants visuels de l'interface
├── scripts/
│   └── evaluate.py           # Évalue le modèle et écrit models/metrics.json
├── tests/                    # Tests automatiques (pytest)
├── models/
│   ├── cnn_intel_best.keras/ # Modèle entraîné (architecture et poids)
│   └── metrics.json          # Métriques sur le jeu de test
├── samples/                  # Échantillon de 60 images du jeu de test
├── data/                     # Dataset Intel : seg_train, seg_test, seg_pred
├── assets/                   # Logos de l'ENSET et de l'UH2C
├── docs/screenshots/         # Captures d'écran
├── .streamlit/config.toml    # Thème de l'interface
├── requirements.txt          # Dépendances de l'application
└── requirements-dev.txt      # Dépendances de développement (tests)
```

## Installation

Python 3.14 est requis pour les versions indiquées dans `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

L'application fonctionne directement avec l'échantillon de 60 images du
dossier `samples/` et les métriques fournies dans `models/metrics.json`.

Pour utiliser les 3 000 images du jeu de test ou recalculer les métriques,
téléchargez le dataset depuis
[Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
et placez-le dans `data/`, avec un dossier par scène :
`data/seg_test/seg_test/<scène>/<image>.jpg`. L'application l'utilise alors
automatiquement à la place de l'échantillon.

## Utilisation

Toutes les commandes se lancent depuis la racine du projet.

| Action | Commande |
|---|---|
| Lancer l'application | `streamlit run app.py` |
| Recalculer les métriques | `python -m scripts.evaluate` |
| Lancer les tests | `python -m pytest` |

L'application s'ouvre sur <http://localhost:8501>.

## Limites

- Le modèle choisit toujours l'une des six scènes, même pour une image qui n'y
  correspond pas.
- Une confiance élevée ne garantit pas une bonne réponse : l'avertissement
  affiché sous 60 % ne repère qu'une partie des erreurs.
- Le modèle est enregistré sous forme de dossier décompressé, d'où son chargement
  par `config.json` et `model.weights.h5` plutôt que par `keras.models.load_model`.

## Auteur

Projet réalisé par **Hicham OUAOUCHE** dans le cadre du module IA et Frameworks
à l'École Normale Supérieure de l'Enseignement Technique (ENSET) de Mohammedia,
Université Hassan II de Casablanca.

Encadrant : **Asmae OUHMIDA**
