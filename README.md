# Optimisation des rendements agricoles

## Jeux de données

Le brief de [mission.md](/Users/steph/Code/Python/Jupyter/OCR_Projet12/ressources/mission.md) distingue deux grands ensembles :

- `Agriculture CropYield Dataset` : données de rendement utilisées pour l'analyse des facteurs clés ;
- `CropYield Prediction Dataset` : données agronomiques et climatiques annuelles utilisées pour valider cette analyse et construire la base de modélisation.

Le tableau ci-dessous résume la nature de chaque fichier dans le projet.

| Fichier | Type de données | Granularité | Rôle dans le projet |
|---|---|---|---|
| [crop_yield.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/simulation/crop_yield.csv) | Données simulées de rendement par culture avec variables agronomiques associées. Présenté dans le brief comme un jeu de données de rendement historique, mais utilisé ici surtout comme jeu d'analyse amont très pédagogique. | Observation individuelle sans clé `area + year` exploitable | Analyse exploratoire, nettoyage, ACP, lecture métier des facteurs associés au rendement |
| [yield.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/historique/yield.csv) | Données historiques annuelles de rendement | `area + crop + year` | Table de base du dataset consolidé et source de la cible de modélisation |
| [rainfall.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/historique/rainfall.csv) | Données historiques climatiques annuelles de pluie | `area + year` | Enrichissement climatique du dataset consolidé |
| [temp.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/historique/temp.csv) | Données historiques climatiques annuelles de température | `area + year` après agrégation des doublons | Enrichissement climatique du dataset consolidé |
| [pesticides.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/historique/pesticides.csv) | Données historiques annuelles d'intrants | `area + year` | Enrichissement agronomique du dataset consolidé |
| [yield_df.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/historique/yield_df.csv) | Données historiques annuelles déjà enrichies | `area + crop + year` | Fichier d'audit et de validation du scénario de fusion, pas table de base |
| [dataset_consolide.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/dataset_consolide.csv) | Données consolidées produites par le projet | `area + crop + year` | Source de vérité pour la modélisation et l'API |

Les chemins de ces fichiers sont centralisés dans [project_paths.yaml](/Users/steph/Code/Python/Jupyter/OCR_Projet12/config/project_paths.yaml). Les notebooks et scripts du projet doivent s'appuyer sur cette configuration plutôt que sur des chemins codés en dur.

## Générerer le rapport

```bash
jupyter nbconvert rapport.ipynb --to pdf --no-input
```

## ACP Exploratoire

### Rôle des trois éléments

- `preparation.ipynb` est la source de vérité pour l'ACP exploratoire sur `data/simulation/crop_yield.csv`.
- `preparation.ipynb` écrit les tableaux et figures dans `artifacts/pca/`.
- `rapport.ipynb` ne recalcule pas l'ACP : il relit uniquement les artefacts présents dans `artifacts/pca/`.
- `scripts/acp.py` est un raccourci headless pour régénérer ces mêmes artefacts sans relancer tout `preparation.ipynb`.

### Quand utiliser `scripts/acp.py`

Utiliser ce script si :

- les fichiers de `artifacts/pca/` sont absents ;
- il y a eu une modification dans la partie ACP dans `preparation.ipynb` et il faut regénérer rapidement les sorties ;
- il faut mettre à jour `rapport.ipynb` sans relancer toute la préparation des données.

Commande :

```bash
python3 scripts/acp.py
```

### Flux recommandé

Flux normal :

1. exécuter `preparation.ipynb` ;
2. vérifier que `artifacts/pca/` a bien été regénéré ;
3. exécuter `rapport.ipynb` ou l'exporter en PDF.

Flux rapide quand seule l'ACP du rapport doit être rafraîchie :

1. lancer `python3 scripts/acp.py` ;
2. ouvrir ou réexécuter `rapport.ipynb`.

## MLflow

MLflow est stocké dans une base SQLite :

- tracking DB : `artifacts/mlflow.db`
- artefacts MLflow : `artifacts/mlruns/` ; chaque run candidat contient aussi un artefact `model/` avec le pipeline entraîné
- modèles sauvegardés : `artifacts/models/`
- tableau de comparaison : `artifacts/model_comparison.csv`

### Comment lancer l'interface

L'interface MLflow doit être lancée avec le script du projet, afin d'utiliser le bon backend store :

```bash
python3 mlflow/mlflow.py
```

Par défaut, l'UI sera disponible sur `http://127.0.0.1:5000`.

Important :

- il n'est pas nécessaire de laisser l'interface ouverte pendant l'exécution du notebook ;
- `modelisation.ipynb` écrit directement dans `artifacts/mlflow.db` ;
- les artefacts des runs sont stockés dans `artifacts/mlruns/` ;
- l'UI peut donc être ouverte avant ou après l'exécution du notebook.
- `mlflow/mlflow.py` lance maintenant `python -m mlflow` avec l'interpréteur actif, ce qui évite d'utiliser par erreur un `mlflow` système d'une autre version.
- `mlflow/mlflow.py` lance `mlflow server` et fixe explicitement `artifacts/mlruns/` comme racine d'artefacts.
- si tu vois une erreur Alembic du type `Can't locate revision identified by ...`, la cause probable est un décalage de version entre le MLflow qui a créé `artifacts/mlflow.db` et celui qui essaie de l'ouvrir.

### Comment repartir d'un tracking propre

Pour repartir d'un historique MLflow vide, arrêter d'abord l'interface MLflow et les notebooks en cours, puis supprimer la base locale :

```bash
rm -f artifacts/mlflow.db
```

Au prochain lancement de `modelisation.ipynb` ou de `python3 mlflow/mlflow.py`, la base sera recréée automatiquement.

Si tu veux aussi nettoyer les anciens artefacts du backend fichier abandonné, tu peux supprimer en plus :

```bash
rm -rf artifacts/mlruns
```

Si un ancien dossier racine `mlruns/` subsiste encore vide après migration, tu peux aussi le supprimer :

```bash
rm -rf mlruns
```

### Mémo git flow

git flow release finish "nom"

- premier vim -> merge sur main (:wq)
- deuxième vim -> tag -> i -> 1.x -> Échap
- deuxième vim -> merge sur develop (:wq)
