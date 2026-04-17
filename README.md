# Optimisation des rendements agricoles

## Générerer le rapport

```bash
jupyter nbconvert rapport.ipynb --to pdf --no-input
```

## ACP Exploratoire

### Rôle des trois éléments

- `preparation.ipynb` est la source de vérité pour l'ACP exploratoire sur `data/crop_yield.csv`.
- `preparation.ipynb` écrit les tableaux et figures dans `artifacts/pca/`.
- `rapport.ipynb` ne recalcule pas l'ACP : il relit uniquement les artefacts présents dans `artifacts/pca/`.
- `scripts/acp.py` est un raccourci headless pour régénérer ces mêmes artefacts sans relancer tout `preparation.ipynb`.

### Quand utiliser `scripts/acp.py`

Utiliser ce script si :

- les fichiers de `artifacts/pca/` sont absents ;
- tu as modifié la partie ACP dans `preparation.ipynb` et tu veux regénérer rapidement les sorties ;
- tu veux mettre à jour `rapport.ipynb` sans relancer toute la préparation des données.

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

### Où sont stockés les runs

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
