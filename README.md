# Optimisation des rendements agricoles

[![Deploy Hugging Face Space](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml/badge.svg)](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml)

## Démo Streamlit

Une démo Streamlit autonome est préparée dans [demo_streamlit](/Users/steph/Code/Python/Jupyter/OCR_Projet12/demo_streamlit) pour être déployée sur le Space Hugging Face :

- Space : `stephmnt/rendement_agricole`
- dossier source : [demo_streamlit](/Users/steph/Code/Python/Jupyter/OCR_Projet12/demo_streamlit)
- application Streamlit : [streamlit_app.py](/Users/steph/Code/Python/Jupyter/OCR_Projet12/demo_streamlit/src/streamlit_app.py)
- logique métier testable : [app_logic.py](/Users/steph/Code/Python/Jupyter/OCR_Projet12/demo_streamlit/src/app_logic.py)

L'application :

- recharge un modèle léger à partir de [dataset_consolide.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/dataset_consolide.csv) au démarrage ;
- propose une prédiction de rendement ;
- propose une recommandation de cultures.

### Lancer en local

```bash
pip install -r demo_streamlit/requirements.txt
streamlit run demo_streamlit/src/streamlit_app.py
```

### Déployer sur Hugging Face

Le workflow [deploy_hf_space.yml](/Users/steph/Code/Python/Jupyter/OCR_Projet12/.github/workflows/deploy_hf_space.yml) :

1. installe les dépendances de la démo ;
2. exécute `pytest` sur [test_demo_streamlit_logic.py](/Users/steph/Code/Python/Jupyter/OCR_Projet12/tests/test_demo_streamlit_logic.py) ;
3. assemble un payload minimal pour le Space ;
4. synchronise ce payload vers `stephmnt/rendement_agricole`.

Le dossier [notebooks](/Users/steph/Code/Python/Jupyter/OCR_Projet12/notebooks) reste dans le dépôt GitHub, mais il n'est pas recopié dans le payload envoyé sur Hugging Face.

Pré-requis GitHub :

- créer un secret `HF_TOKEN` avec un token Hugging Face en écriture sur le Space ;
- pousser sur `main`, ou lancer le workflow manuellement.

## Générerer le rapport

```bash
jupyter nbconvert notebooks/rapport.ipynb --to pdf --no-input
```

## ACP Exploratoire

### Rôle des trois éléments

- `notebooks/preparation.ipynb` est la source de vérité pour l'ACP exploratoire sur `data/simulation/crop_yield.csv`.
- `notebooks/preparation.ipynb` écrit les tableaux et figures dans `artifacts/pca/`.
- `notebooks/rapport.ipynb` ne recalcule pas l'ACP : il relit uniquement les artefacts présents dans `artifacts/pca/`.
- `scripts/acp.py` est un raccourci headless pour régénérer ces mêmes artefacts sans relancer tout `notebooks/preparation.ipynb`.

### Quand utiliser `scripts/acp.py`

Utiliser ce script si :

- les fichiers de `artifacts/pca/` sont absents ;
- tu as modifié la partie ACP dans `notebooks/preparation.ipynb` et tu veux regénérer rapidement les sorties ;
- tu veux mettre à jour `notebooks/rapport.ipynb` sans relancer toute la préparation des données.

Commande :

```bash
python3 scripts/acp.py
```

### Flux recommandé

Flux normal :

1. exécuter `notebooks/preparation.ipynb` ;
2. vérifier que `artifacts/pca/` a bien été regénéré ;
3. exécuter `notebooks/rapport.ipynb` ou l'exporter en PDF.

Flux rapide quand seule l'ACP du rapport doit être rafraîchie :

1. lancer `python3 scripts/acp.py` ;
2. ouvrir ou réexécuter `notebooks/rapport.ipynb`.

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
- `notebooks/modelisation.ipynb` écrit directement dans `artifacts/mlflow.db` ;
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

Au prochain lancement de `notebooks/modelisation.ipynb` ou de `python3 mlflow/mlflow.py`, la base sera recréée automatiquement.

Si tu veux aussi nettoyer les anciens artefacts du backend fichier abandonné, tu peux supprimer en plus :

```bash
rm -rf artifacts/mlruns
```

Si un ancien dossier racine `mlruns/` subsiste encore vide après migration, tu peux aussi le supprimer :

```bash
rm -rf mlruns
```
