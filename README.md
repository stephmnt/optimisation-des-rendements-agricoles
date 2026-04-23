# Optimisation des rendements agricoles

[![Deploy Hugging Face Space](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml/badge.svg)](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml)

## Application

Le dépôt embarque maintenant une architecture unique pour la démo et le déploiement :

- Space : `stephmnt/rendement_agricole`
- API FastAPI : [main.py](/Users/steph/Code/Python/Jupyter/OCR_Projet12/main.py)
- interface Streamlit : [streamlit_app.py](/Users/steph/Code/Python/Jupyter/OCR_Projet12/streamlit/src/streamlit_app.py)
- dépendances UI/runtime : [streamlit/requirements.txt](/Users/steph/Code/Python/Jupyter/OCR_Projet12/streamlit/requirements.txt)
- environnement projet complet : [requirements.txt](/Users/steph/Code/Python/Jupyter/OCR_Projet12/requirements.txt)

Le conteneur Docker lance :

- FastAPI en interne sur `127.0.0.1:8000` ;
- Streamlit sur `8501` ;
- Streamlit qui interroge FastAPI au lieu de recalculer un modèle côté front.

Le démarrage du conteneur est défini directement dans le [Dockerfile](/Users/steph/Code/Python/Jupyter/OCR_Projet12/Dockerfile), et la construction du payload Hugging Face est définie directement dans [.github/workflows/deploy_hf_space.yml](/Users/steph/Code/Python/Jupyter/OCR_Projet12/.github/workflows/deploy_hf_space.yml).

### Lancer en local

```bash
pip install -r streamlit/requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Dans un second terminal :

```bash
API_BASE_URL=http://127.0.0.1:8000 streamlit run streamlit/src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

### Tester en Docker local

Le [Dockerfile](/Users/steph/Code/Python/Jupyter/OCR_Projet12/Dockerfile) à la racine sert au runtime complet FastAPI + Streamlit :

```bash
docker build --no-cache -t rendement-agricole-space .
docker run --rm -p 8501:8501 rendement-agricole-space
```

Puis ouvrir `http://localhost:8501`.

### Déployer sur Hugging Face

Le workflow [deploy_hf_space.yml](/Users/steph/Code/Python/Jupyter/OCR_Projet12/.github/workflows/deploy_hf_space.yml) :

1. installe les dépendances applicatives ;
2. exécute `pytest` sur l'API et le client Streamlit ;
3. assemble un payload Docker minimal dans `.hf_space_build/` ;
4. construit l'image Docker du Space ;
5. synchronise ce payload vers `stephmnt/rendement_agricole`.

Le dossier [notebooks](/Users/steph/Code/Python/Jupyter/OCR_Projet12/notebooks) reste dans le dépôt GitHub, mais il n'est pas recopié dans le payload envoyé sur Hugging Face.

Le Space Hugging Face utilise le [Dockerfile](/Users/steph/Code/Python/Jupyter/OCR_Projet12/Dockerfile) racine, recopié par le workflow dans `.hf_space_build/`.

### Secrets et Variables GitHub Actions

Le workflow utilise :

- un seul secret GitHub Actions à créer : `HF_TOKEN`
- aucune variable GitHub Actions personnalisée de type `vars.*`
- aucun secret d'environnement GitHub distinct

Configuration attendue pour `HF_TOKEN` :

- type : Hugging Face User Access Token
- permission minimale : écriture sur le Space `stephmnt/rendement_agricole`
- emplacement GitHub : `Settings > Secrets and variables > Actions > New repository secret`

Le workflow s'appuie aussi sur les contextes GitHub natifs suivants, qui ne nécessitent aucune configuration manuelle :

- `${{ github.repository }}`
- `${{ github.ref }}`

Déclenchement :

- push sur `main`
- lancement manuel via `workflow_dispatch`

Important :

- `HF_TOKEN` n'est requis que pour le job de déploiement ;
- les jobs `test` et `build` n'utilisent pas de secret ;
- `API_BASE_URL`, `API_HOST`, `API_PORT`, `STREAMLIT_HOST` et `STREAMLIT_PORT` sont des variables d'exécution du conteneur Docker, pas des variables GitHub Actions.

### Audit des Requirements

L'audit des dépendances explicites a été vérifié avant push :

- [streamlit/requirements.txt](/Users/steph/Code/Python/Jupyter/OCR_Projet12/streamlit/requirements.txt) contient les dépendances réellement nécessaires au runtime Docker/Hugging Face et à la CI de tests ;
- `httpx` y est conservé volontairement, car `fastapi.testclient.TestClient` en dépend pour la suite `pytest` ;
- [requirements.txt](/Users/steph/Code/Python/Jupyter/OCR_Projet12/requirements.txt) reste l'environnement complet du projet : notebooks, MLflow, scripts d'analyse, API et Streamlit ;
- `xgboost` y est conservé volontairement, car il est encore utilisé dans les notebooks d'expérimentation du projet ;
- aucune dépendance clairement obsolète n'a été conservée par erreur après le refactor FastAPI + Streamlit.

## Jeux de données

Le brief de [mission.md](/Users/steph/Code/Python/Jupyter/OCR_Projet12/ressources/mission.md) distingue deux grands ensembles :

- `Agriculture CropYield Dataset` : données de rendement utilisées pour l'analyse des facteurs clés ;
- `CropYield Prediction Dataset` : données agronomiques et climatiques annuelles utilisées pour valider cette analyse et construire la base de modélisation.

Le tableau ci-dessous résume la nature de chaque fichier dans le projet.

| Fichier | Type de données | Granularité | Rôle dans le projet |
|---|---|---|---|
| [crop_yield.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/crop_yield.csv) | Données simulées de rendement par culture avec variables agronomiques associées. Présenté dans le brief comme un jeu de données de rendement historique, mais utilisé ici surtout comme jeu d'analyse amont très pédagogique. | Observation individuelle sans clé `area + year` exploitable | Analyse exploratoire, nettoyage, ACP, lecture métier des facteurs associés au rendement |
| [yield.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/Crop%20Yield%20Prediction%20Dataset/yield.csv) | Données historiques annuelles de rendement | `area + crop + year` | Table de base du dataset consolidé et source de la cible de modélisation |
| [rainfall.csv](</Users/steph/Code/Python/Jupyter/OCR_Projet12/data/Crop Yield Prediction Dataset/rainfall.csv>) | Données historiques climatiques annuelles de pluie | `area + year` | Enrichissement climatique du dataset consolidé |
| [temp.csv](</Users/steph/Code/Python/Jupyter/OCR_Projet12/data/Crop Yield Prediction Dataset/temp.csv>) | Données historiques climatiques annuelles de température | `area + year` après agrégation des doublons | Enrichissement climatique du dataset consolidé |
| [pesticides.csv](</Users/steph/Code/Python/Jupyter/OCR_Projet12/data/Crop Yield Prediction Dataset/pesticides.csv>) | Données historiques annuelles d'intrants | `area + year` | Enrichissement agronomique du dataset consolidé |
| [yield_df.csv](</Users/steph/Code/Python/Jupyter/OCR_Projet12/data/Crop Yield Prediction Dataset/yield_df.csv>) | Données historiques annuelles déjà enrichies | `area + crop + year` | Fichier d'audit et de validation du scénario de fusion, pas table de base |
| [dataset_consolide.csv](/Users/steph/Code/Python/Jupyter/OCR_Projet12/data/dataset_consolide.csv) | Données consolidées produites par le projet | `area + crop + year` | Source de vérité pour la modélisation et l'API |

Les chemins de ces fichiers sont centralisés dans [project_paths.yaml](/Users/steph/Code/Python/Jupyter/OCR_Projet12/config/project_paths.yaml). Les notebooks et scripts du projet doivent s'appuyer sur cette configuration plutôt que sur des chemins codés en dur.

## Générer le rapport

```bash
jupyter nbconvert notebooks/rapport.ipynb --to pdf --no-input
```

## ACP Exploratoire

### Rôle des trois éléments

- `notebooks/preparation.ipynb` est la source de vérité pour l'ACP exploratoire sur `data/crop_yield.csv`.
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

### Mémo git flow

git flow release finish "nom"

- premier vim -> merge sur main (:wq)
- deuxième vim -> tag -> i -> 1.x -> Échap
- deuxième vim -> merge sur develop (:wq)
