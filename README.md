# Optimisation des rendements agricoles

[![Déploiement](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml/badge.svg)](https://github.com/stephmnt/optimisation-des-rendements-agricoles/actions/workflows/deploy_hf_space.yml)
[![GitHub Release Date](https://img.shields.io/github/release-date/stephmnt/optimisation-des-rendements-agricoles?display_date=published_at&style=flat-square)](https://github.com/stephmnt/optimisation-des-rendements-agricoles/releases)
[![project_license](https://img.shields.io/github/license/stephmnt/optimisation-des-rendements-agricoles.svg)](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/LICENSE)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-stephanemanet-0A66C2?logo=linkedin&logoColor=white)](https://linkedin.com/in/stephanemanet)

## Application

Cette application aide à explorer le rendement agricole de deux manières :

- estimer le rendement probable d'une culture pour un pays et des conditions de parcelle données ;
- comparer plusieurs cultures pour identifier celles qui semblent les plus prometteuses dans un contexte donné.

L'interface publique est disponible sur Hugging Face :

- Space : `stephmnt/rendement_agricole`

Le projet repose sur trois briques principales :

- une API FastAPI pour charger les modèles et calculer les prédictions : [main.py](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/main.py) ;
- une interface Streamlit orientée utilisateur final : [streamlit_app.py](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/streamlit/src/streamlit_app.py) ;
- un ensemble d'artefacts de modèles et de données préparés dans `artifacts/` et `data/`.

En pratique :

- FastAPI tourne en interne sur `127.0.0.1:8000` ;
- Streamlit est exposé sur `8501` ;
- l'interface Streamlit appelle l'API pour obtenir les résultats, sans recalculer les modèles côté front.

Le runtime applicatif est défini dans le [Dockerfile](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/Dockerfile), et le déploiement du Space est orchestré par [.github/workflows/deploy_hf_space.yml](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/.github/workflows/deploy_hf_space.yml).

### Ce que voit l'utilisateur

L'application propose deux parcours simples :

1. `Prédiction`
   Estimer le rendement d'une culture en fonction d'un pays et des conditions de parcelle saisies.
2. `Recommandation`
   Comparer plusieurs cultures et afficher un classement lisible pour aider à la décision.

Les libellés ont été reformulés pour un public non technique :

- les types de sol et les conditions météo sont traduits ;
- l'interface évite le jargon interne du projet ;
- les résultats mettent en avant le rendement estimé et l'impact des conditions de parcelle.

### Lancer en local

Le plus simple est de lancer d'abord l'API, puis l'interface Streamlit.

```bash
pip install -r streamlit/requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

Dans un second terminal :

```bash
API_V2_BASE_URL=http://127.0.0.1:8000 streamlit run streamlit/src/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Puis ouvrir `http://localhost:8501`.

### Tester en Docker local

Le [Dockerfile](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/Dockerfile) à la racine permet de lancer le runtime complet FastAPI + Streamlit :

```bash
docker build --no-cache -t rendement-agricole-space .
docker run --rm -p 8501:8501 rendement-agricole-space
```

Puis ouvrir `http://localhost:8501`.

### Régénérer les artefacts du projet

Pour reconstruire les artefacts de données et de modèles sans relancer chaque notebook à la main, utiliser les points d'entrée disponibles.

Étapes disponibles :

* `python3 scripts/run_preparation.py`
* `python3 scripts/train_historical_model.py`
* `python3 scripts/train_simulation_model.py --force-retrain`
* `python3 scripts/validate_runtime.py`
* `python3 scripts/run_full_pipeline.py`

Exemple pragmatique pour reconstruire les artefacts applicatifs avec le pipeline local actuel :

```bash
python3 scripts/run_full_pipeline.py
```

Repère rapide :

* `scripts/train_historical_model.py` régénère l'artefact historique à partir de [scripts/experience_1.py](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/scripts/experience_1.py) ;
* `scripts/train_simulation_model.py` régénère l'artefact du modèle local ;
* `scripts/promote_registered_model.py` réexporte ensuite les deux modèles runtime depuis MLflow vers `artifacts/models/` ;
* `scripts/run_preparation.py` exécute encore [notebooks/preparation.ipynb](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/notebooks/preparation.ipynb) en mode headless pour préparer les données ;
* `scripts/run_full_pipeline.py` orchestre la chaîne officielle : préparation, entraînement, enregistrement MLflow, promotion runtime et validation.
* par défaut, `scripts/run_full_pipeline.py` promeut la dernière version MLflow disponible pour chaque modèle runtime ; `--historical-version` et `--simulation-version` permettent de figer une version précise si besoin.
* `notebooks/experience_2.ipynb` et `notebooks/experience_3.ipynb` restent conservés dans le dépôt comme archives ou supports de vérification manuelle, mais ils ne font plus partie du pipeline standard.

### Déployer sur Hugging Face

Le workflow [deploy_hf_space.yml](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/.github/workflows/deploy_hf_space.yml) :

1. installer les dépendances applicatives ;
2. exécuter `pytest` sur l'API et le client Streamlit ;
3. assembler un payload Docker minimal dans `.hf_space_build/`, avec `main.py`, `scripts/`, les artefacts des modèles historique et local, ainsi que le dataset d'entrée de `experience_1` ;
4. construire l'image Docker du Space ;
5. synchroniser ce payload vers `stephmnt/rendement_agricole`.

Le dossier [notebooks](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/notebooks) reste dans le dépôt GitHub, mais il n'est pas recopié dans le payload envoyé sur Hugging Face.

Le Space Hugging Face utilise le [Dockerfile](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/Dockerfile) racine, recopié par le workflow dans `.hf_space_build/`.

Le workflow [train_full_pipeline.yml](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/.github/workflows/train_full_pipeline.yml) permet de relancer séparément le pipeline d'entraînement sur GitHub Actions :

1. installation de l'environnement complet du projet ;
2. exécution de `scripts/run_full_pipeline.py` ;
3. regression tests applicatifs ;
4. publication des artefacts régénérés comme artefacts GitHub téléchargeables.

Ce workflow reste séparé du workflow de déploiement :

* `train_full_pipeline.yml` gère préparation, entraînement, enregistrement MLflow, promotion runtime et validation ;
* `deploy_hf_space.yml` gère uniquement les tests applicatifs, le build Docker et le déploiement du Space.

### Secrets et Variables GitHub Actions

Secret GitHub Actions requis : `HF_TOKEN`

### Environnements Python

* [streamlit/requirements.txt](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/streamlit/requirements.txt) contient les dépendances réellement nécessaires au runtime Docker/Hugging Face et à la CI de tests ;
* `httpx` y est conservé volontairement, car `fastapi.testclient.TestClient` en dépend pour la suite `pytest` ;
* [requirements.txt](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/requirements.txt) correspond à l'environnement complet du projet : notebooks, MLflow, scripts d'analyse, API et Streamlit ;
* `xgboost` y est conservé volontairement, car il est encore utilisé dans `scripts/experience_1.py` et dans les notebooks d'expérimentation du projet ;
* aucune dépendance clairement obsolète n'a été conservée par erreur après le refactor FastAPI + Streamlit.

## Jeux de données

* `Agriculture CropYield Dataset` : données de rendement utilisées pour l'analyse des facteurs clés ;
* `CropYield Prediction Dataset` : données agronomiques et climatiques annuelles utilisées pour valider cette analyse et construire la base de modélisation.

Le tableau ci-dessous résume la nature de chaque fichier dans le projet.

| Fichier                                                                                                                         | Type de données                                                                                                                                                                                                               | Granularité                                                 | Rôle dans le projet                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [crop_yield.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/simulation/crop_yield.csv)    | Données simulées de rendement par culture avec variables agronomiques associées. Présenté dans le brief comme un jeu de données de rendement historique, mais utilisé ici surtout comme jeu d'analyse amont très pédagogique. | Observation individuelle sans clé `area + year` exploitable | Analyse exploratoire, nettoyage, ACP, lecture métier des facteurs associés au rendement |
| [yield.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/historique/yield.csv)              | Données historiques annuelles de rendement                                                                                                                                                                                    | `area + crop + year`                                        | Table de base du dataset consolidé et source de la cible de modélisation                |
| [rainfall.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/historique/rainfall.csv)        | Données historiques climatiques annuelles de pluie                                                                                                                                                                            | `area + year`                                               | Enrichissement climatique du dataset consolidé                                          |
| [temp.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/historique/temp.csv)                | Données historiques climatiques annuelles de température                                                                                                                                                                      | `area + year` après agrégation des doublons                 | Enrichissement climatique du dataset consolidé                                          |
| [pesticides.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/historique/pesticides.csv)    | Données historiques annuelles d'intrants                                                                                                                                                                                      | `area + year`                                               | Enrichissement agronomique du dataset consolidé                                         |
| [yield_df.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/historique/yield_df.csv)        | Données historiques annuelles déjà enrichies                                                                                                                                                                                  | `area + crop + year`                                        | Fichier d'audit et de validation du scénario de fusion, pas table de base               |
| [dataset_consolide.csv](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/data/dataset_consolide.csv) | Données consolidées produites par le projet                                                                                                                                                                                   | `area + crop + year`                                        | Source de vérité pour la modélisation et l'API                                          |

Les chemins de ces fichiers sont centralisés dans [project_paths.yaml](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/config/project_paths.yaml). Les notebooks et scripts du projet doivent s'appuyer sur cette configuration plutôt que sur des chemins codés en dur.

## Générer le rapport

```bash
jupyter nbconvert notebooks/rapport.ipynb --to pdf --no-input
```

## ACP Exploratoire

### Rôle des trois éléments

* `notebooks/preparation.ipynb` est la source de vérité pour l'ACP exploratoire sur `data/crop_yield.csv`.
* `notebooks/preparation.ipynb` écrit les tableaux et figures dans `artifacts/pca/`.
* `notebooks/rapport.ipynb` ne recalcule pas l'ACP : il relit uniquement les artefacts présents dans `artifacts/pca/`.
* `scripts/acp.py` est un raccourci headless pour régénérer ces mêmes artefacts sans relancer tout `notebooks/preparation.ipynb`.

### Utiliser `scripts/acp.py`

Utiliser ce script si :

* les fichiers de `artifacts/pca/` sont absents ;
* la partie ACP de `notebooks/preparation.ipynb` a été modifiée et les sorties doivent être régénérées rapidement ;
* `notebooks/rapport.ipynb` doit être mis à jour sans relancer toute la préparation des données.

Commande :

```bash
python3 scripts/acp.py
```

### Flux recommandé

Flux normal :

1. exécuter `notebooks/preparation.ipynb` ;
2. vérifier que `artifacts/pca/` a bien été régénéré ;
3. exécuter `notebooks/rapport.ipynb` ou l'exporter en PDF.

Flux rapide quand seule l'ACP du rapport doit être rafraîchie :

1. lancer `python3 scripts/acp.py` ;
2. ouvrir ou réexécuter `notebooks/rapport.ipynb`.

## MLflow

MLflow est stocké dans une base SQLite :

* tracking DB : `artifacts/mlflow.db`
* artefacts MLflow : `artifacts/mlruns/` ; chaque run candidat contient aussi un artefact `model/` avec le pipeline entraîné
* modèles sauvegardés : `artifacts/models/`
* tableau de comparaison historique : `artifacts/model_comparison.csv`
* expérience d'orchestration du pipeline complet : `run_full_pipeline`
* registered models runtime attendus :

  * `p1_historical_pipeline`
  * `p23_simulation_pipeline`

### Lancer l'interface

L'interface MLflow doit être lancée avec le script du projet, afin d'utiliser le bon backend store :

```bash
python3 mlflow/mlflow.py
```

Par défaut, l'UI sera disponible sur `http://127.0.0.1:5000`.

Important :

* lancer `mlflow/mlflow.py` avec le bouton lecture de VS Code utilise la même base officielle : `artifacts/mlflow.db` ;
* si `.venv/bin/python` existe, `mlflow/mlflow.py` se relance avec cet interpréteur pour éviter les écarts de version MLflow ;
* `scripts/run_full_pipeline.py`, `scripts/experience_1.py`, `scripts/train_simulation_model.py` et `scripts/promote_registered_model.py` partagent le même tracking URI par défaut ;
* chaque exécution de `scripts/run_full_pipeline.py` ajoute un run dans l'expérience MLflow `run_full_pipeline`, en plus des runs détaillés dans `experience_1` et `simulation_runtime` ;
* il n'est pas nécessaire de laisser l'interface ouverte pendant l'exécution des notebooks ou des scripts d'entraînement ;
* la chaîne officielle écrit dans `artifacts/mlflow.db` via `scripts/experience_1.py` et `scripts/train_simulation_model.py` ;
* `notebooks/experience_2.ipynb` reste consultable dans le dépôt, mais il n'appartient plus au chemin de régénération standard ;
* les artefacts des runs sont stockés dans `artifacts/mlruns/` ;
* l'UI peut donc être ouverte avant ou après l'exécution de ces étapes.
* la piste historique `notebooks/modelisation.ipynb` est abandonnée ; le dépôt conserve seulement ses artefacts utiles pour archive.
* `mlflow/mlflow.py` lance maintenant `python -m mlflow` avec l'interpréteur actif, ce qui évite d'utiliser par erreur un `mlflow` système d'une autre version.
* `mlflow/mlflow.py` lance `mlflow server` et fixe explicitement `artifacts/mlruns/` comme racine d'artefacts.
* en cas d'erreur Alembic du type `Can't locate revision identified by ...`, la cause probable est un décalage de version entre le MLflow qui a créé `artifacts/mlflow.db` et celui qui essaie de l'ouvrir.

### Promouvoir les modèles runtime

Le script [scripts/promote_registered_model.py](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/scripts/promote_registered_model.py) exporte les deux registered models réellement utilisés par l'API :

* historique :

  * registered model attendu : `p1_historical_pipeline`
  * artefacts runtime exportés :

    * `artifacts/models/p1_historical_pipeline.joblib`
    * `artifacts/models/p1_historical_metadata.json`
* local / simulation :

  * registered model attendu : `p23_simulation_pipeline`
  * artefacts runtime exportés :

    * `artifacts/models/p23_simulation_pipeline.joblib`
    * `artifacts/models/p23_simulation_metadata.json`

Le script reste strict :

* s'il manque le registered model historique, il échoue ;
* s'il manque le registered model local, il échoue ;
* s'il y a plusieurs versions possibles sans choix explicite, il échoue et demande la version ;
* s'il ne peut pas recharger l'artefact exporté, il échoue aussi.

Exemples :

```bash
python3 scripts/promote_registered_model.py
python3 scripts/promote_registered_model.py --historical-version 2 --simulation-version 5
```

Les fichiers metadata exportés tracent notamment :

* le nom du registered model ;
* la version MLflow exportée ;
* le `run_id` source ;
* l'URI `models:/...` source ;
* la date d'export ;
* le rôle runtime du modèle : `historical` ou `simulation`.

Important :

* l'API finale [main.py](https://github.com/stephmnt/optimisation-des-rendements-agricoles/blob/main/main.py) ne charge pas un artefact générique de type `best_pipeline.joblib` ;
* elle charge explicitement `p1_historical_*` et `p23_simulation_*` ;
* la promotion MLflow sert donc à exporter ou revalider ces deux artefacts runtime, pas à introduire un troisième modèle central.

### Repartir d'un tracking propre

Pour repartir d'un historique MLflow vide, arrêter d'abord l'interface MLflow et les notebooks en cours, puis supprimer la base locale :

```bash
rm -f artifacts/mlflow.db
```

Au prochain lancement de `python3 scripts/experience_1.py`, de `python3 scripts/train_simulation_model.py` ou de `python3 scripts/run_full_pipeline.py`, la base sera recréée automatiquement.

Pour nettoyer aussi les anciens artefacts du backend fichier abandonné, supprimer en plus :

```bash
rm -rf artifacts/mlruns
```

Si un ancien dossier racine `mlruns/` subsiste encore vide après migration, le supprimer également :

```bash
rm -rf mlruns
```
