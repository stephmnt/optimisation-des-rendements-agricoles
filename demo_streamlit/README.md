---
title: Rendement Agricole
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8501
tags:
- streamlit
- agriculture
pinned: false
short_description: Démo Streamlit de prédiction et recommandation de cultures
license: mit
---

# Rendement Agricole

Cette application Streamlit propose deux usages simples :

- prédire un rendement à l'hectare pour une culture donnée ;
- recommander les cultures les plus prometteuses pour un contexte donné.

L'application recharge un modèle léger à partir de `data/dataset_consolide.csv` au démarrage pour éviter de déployer un artefact lourd.

## Lancer en local

```bash
pip install -r demo_streamlit/requirements.txt
streamlit run demo_streamlit/src/streamlit_app.py
```

## Déploiement GitHub Actions

Le workflow `.github/workflows/deploy_hf_space.yml` :

- exécute `pytest` ;
- assemble un payload minimal pour le Space ;
- synchronise ce payload vers `stephmnt/rendement_agricole`.

Le dépôt GitHub doit contenir un secret `HF_TOKEN` avec un token Hugging Face disposant d'un droit d'écriture sur le Space.
