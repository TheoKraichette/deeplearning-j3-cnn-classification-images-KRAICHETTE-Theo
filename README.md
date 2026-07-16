# CNN — Radiographie thoracique : normal vs pneumonie

Classification binaire d'images en trois itérations, sur le même problème, pour mesurer l'apport de chaque technique de régularisation et du transfer learning.

| Itération | Approche | Régularisation |
|-----------|----------|----------------|
| TP1 | CNN from scratch | aucune (overfitting attendu) |
| TP2 | même architecture + data augmentation + Dropout | augmentation + Dropout |
| TP3 | MobileNetV2 pré-entraîné (ImageNet) + tête custom | freezing puis fine-tuning |

Modèle éducatif : il ne constitue en aucun cas un outil de diagnostic médical.

## Dataset

Radiographies thoraciques labellisées NORMAL / PNEUMONIA (~5 900 images), dataset Kaggle `paultimothymooney/chest-xray-pneumonia`.

Le notebook nettoie l'archive (dossiers `__MACOSX`, copie imbriquée) puis organise en `train` / `val` / `test`. Le `test` officiel (patient-disjoint) est conservé pour l'évaluation finale ; le `val` officiel ne faisant que 16 images, `train` et `val` sont refaits par un split 80/20 (`seed=42`) sur le reste.

Les classes sont déséquilibrées (~2,7:1 en faveur de pneumonie). Le déséquilibre est traité avec `class_weight` à l'entraînement (pas en jetant des données), et l'analyse regarde des métriques adaptées, pas seulement l'accuracy brute.

## Reproduire

Environnement défini dans `Dockerfile` / `docker-compose.yml` (TensorFlow CPU + Jupyter).

1. Créer `.env` d'après `.env.example` (identifiants Kaggle).
2. `docker compose up --build`, puis ouvrir Jupyter et exécuter `cnn_pneumonie.ipynb`.

## Livrables

- Le notebook avec les trois itérations (phases 1.x → 3.x)
- Les courbes loss/accuracy (matplotlib)
- Le fichier `.keras` du meilleur modèle
- Le tableau de comparaison des trois modèles
- Le fichier `.tflite` (export mobile)

## Résultats

À compléter après entraînement.

| Itération | val_accuracy | Params | Temps | Taille |
|-----------|-------------|--------|-------|--------|
| CNN scratch | – | – | – | – |
| CNN augmenté + Dropout | – | – | – | – |
| MobileNetV2 fine-tuning | – | – | – | – |
