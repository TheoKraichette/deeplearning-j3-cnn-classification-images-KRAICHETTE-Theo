# CNN — Pneumonie bactérienne vs virale (radiographie thoracique)

Classification binaire d'images en trois itérations, sur le même problème, pour mesurer l'apport de chaque technique de régularisation et du transfer learning.

| Itération | Approche | Régularisation |
|-----------|----------|----------------|
| TP1 | CNN from scratch | aucune (overfitting attendu) |
| TP2 | même architecture + data augmentation + Dropout | augmentation + Dropout |
| TP3 | MobileNetV2 pré-entraîné (ImageNet) + tête custom | freezing puis fine-tuning |

Modèle éducatif : il ne constitue en aucun cas un outil de diagnostic médical.

## Dataset

Radiographies de pneumonie labellisées bactérienne / virale (~3 900 images), issues du dataset Kaggle `paultimothymooney/chest-xray-pneumonia`. On écarte les clichés `normal` et on ne garde que les cas pathologiques : distinguer l'origine de l'infection est une tâche fine bien plus difficile que « normal vs pneumonie » (les deux motifs radiologiques se recouvrent largement). Le type d'infection n'est pas dans un dossier mais encodé dans le nom de fichier (`personXX_bacteria_YY.jpeg`), sur lequel le tri s'appuie.

Le notebook nettoie l'archive (dossiers `__MACOSX`, copie imbriquée) puis organise en `train` / `val` / `test`. Le `test` officiel (patient-disjoint) est conservé pour l'évaluation finale ; le `val` officiel ne faisant que 16 images, `train` et `val` sont refaits par un split 80/20 (`seed=42`) sur le reste.

Les classes sont déséquilibrées (~1,9:1 en faveur du bactérien). Le déséquilibre est traité avec `class_weight` à l'entraînement (pas en jetant des données), et l'analyse regarde des métriques adaptées, pas seulement l'accuracy brute.

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
