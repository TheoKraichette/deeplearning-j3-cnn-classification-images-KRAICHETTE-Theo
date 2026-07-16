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

Le notebook nettoie l'archive (dossiers `__MACOSX`, copie imbriquée) puis organise en `train` / `val` / `test`. Le `test` officiel (patient-disjoint) est conservé pour l'évaluation finale ; le `val` officiel ne faisant que 16 images, `train` et `val` sont refaits par un split 80/20 (`seed=42`) sur le reste. Ce re-split est au niveau image : `val` n'est donc pas patient-disjoint et sert de signal de suivi et de sélection, pas de métrique finale — la mesure de généralisation honnête reste le `test` officiel.

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

Tâche volontairement difficile (les motifs bactérien et viral se recouvrent), d'où une progression réelle mais modeste en validation. L'intérêt est le rapport performance/coût : MobileNetV2 fait aussi bien que le CNN from scratch avec presque deux fois moins de paramètres et un modèle deux fois plus léger.

| Itération | val_accuracy | test (patient-disjoint) | Params | Temps | Taille |
|-----------|-------------|------------------------|--------|-------|--------|
| CNN scratch | 73,5 % | 85,9 % | 4 287 809 | 239 s | 49,1 Mo |
| CNN augmenté + Dropout | 74,7 % | 90,0 % | 4 287 809 | 628 s | 49,1 Mo |
| MobileNetV2 (tête + fine-tuning) | 76,0 % | 86,9 % | 2 422 081 | 1206 s | 23,4 Mo |

Le test set officiel (patient-disjoint), gardé intact jusqu'à l'évaluation finale, se révèle plus facile que le split de validation : les trois modèles y gagnent 11 à 15 points. Les écarts (86-90 %) restent dans la marge d'erreur d'un test set de 390 images.

### Exploration (phase 3.5)

Deux directions au-delà du TP3 : **3.5 A — MobileNetV2 + augmentation** et **3.5 B — DenseNet121**. En évaluant les **7 états** sur le test set :

| État | test |
|------|------|
| CNN scratch | 85,9 % |
| CNN augmenté | 90,0 % |
| MobileNet — tête (freeze) | 89,0 % |
| MobileNet — fine-tuning | 86,9 % |
| MobileNet+aug — tête (freeze) | 89,7 % |
| **MobileNet+aug — fine-tuning** | **90,0 %** |
| DenseNet121 — tête | 87,9 % |

Enseignement clé : **le fine-tuning n'est bénéfique que si le backbone est régularisé par l'augmentation.** Sur le MobileNet fidèle (Dropout seul), le fine-tuning *dégrade* le test (89,0 → 86,9 %, overfit sur ~3 100 images) ; sur le MobileNet+aug, le *même* fine-tuning l'*améliore* (89,7 → 90,0 %) — même backbone, même stratégie de freezing, seule l'augmentation change. Trois voies mènent à ~90 % (CNN augmenté, MobileNet+aug fine-tuné, et de peu les têtes gelées) : ce qui paie sur cette tâche fine-grained, c'est la **régularisation**, qu'on parte de zéro ou d'un backbone pré-entraîné.

> Les modèles n'ont pas de seed global sur l'initialisation des poids ni le Dropout (le split et l'ordre des données, eux, sont seedés avec `seed=42`) : les chiffres proviennent d'un run représentatif, une ré-exécution varie de ~1-2 points.

## Organisation du dépôt

- `cnn_pneumonie.ipynb` — le notebook complet (phases 1.x → 3.5 A/B)
- `models/` — les `.keras` des modèles (scratch, augmenté, MobileNetV2, + checkpoint tête de MobileNet+aug) et le `.tflite` (MobileNetV2)
- `docs/` — les figures `.png` (courbes, comparaisons, matrice de confusion)
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `.env.example` — environnement
