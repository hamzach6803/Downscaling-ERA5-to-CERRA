# Architecture et parametres - DeepSD

## Objectif

DeepSD est un reseau convolutionnel supervise pour transformer un champ ERA5 basse resolution en champ CERRA haute resolution.

## Architecture

- Type: reseau CNN avec upsampling bilineaire.
- Entree: tenseur `1 x H x W` ERA5 normalise.
- Sortie: tenseur `1 x H_hr x W_hr` CERRA normalise.
- Architecture:
  - Conv2D `1 -> channels`, noyau `3x3`, padding `1`.
  - ReLU.
  - Conv2D `channels -> channels`, noyau `3x3`.
  - ReLU.
  - Conv2D `channels -> channels`, noyau `3x3`.
  - ReLU.
  - Upsampling bilineaire avec facteur `scale`.
  - Conv2D `channels -> channels/2`.
  - ReLU.
  - Conv2D `channels/2 -> channels/2`.
  - ReLU.
  - Conv2D `channels/2 -> 1`.

## Parametres principaux

- `channels`: `96` par defaut dans `model.py`.
- `scale`: calcule automatiquement avec le rapport entre la grille CERRA et la grille ERA5.
- Epochs: `50`.
- Batch size: `4` par defaut.
- Train ratio: `0.8`.
- Optimiseur: `AdamW`.
- Learning rate: `1e-4`.
- Weight decay: `1e-5`.
- Loss: erreur supervisee entre prediction et CERRA normalise.

## Fichiers sauvegardes

- Checkpoint: `checkpoints/deepsd_best_<variable>.pth`.
- Contenu:
  - `model_state`
  - `scale`
  - `x_mu`, `x_std`
  - `y_mu`, `y_std`
  - `train_ratio`
  - `variable`

## Prediction

Le script `predict.py` charge le meilleur checkpoint, applique le CNN, ajuste si necessaire la taille finale par interpolation bilineaire, puis denormalise la prediction.
