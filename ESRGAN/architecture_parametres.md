# Architecture et parametres - ESRGAN

## Objectif

ESRGAN est un modele de super-resolution supervise base sur un generateur RRDB et un discriminateur adversarial. Il apprend a produire une carte CERRA realiste a partir d'une entree ERA5 interpolee.

## Architecture du generateur

- Classe: `ESRGenerator`.
- Entree: `1` canal.
- Couche initiale: Conv2D `1 -> channels`, noyau `3x3`.
- Corps: empilement de blocs `RRDB`.
- Chaque `RRDB` contient trois `DenseResidualBlock`.
- Chaque `DenseResidualBlock` utilise:
  - croissance dense `growth=32`.
  - cinq convolutions `3x3`.
  - activation `LeakyReLU(0.2)`.
  - facteur residuel `beta=0.2`.
- Sortie: Conv2D finale vers `1` canal.

## Architecture du discriminateur

- Entree: carte CERRA ou carte generee.
- Plusieurs convolutions `3x3` avec `LeakyReLU(0.2)`.
- Sous-echantillonnage par convolutions avec stride `2`.
- BatchNorm sur les couches intermediaires.
- `AdaptiveAvgPool2d(1)`.
- MLP final vers un logit reel/faux.

## Parametres principaux

- Epochs: `20`.
- Batch size: `4`.
- Train ratio: `0.8`.
- Patch size: `64`.
- `channels`: `32`.
- `rrdb_blocks`: `2`.
- Optimiseur generateur: `AdamW`, learning rate `1e-4`, betas `(0.9, 0.99)`.
- Optimiseur discriminateur: `AdamW`, learning rate `1e-4`, betas `(0.9, 0.99)`.
- Loss generateur:

```text
L_G = l1_weight * L1(fake, real) + adv_weight * BCE(D(fake), real)
```

- `l1_weight`: `1.0`.
- `adv_weight`: `1e-3`.
- AMP active sur GPU sauf avec `--no-amp`.

## Fichiers sauvegardes

- Checkpoint: `checkpoints/esrgan_generator_<variable>.pth`.
- Courbe de loss: `checkpoints/esrgan_loss_<variable>.png`.
- Parametres sauvegardes:
  - `generator_state`
  - `x_mu`, `x_std`
  - `y_mu`, `y_std`
  - `train_ratio`
  - `channels`
  - `rrdb_blocks`
  - `patch_size`

## Prediction

Le script `predict.py` charge uniquement le generateur, applique le modele sur ERA5 normalise, ajuste la taille si necessaire, puis denormalise la prediction.
