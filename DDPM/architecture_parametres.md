# Architecture et parametres - DDPM Cosine

## Objectif

DDPM est un modele de diffusion conditionnel. Il predit le bruit ajoute a une cible CERRA bruitee, en utilisant ERA5 interpole comme condition.

## Architecture

- Classe: `CosineDDPM`.
- Type: U-Net convolutionnel leger.
- Entrees:
  - `cond`: ERA5 interpole sur la grille CERRA.
  - `noisy_target`: cible CERRA bruitee.
- Les deux entrees sont concatenees, donc `2` canaux en entree.
- Encodeur:
  - `UNetBlock(2, channels)`.
  - MaxPool.
  - `UNetBlock(channels, channels*2)`.
- Bloc central:
  - `UNetBlock(channels*2, channels*2)`.
- Decodeur:
  - Upsampling bilineaire.
  - concatenation avec les features encodeur.
  - `UNetBlock(channels*4, channels)`.
  - Upsampling final.
  - Conv2D de sortie vers `1` canal.

## Parametres principaux

- `channels`: `64` par defaut.
- Nombre d'etapes de diffusion: `20`.
- Schedule: cosinus.
- `COSINE_S`: `0.008`.
- Epochs: `20`.
- Batch size: `4`.
- Train ratio: `0.8`.
- Optimiseur: `AdamW`.
- Learning rate: `1e-4`.
- Loss: `MSELoss` entre le bruit predit et le bruit reel.

## Fichiers sauvegardes

- Checkpoint: `checkpoints/ddpm_cosine_best_<variable>.pth`.
- Courbe de loss: `checkpoints/ddpm_loss_<variable>.png`.
- Parametres sauvegardes:
  - `model_state`
  - `x_mu`, `x_std`
  - `y_mu`, `y_std`
  - `train_ratio`
  - `steps`
  - `cosine_s`

## Prediction

La prediction part d'un bruit aleatoire sur la grille CERRA et applique les etapes de denoising conditionnees par ERA5. Dans ce projet, un filtre bilateral est applique apres prediction pour reduire le bruit spatial.
