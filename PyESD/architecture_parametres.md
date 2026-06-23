# Architecture et parametres - PyESD

## Objectif

PyESD est utilise comme modele de reference simple pour le downscaling de `t2m`. Il apprend une regression lineaire pixel par pixel entre ERA5 interpole sur la grille CERRA et la reference CERRA.

## Architecture

- Type: regression lineaire locale.
- Entree: champ ERA5 interpole sur la grille CERRA.
- Sortie: champ CERRA reconstruit.
- Pour chaque pixel, le modele apprend deux parametres:
  - `slope`
  - `intercept`
- La prediction normalisee suit:

```text
y_hat = x * slope + intercept
```

La sortie est ensuite denormalisee avec `y_mu` et `y_std`.

## Parametres principaux

- Variable: definie par `utils.METEO_VARIABLE`, generalement `t2m`.
- Train ratio: `0.8`.
- Jeu de test: `0.2`.
- Normalisation: moyenne et ecart-type calcules sur le jeu d'entrainement.
- Entrainement: calcul streaming par chunks pour limiter la memoire.
- Optimiseur: aucun optimiseur iteratif, solution analytique par regression lineaire.

## Fichiers sauvegardes

- Checkpoint: `checkpoints/pyesd_linear_<variable>.npz`.
- Parametres sauvegardes:
  - `slope`
  - `intercept`
  - `x_mu`, `x_std`
  - `y_mu`, `y_std`
  - `train_ratio`
  - `variable`

## Prediction

Le script `predict.py` charge le checkpoint, normalise ERA5, applique la regression pixel par pixel, denormalise la sortie et sauvegarde la prediction NetCDF.
