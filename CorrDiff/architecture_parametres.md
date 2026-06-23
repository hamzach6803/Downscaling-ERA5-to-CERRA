# Architecture et parametres - CorrDiff

## Objectif

CorrDiff combine une baseline lineaire et un modele de diffusion par score. La regression lineaire produit une premiere estimation, puis le modele de score apprend le residu CERRA moins baseline.

## Architecture

- Baseline: regression lineaire pixel par pixel.
- Modele generatif: `ScoreUNet`.
- Entrees du U-Net:
  - ERA5 normalise haute resolution/interpole.
  - baseline lineaire.
  - residu bruite.
  - carte de sigma.
- Nombre de canaux d'entree: `4`.
- Blocs convolutionnels:
  - `ScoreBlock`: Conv2D, SiLU, Conv2D, SiLU.
  - encodeur a deux niveaux.
  - bloc central.
  - decodeur avec skip connection.
  - Conv2D finale vers `1` canal de score.

## Parametres principaux

- Epochs: `20`.
- Batch size: `4` par defaut, souvent `1` en low memory.
- Train ratio: `0.8`.
- Patch size: `64`.
- Channels: `32`.
- Optimiseur: `AdamW`.
- Learning rate: `1e-4`.
- Weight decay: `1e-5`.
- `sigma_min`: `0.01`.
- `sigma_max`: `1.0`.
- `lambda_consistency`: `0.0`.
- AMP: active sur GPU.

## Prediction

- Sample steps: `20`.
- Langevin steps: `1`.
- SNR: `0.15`.
- Prediction finale:

```text
y_hat = baseline + residual_sampled
```

La sortie est ensuite denormalisee. Dans les scripts de comparaison, un filtre bilateral est applique a CorrDiff pour lisser la carte.

## Fichiers sauvegardes

- Checkpoint: `checkpoints/corrdiff_score_<variable>.pth`.
- Courbe de loss: `checkpoints/corrdiff_loss_<variable>.png`.
- Prediction: `predictions/corrdiff_pred_<variable>.nc`.
