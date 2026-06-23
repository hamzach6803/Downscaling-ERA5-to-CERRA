# Projet   - Downscaling ERA5 vers CERRA

Ce projet compare plusieurs approches de downscaling des donnees ERA5 vers la
grille CERRA: regression linéaire, CNN de super-resolution, GAN et diffusion
model.

## Structure du projet

```text
model/
  README.md
  requirements.txt
  configs/                 # configuration globale et fiches des modeles
  src/downscaling/          # code commun reutilisable
  PyESD/                    # modele PyESD + checkpoints originaux
  DeepSD/                   # modele DeepSD + checkpoints originaux
  DDPM/                     # modele DDPM + checkpoints originaux
  ESRGAN/                   # modele ESRGAN + checkpoints originaux
  CorrDiff/                 # modele CorrDiff + checkpoints originaux
  data/                     # donnees locales
  to_predict/               # fichier prepare pour la prediction rapide
  outputs/                  # resultats, copies organisees, figures, metriques
  mask_results*/            # analyses d'evenements extremes
  docs/                     # documentation et textes du rapport
  scripts/                  # futurs scripts d'automatisation
  notebooks/                # notebooks d'exploration
```

## Modeles disponibles

- `PyESD`: regression lineaire pixel par pixel.
- `DeepSD`: CNN de super-resolution.
- `DDPM`: diffusion conditionnelle.
- `ESRGAN`: GAN de super-resolution.
- `CorrDiff`: regression lineaire + diffusion sur les residus.

## Donnees attendues

Par defaut, les chemins sont definis dans `src/downscaling/utils.py` et
exposes par le wrapper `utils.py`:

```text
data/
  era5_2021_2025.nc
  cerra_2021_2025_t2m_wgs84.nc
```

## Workflow conseille

Preparer une date a predire:

```bash
python make_to_predict.py --date 2025-01-03
```

Entrainer:

```bash
python PyESD/train.py --chunk-size 4 --num-workers 0
python DeepSD/train.py --batch-size 1 --max-train-samples 128 --num-workers 0 --device cpu
python DDPM/train.py --batch-size 1 --max-train-samples 128 --num-workers 0 --device cpu
python ESRGAN/train.py --batch-size 1 --patch-size 32 --channels 16 --rrdb-blocks 1 --max-train-samples 128 --num-workers 0 --device cpu
python CorrDiff/train.py --batch-size 1 --patch-size 32 --channels 16 --max-train-samples 128 --num-workers 0 --device cpu
```

Predire:

```bash
python PyESD/predict.py
python DeepSD/predict.py --to-predict to_predict/to_predict_t2m.nc
python DDPM/predict.py --to-predict to_predict/to_predict_t2m.nc
python ESRGAN/predict.py --to-predict to_predict/to_predict_t2m.nc
python CorrDiff/predict.py --to-predict to_predict/to_predict_t2m.nc --sample-steps 5
```

Comparer les modeles:

```bash
python compare_all.py
python compare_selected_test_mean.py --reuse-cache --palette-mode uniform --device cpu
python results_extreme90.py --use-saved-predictions --palette-mode uniform
```

Les nouvelles sorties de comparaison vont dans:

```text
outputs/comparisons/
```

## Organisation des resultats

```text
outputs/
  checkpoints/      # copies organisees des checkpoints deja entraines
  predictions/      # copies organisees des predictions existantes
  comparisons/      # figures et CSV de comparaison
  masks/            # analyses d'evenements extremes
  figures/          # figures globales
  archives/         # fichiers zip ou exports
```

Important: ne supprime pas les dossiers `checkpoints/` dans les dossiers de
modeles. Ils sont la source utilisee par les scripts d'entrainement, prediction
et comparaison.

## Notes de compatibilite

- `utils.py` et `corrdiff_core.py` restent a la racine comme wrappers.
- Le vrai code commun est maintenant dans `src/downscaling/`.
- Les scripts Python principaux restent a la racine pour conserver les
  commandes existantes.
- Les documents generaux ont ete deplaces vers `docs/`.
