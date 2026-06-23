# README Results

Ce fichier explique comment utiliser les deux scripts de resultats :

- `results_extreme90.py` : trace les resultats sur les 90 cas extremes.
- `results_test20.py` : trace les resultats sur le jeu de test 20 %.

Les deux scripts calculent les predictions des modeles, generent les figures et sauvegardent les metriques. Ils peuvent aussi sauvegarder les predictions en `.nc`, puis les reutiliser plus tard pour refaire uniquement le plotage.

## 1. Script `results_extreme90.py`

Ce script utilise `results_90times.py` en changeant le dossier de sortie vers :

```text
comparisons1/results_extreme90/
```

### Commande simple

```powershell
python results_extreme90.py
```

Cette commande recalcule les predictions des modeles puis genere les figures et les fichiers CSV.

### Sauvegarder les predictions en `.nc`

```powershell
python results_extreme90.py --save-predictions
```

Les predictions sont sauvegardees dans :

```text
comparisons1/results_extreme90/predictions/
```

Exemple de fichier :

```text
deepsd_pred_90times_t2m.nc
```

### Reutiliser les predictions deja sauvegardees

```powershell
python results_extreme90.py --use-saved-predictions
```

Cette commande ne recalcule pas les modeles. Elle charge les fichiers `.nc` deja presents dans le dossier `predictions/`, puis regenere les plots et les metriques.

### Arguments disponibles

`--era5`
: Chemin vers le fichier ERA5 contenant les 90 dates. Par defaut, le script utilise `data/era5_2025_90times.nc`.

`--cerra`
: Chemin vers le fichier CERRA contenant les 90 dates. Par defaut, le script utilise `data/cerra_2025_90times.nc`.

`--variable`
: Variable a traiter. Par defaut, la variable vient de `utils.DEFAULT_VARIABLE`, generalement `t2m`.

`--models`
: Liste des modeles a utiliser. Exemple :

```powershell
python results_extreme90.py --models DeepSD CorrDiff PyESD
```

Modeles disponibles :

```text
Interpolation PyESD DeepSD ESRGAN DDPM CorrDiff
```

`--date`
: Date precise a tracer. Exemple :

```powershell
python results_extreme90.py --date 2021-01-02
```

Le script choisit la date disponible la plus proche.

`--time-index`
: Index temporel a tracer si `--date` n'est pas fourni. Exemple :

```powershell
python results_extreme90.py --time-index 0
```

`--seed`
: Graine utilisee quand aucune date et aucun index ne sont donnes. Cela permet de reproduire le choix aleatoire de la date.

`--batch-size`
: Taille des lots pendant la prediction des modeles PyTorch. Valeur par defaut : `1`.

`--device`
: Appareil utilise pour les modeles PyTorch. Valeurs possibles :

```text
cpu cuda
```

Exemple :

```powershell
python results_extreme90.py --device cpu
```

`--corrdiff-sample-steps`
: Nombre d'etapes d'echantillonnage pour CorrDiff. Valeur par defaut : `20`.

`--corrdiff-langevin-steps`
: Nombre d'etapes Langevin pour CorrDiff. Valeur par defaut : `1`.

`--corrdiff-snr`
: Parametre SNR utilise par CorrDiff. Valeur par defaut : `0.15`.

`--ddpm-steps`
: Nombre d'etapes de denoising pour DDPM. Si l'argument n'est pas donne, le script utilise la valeur sauvegardee dans le checkpoint.

`--bilateral-window`
: Taille de fenetre du filtre bilateral applique a DDPM et CorrDiff. Valeur par defaut : `5`.

`--bilateral-sigma-spatial`
: Sigma spatial du filtre bilateral. Valeur par defaut : `2.0`.

`--palette-mode`
: Mode de palette pour les cartes. Valeurs possibles :

```text
uniform uniforme unforme shared global adaptive adaptative ch
```

`uniform` utilise la meme echelle de couleurs entre CERRA et les predictions. `adaptive` utilise une echelle adaptee a chaque carte. `ch` utilise une palette adaptee pour `DDPM` et `CorrDiff`, tandis que CERRA et les autres modeles gardent une palette commune.

`--save-predictions`
: Sauvegarde les predictions calculees en `.nc`.

`--use-saved-predictions`
: Charge les predictions deja sauvegardees en `.nc` au lieu de recalculer les modeles.

## 2. Script `results_test20.py`

Ce script trace les resultats sur la partie test 20 % du dataset principal.

Le dossier de sortie est :

```text
comparisons2/results_test20/
```

### Commande simple

```powershell
python results_test20.py
```

### Sauvegarder les predictions en `.nc`

```powershell
python results_test20.py --save-predictions
```

Les predictions sont sauvegardees dans :

```text
comparisons2/results_test20/predictions/
```

Exemple de fichier :

```text
deepsd_pred_test20_t2m.nc
```

### Reutiliser les predictions deja sauvegardees

```powershell
python results_test20.py --use-saved-predictions
```

Cette commande recharge les fichiers `.nc`, puis regenere les plots et les metriques sans recalculer les modeles.

### Arguments disponibles

`--era5`
: Chemin vers le fichier ERA5 complet. Par defaut, le script utilise `utils.DEFAULT_ERA5`.

`--cerra`
: Chemin vers le fichier CERRA complet. Par defaut, le script utilise `utils.CERRA_DATA_PATH`.

`--variable`
: Variable a traiter. Par defaut, la variable vient de `utils.DEFAULT_VARIABLE`, generalement `t2m`.

`--train-ratio`
: Pourcentage utilise pour separer train/test. Par defaut, la valeur vient de `utils.TRAIN_RATIO`. Le script prend la partie apres le train comme jeu de test 20 %.

`--models`
: Liste des modeles a utiliser. Exemple :

```powershell
python results_test20.py --models Interpolation DeepSD CorrDiff
```

Modeles disponibles :

```text
Interpolation PyESD DeepSD ESRGAN DDPM CorrDiff
```

`--date`
: Date precise a tracer. Le script prend la date de test la plus proche.

`--time-index`
: Index temporel dans le jeu de test si aucune date n'est donnee.

`--seed`
: Graine utilisee pour choisir une date aleatoire de maniere reproductible.

`--max-test-samples`
: Nombre maximum de dates du jeu de test a utiliser. Exemple :

```powershell
python results_test20.py --max-test-samples 20
```

`--batch-size`
: Taille des lots pendant la prediction. Valeur par defaut : `1`.

`--device`
: Appareil utilise pour les modeles PyTorch. Valeurs possibles :

```text
cpu cuda
```

`--corrdiff-sample-steps`
: Nombre d'etapes d'echantillonnage pour CorrDiff. Valeur par defaut : `20`.

`--corrdiff-langevin-steps`
: Nombre d'etapes Langevin pour CorrDiff. Valeur par defaut : `1`.

`--corrdiff-snr`
: Parametre SNR utilise par CorrDiff. Valeur par defaut : `0.15`.

`--bilateral-window`
: Taille de fenetre du filtre bilateral. Valeur par defaut : `5`.

`--bilateral-sigma-spatial`
: Sigma spatial du filtre bilateral. Valeur par defaut : `2.0`.

`--save-predictions`
: Sauvegarde les predictions calculees en `.nc`.

`--use-saved-predictions`
: Charge les predictions deja sauvegardees en `.nc` au lieu de recalculer les modeles.

## 3. Workflow recommande

Premiere execution, avec calcul des predictions et sauvegarde :

```powershell
python results_extreme90.py --save-predictions
python results_test20.py --save-predictions
```

Executions suivantes, pour refaire seulement les plots :

```powershell
python results_extreme90.py --use-saved-predictions
python results_test20.py --use-saved-predictions
```

Pour tracer seulement quelques modeles deja sauvegardes :

```powershell
python results_extreme90.py --use-saved-predictions --models DeepSD CorrDiff PyESD
python results_test20.py --use-saved-predictions --models DeepSD CorrDiff PyESD
```
