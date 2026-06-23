# Contenu du paquet zip

Le paquet essentiel du projet contient:

- le code Python principal;
- `src/downscaling/`;
- les configurations dans `configs/`;
- les fichiers `README`, documentation et scripts de lancement;
- les dossiers de modeles avec leurs fichiers `train.py`, `predict.py`,
  `config.yaml`, `model.py` et documentation;
- les checkpoints originaux entraines (`*.pth`, `*.npz`) dans les dossiers
  `checkpoints/` des modeles.

Le paquet exclut volontairement:

- `outputs/`, sauf la documentation de structure;
- les predictions generees (`predictions/`, `*.nc`, `*.png`, `*.csv`);
- les caches Python (`__pycache__/`, `*.pyc`);
- les fichiers `.zip` deja telecharges;
- les donnees brutes locales et fichiers `to_predict/`.

Les checkpoints ne sont pas supprimes du projet original. Ils sont seulement
ajoutes au zip depuis leurs emplacements originaux.

