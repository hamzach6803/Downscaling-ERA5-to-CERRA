# Architecture, parametres et formules des modeles

Ce document presente les modeles de downscaling ERA5 vers CERRA avec leurs formules et leur correspondance avec l'implementation Python.

## Notation commune

- $x$: champ ERA5.
- $x_{hr}$: champ ERA5 interpole sur la grille CERRA.
- $y$: champ CERRA cible.
- $t$: indice temporel.
- $(i,j)$: position spatiale latitude/longitude.
- $\theta$: parametres entrainables d'un modele.
- $r = 0.8$: proportion temporelle utilisee pour l'entrainement.

## Pretraitement commun

Implementation principale: `utils.py`.

### Alignement temporel

Les deux jeux de donnees sont alignes sur les dates communes:

$$
\mathcal{T} = \mathcal{T}_{ERA5} \cap \mathcal{T}_{CERRA}
$$

$$
x = ERA5[\mathcal{T}], \qquad y = CERRA[\mathcal{T}]
$$

### Interpolation spatiale

ERA5 est interpole sur la grille CERRA:

$$
x_{hr}(t,i,j)
= \operatorname{Interp}_{lin}
\left(x(t), \operatorname{lat}^{CERRA}_i, \operatorname{lon}^{CERRA}_j\right)
$$

Dans le code:

```python
era5_interp = era5.interp(latitude=cerra.latitude, longitude=cerra.longitude, method="linear")
```

### Nettoyage des valeurs manquantes

Les valeurs manquantes sont interpolees dans le temps, puis les NaN restants sont remplaces par la moyenne:

$$
z_{clean}
= \operatorname{fillna}
\left(\operatorname{interpolate\_na}(z,\operatorname{dim}=time), \overline{z}\right)
$$

### Normalisation

Les statistiques sont calculees uniquement sur le train:

$$
\mu_x = \operatorname{mean}(x_{train}), \qquad
\sigma_x = \operatorname{std}(x_{train})
$$

$$
\mu_y = \operatorname{mean}(y_{train}), \qquad
\sigma_y = \operatorname{std}(y_{train})
$$

La normalisation est:

$$
x_{norm} = \frac{x_{hr} - \mu_x}{\sigma_x}
$$

$$
y_{norm} = \frac{y - \mu_y}{\sigma_y}
$$

La denormalisation est:

$$
\hat{y} = \hat{y}_{norm}\sigma_y + \mu_y
$$

## 1. PyESD: regression lineaire pixel par pixel

Implementation: `PyESD/train.py`, `PyESD/predict.py`, `utils.py`.

Pour chaque pixel $(i,j)$, PyESD apprend une regression lineaire locale:

$$
y_{norm}(t,i,j)
= a(i,j)x_{norm}(t,i,j) + b(i,j) + \varepsilon(t,i,j)
$$

Les parametres sont calcules analytiquement:

$$
a(i,j)
= \frac{
\operatorname{Cov}\left(x_{norm}(:,i,j), y_{norm}(:,i,j)\right)
}{
\max\left(\operatorname{Var}\left(x_{norm}(:,i,j)\right), 10^{-8}\right)
}
$$

$$
b(i,j)
= \operatorname{mean}\left(y_{norm}(:,i,j)\right)
- a(i,j)\operatorname{mean}\left(x_{norm}(:,i,j)\right)
$$

La prediction normalisee est:

$$
\hat{y}_{norm}(t,i,j)
= a(i,j)x_{norm}(t,i,j) + b(i,j)
$$

Puis:

$$
\hat{y}(t,i,j)
= \hat{y}_{norm}(t,i,j)\sigma_y + \mu_y
$$

Parametres sauvegardes:

$$
\text{slope}=a, \qquad \text{intercept}=b
$$

Checkpoint: `PyESD/checkpoints/pyesd_linear_<variable>.npz`.

## 2. DeepSD: CNN supervise

Implementation: `DeepSD/model.py`, `DeepSD/train.py`, `DeepSD/predict.py`.

DeepSD approxime une fonction non lineaire:

$$
\hat{y}_{norm} = f_\theta(x_{norm})
$$

L'architecture est:

$$
h_1 = \operatorname{ReLU}\left(\operatorname{Conv}_{3 \times 3}^{1 \to C}(x)\right)
$$

$$
h_2 = \operatorname{ReLU}\left(\operatorname{Conv}_{3 \times 3}^{C \to C}(h_1)\right)
$$

$$
h_3 = \operatorname{ReLU}\left(\operatorname{Conv}_{3 \times 3}^{C \to C}(h_2)\right)
$$

$$
h_4 = \operatorname{Upsample}_{bilinear}(h_3, scale)
$$

$$
h_5 = \operatorname{ReLU}\left(\operatorname{Conv}_{3 \times 3}^{C \to C/2}(h_4)\right)
$$

$$
h_6 = \operatorname{ReLU}\left(\operatorname{Conv}_{3 \times 3}^{C/2 \to C/2}(h_5)\right)
$$

$$
\hat{y}_{norm}
= \operatorname{Conv}_{3 \times 3}^{C/2 \to 1}(h_6)
$$

Le facteur d'agrandissement est:

$$
scale = \operatorname{round}\left(\frac{H_{CERRA}}{H_{ERA5}}\right)
$$

La fonction de perte est la MSE:

$$
\mathcal{L}(\theta)
= \frac{1}{N}\sum_{n=1}^{N}
\left\lVert f_\theta(x_n) - y_n \right\rVert_2^2
$$

Parametres principaux: $C=96$, epochs $=50$, AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$.

Checkpoint: `DeepSD/checkpoints/deepsd_best_<variable>.pth`.

## 3. CAE: autoencodeur convolutionnel supervise

Implementation: `CAE/model.py`, `CAE/train.py`, `CAE/predict.py`.

Le CAE apprend une transformation deterministe:

$$
\hat{y}_{norm} = D_\theta(E_\theta(x_{norm}))
$$

avec:

- $E_\theta$: encodeur convolutionnel;
- $D_\theta$: decodeur convolutionnel;
- un bottleneck spatial compact entre les deux.

L'encodeur applique deux blocs convolutionnels avec sous-echantillonnage:

$$
z = E_\theta(x_{norm})
$$

Le decodeur reconstruit la carte CERRA normalisee:

$$
\hat{y}_{norm} = D_\theta(z)
$$

La prediction finale est:

$$
\hat{y} = \hat{y}_{norm}\sigma_y + \mu_y
$$

La fonction de perte est:

$$
\mathcal{L}(\theta)
= \frac{1}{N}\sum_{n=1}^{N}
\left\lVert D_\theta(E_\theta(x_n)) - y_n \right\rVert_2^2
$$

Parametres principaux: $C=32$, bottleneck $Z=128$, epochs $=50$, AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$.

Checkpoint: `CAE/checkpoints/cae_best_<variable>.pth`.

## 4. ESRGAN: generateur RRDB et discriminateur

Implementation: `ESRGAN/model.py`, `ESRGAN/train.py`, `ESRGAN/predict.py`.

Le generateur apprend:

$$
\hat{y}_{fake,norm} = G_\theta(x_{norm})
$$

### Generateur RRDB

La structure globale est:

$$
f_0 = \operatorname{Conv}_{3 \times 3}^{1 \to C}(x)
$$

$$
f_{body}
= \operatorname{Conv}_{3 \times 3}^{C \to C}
\left(\operatorname{RRDB}_N \circ \cdots \circ \operatorname{RRDB}_1(f_0)\right)
$$

$$
\hat{y}_{fake,norm}
= \operatorname{Tail}(f_0 + f_{body})
$$

Dans un bloc dense residuel:

$$
z_1 = \phi(W_1[x])
$$

$$
z_2 = \phi(W_2[x,z_1])
$$

$$
z_3 = \phi(W_3[x,z_1,z_2])
$$

$$
z_4 = \phi(W_4[x,z_1,z_2,z_3])
$$

$$
z_5 = W_5[x,z_1,z_2,z_3,z_4]
$$

avec $\phi=\operatorname{LeakyReLU}(0.2)$.

La sortie du bloc dense residuel est:

$$
\operatorname{DRB}(x) = x + \beta z_5
$$

avec:

$$
\beta = 0.2
$$

Un RRDB contient trois blocs denses residuels:

$$
\operatorname{RRDB}(x)
= x + \beta \operatorname{DRB}_3(\operatorname{DRB}_2(\operatorname{DRB}_1(x)))
$$

### Discriminateur

Le discriminateur produit un logit:

$$
D_\phi(y) \in \mathbb{R}
$$

### Pertes adversariales

La perte du discriminateur est:

$$
\mathcal{L}_D
= \frac{1}{2}
\left[
\operatorname{BCE}(D_\phi(y_{real}), 1)
+
\operatorname{BCE}(D_\phi(G_\theta(x)), 0)
\right]
$$

La perte du generateur est:

$$
\mathcal{L}_G
= \lambda_{L1}\left\lVert G_\theta(x)-y_{real}\right\rVert_1
+ \lambda_{adv}\operatorname{BCE}(D_\phi(G_\theta(x)), 1)
$$

avec:

$$
\lambda_{L1}=1.0, \qquad \lambda_{adv}=10^{-3}
$$

Parametres principaux: $C=32$, RRDB blocks $=2$, patch size $=64$, AdamW, learning rate $10^{-4}$.

Checkpoint: `ESRGAN/checkpoints/esrgan_generator_<variable>.pth`.

## 5. DDPM Cosine: diffusion conditionnelle

Implementation: `DDPM/model.py`, `DDPM/train.py`, `DDPM/predict.py`.

Le modele apprend a predire le bruit ajoute a la cible CERRA normalisee.

### Schedule cosinus

Pour $S=20$ et $s=0.008$:

$$
\bar{\alpha}(t)
= \cos^2
\left(
\frac{\frac{t}{S}+s}{1+s}
\frac{\pi}{2}
\right)
$$

Dans le code, cette valeur est bornee:

$$
\bar{\alpha}(t) \in [10^{-5}, 0.999]
$$

### Bruitage direct

Avec:

$$
\varepsilon \sim \mathcal{N}(0,I)
$$

la cible bruitee est:

$$
y_t
= \sqrt{\bar{\alpha}(t)}y_{norm}
+ \sqrt{1-\bar{\alpha}(t)}\varepsilon
$$

### Reseau de bruit

La condition est:

$$
c = \operatorname{Upsample}_{bilinear}(x_{norm}, H_{CERRA}, W_{CERRA})
$$

L'entree du U-Net est:

$$
X = [c, y_t]
$$

Le modele predit:

$$
\hat{\varepsilon}
= \varepsilon_\theta(c,y_t)
$$

Architecture simplifiee:

$$
e_1 = B_{2 \to C}(X)
$$

$$
e_2 = B_{C \to 2C}(\operatorname{MaxPool}(e_1))
$$

$$
m = B_{2C \to 2C}(e_2)
$$

$$
d_1 = B_{4C \to C}([\operatorname{Upsample}(m), e_2])
$$

$$
\hat{\varepsilon}
= \operatorname{Conv}_{3 \times 3}^{2C \to 1}
([\operatorname{Upsample}(d_1), e_1])
$$

La perte est:

$$
\mathcal{L}(\theta)
= \mathbb{E}_{t,\varepsilon}
\left[
\left\lVert \varepsilon_\theta(c,y_t)-\varepsilon \right\rVert_2^2
\right]
$$

### Prediction

Le script initialise:

$$
y^{(0)} \sim \mathcal{N}(0,I)
$$

Puis applique une mise a jour de denoising simplifiee:

$$
y^{(k+1)}
= y^{(k)} - \frac{\varepsilon_\theta(c,y^{(k)})}{S}
$$

La prediction finale est:

$$
\hat{y} = y^{(S)}\sigma_y + \mu_y
$$

Un filtre bilateral est applique apres prediction.

Parametres principaux: $C=64$, $S=20$, AdamW, learning rate $10^{-4}$.

Checkpoint: `DDPM/checkpoints/ddpm_cosine_best_<variable>.pth`.

## 6. CorrDiff: diffusion par score sur les residus

Implementation commune: `corrdiff_core.py`.

Scripts: `CorrDiff/train.py`, `CorrDiff/predict.py`.

### Baseline lineaire

CorrDiff commence par une regression lineaire pixel par pixel:

$$
b(t,i,j)
= a(i,j)x_{norm}(t,i,j) + c(i,j)
$$

avec:

$$
a
= \frac{\operatorname{Cov}(x_{norm},y_{norm})}
{\max(\operatorname{Var}(x_{norm}),10^{-8})}
$$

$$
c
= \operatorname{mean}(y_{norm})
- a\operatorname{mean}(x_{norm})
$$

Le residu est:

$$
r(t,i,j)
= y_{norm}(t,i,j) - b(t,i,j)
$$

### Bruitage du residu

Le niveau de bruit est tire log-uniformement:

$$
\log \sigma
\sim \mathcal{U}(\log \sigma_{min}, \log \sigma_{max})
$$

$$
\sigma = \exp(\log \sigma)
$$

avec:

$$
\sigma_{min}=0.01, \qquad \sigma_{max}=1.0
$$

Le residu bruite est:

$$
r_\sigma = r + \sigma \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0,I)
$$

Le score cible est:

$$
s_{target}
= -\frac{\varepsilon}{\sigma}
$$

### ScoreUNet

Les canaux d'entree sont:

$$
X = [x_{norm}, b, r_\sigma, \log(\sigma)\mathbf{1}]
$$

Le reseau approxime:

$$
s_\theta
= \operatorname{ScoreUNet}_\theta(x_{norm}, b, r_\sigma, \sigma)
$$

Architecture:

$$
e_1 = B_{4 \to C}(X)
$$

$$
e_2 = B_{C \to 2C}(\operatorname{MaxPool}(e_1))
$$

$$
m = B_{2C \to 2C}(e_2)
$$

$$
d = B_{3C \to C}([\operatorname{Upsample}(m), e_1])
$$

$$
s_\theta
= \operatorname{Conv}_{3 \times 3}^{C \to 1}(d)
$$

### Perte de score matching

La perte utilise une ponderation par $\sigma^2$:

$$
\mathcal{L}_{score}(\theta)
= \mathbb{E}
\left[
\sigma^2
\left\lVert s_\theta - s_{target} \right\rVert_2^2
\right]
$$

### Sampling Langevin

Initialisation:

$$
r^{(0)}
\sim \mathcal{N}(0,\sigma_{max}^2I)
$$

Suite de niveaux de bruit:

$$
\sigma_k
= \exp\left(
\operatorname{linspace}(\log\sigma_{max}, \log\sigma_{min}, K)_k
\right)
$$

avec $K=20$.

Pas de Langevin:

$$
\eta_k \sim \mathcal{N}(0,I)
$$

$$
\Delta_k = (snr \cdot \sigma_k)^2
$$

$$
r^{(k+1)}
= r^{(k)}
+ \Delta_k s_\theta(x_{norm}, b, r^{(k)}, \sigma_k)
+ \sqrt{2\Delta_k}\eta_k
$$

Le residu est borne:

$$
r^{(k+1)}
\leftarrow \operatorname{clip}(r^{(k+1)}, -6, 6)
$$

La prediction finale est:

$$
\hat{y}_{norm}
= b + r^{(K)}
$$

$$
\hat{y}
= \hat{y}_{norm}\sigma_y + \mu_y
$$

Parametres principaux: $C=32$, patch size $=64$, epochs $=20$, AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$.

Checkpoint: `CorrDiff/checkpoints/corrdiff_score_<variable>.pth`.

## 7. Metriques d'evaluation

Implementation: `evaluate` dans `utils.py`.

### MSE et RMSE

$$
MSE
= \frac{1}{N}\sum_{n=1}^{N}(\hat{y}_n-y_n)^2
$$

$$
RMSE
= \sqrt{MSE}
$$

### MAE

$$
MAE
= \frac{1}{N}\sum_{n=1}^{N}|\hat{y}_n-y_n|
$$

### Biais

$$
Bias
= \frac{1}{N}\sum_{n=1}^{N}(\hat{y}_n-y_n)
$$

### Coefficient $R^2$

$$
SS_{res}
= \sum_{n=1}^{N}(y_n-\hat{y}_n)^2
$$

$$
SS_{tot}
= \sum_{n=1}^{N}(y_n-\overline{y})^2
$$

$$
R^2
= 1 - \frac{SS_{res}}{SS_{tot}}
$$

## 8. Tableau recapitulatif

| Modele | Forme mathematique | Perte / critere | Implementation |
| --- | --- | --- | --- |
| PyESD | $\hat{y}=ax+b$ | solution analytique covariance/variance | `PyESD/train.py` |
| DeepSD | $\hat{y}=f_\theta(x)$ | MSE | `DeepSD/model.py` |
| CAE | $\hat{y}=D_\theta(E_\theta(x))$ | MSE | `CAE/model.py` |
| ESRGAN | $\hat{y}=G_\theta(x)$ | L1 + BCE adversarial | `ESRGAN/model.py` |
| DDPM | $\hat{\varepsilon}=\varepsilon_\theta(c,y_t)$ | MSE sur le bruit | `DDPM/model.py` |
| CorrDiff | $\hat{y}=b+r^{(K)}$ | score matching | `corrdiff_core.py` |

## 9. Formule generale du pipeline

Tous les modeles suivent la forme generale:

$$
\hat{y}
= F_{model}
\left(
\frac{x_{hr}-\mu_x}{\sigma_x}
\right)
\sigma_y + \mu_y
$$

La difference entre les modeles est la definition de $F_{model}$:

$$
F_{model}
\in
\{
\text{regression lineaire},
\text{CNN},
\text{autoencodeur convolutionnel},
\text{GAN},
\text{DDPM},
\text{diffusion par score}
\}
$$

Le flux informatique est:

1. Charger ERA5 et CERRA.
2. Aligner les dates communes.
3. Interpoler ERA5 vers la grille CERRA.
4. Nettoyer les valeurs manquantes.
5. Calculer $\mu_x,\sigma_x,\mu_y,\sigma_y$ sur le train.
6. Normaliser $x$ et $y$.
7. Entrainer le modele.
8. Sauvegarder checkpoint et statistiques.
9. Charger une date de prediction.
10. Produire $\hat{y}_{norm}$.
11. Denormaliser pour obtenir $\hat{y}$.
12. Sauvegarder NetCDF, figures et metriques.
