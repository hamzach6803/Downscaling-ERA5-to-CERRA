# Hyperparameters - CorrDiff

- Generated at: `2026-05-16T13:16:09`

| Parameter | Value |
| --- | --- |
| `model_type` | `CorrDiff score-based residual diffusion` |
| `variable` | `t2m` |
| `era5` | `/content/drive/MyDrive/pfe/data/era5_2021_2025.nc` |
| `cerra` | `/content/drive/MyDrive/pfe/data/cerra_2021_2025_t2m_wgs84.nc` |
| `device` | `cpu` |
| `epochs` | `20` |
| `batch_size` | `1` |
| `train_ratio` | `0.8` |
| `train_samples_full` | `5843` |
| `train_samples_used` | `5843` |
| `optimizer` | `AdamW` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `1e-05` |
| `patch_size` | `64` |
| `channels` | `32` |
| `in_channels` | `4` |
| `sigma_min` | `0.01` |
| `sigma_max` | `1.0` |
| `lambda_consistency` | `0.0` |
| `max_train_samples` | `None` |
| `num_workers` | `0` |
| `amp` | `False` |
| `use_mnt` | `False` |
| `mnt_path` | `` |
| `baseline_type` | `linear_regression` |
| `checkpoint` | `/content/drive/MyDrive/pfe/model/CorrDiff/checkpoints/corrdiff_score_t2m.pth` |
