# SNR vs. Model Correctness — Pearson Correlation

Point-biserial Pearson r between per-file blind SNR (dB) and prediction correctness (0/1), pooled across all 5 CV folds so every audio file is covered exactly once per model.

## SNR Distribution

| Dataset | Files | Mean | Std | Min | Median | Max |
|---------|:-----:|:----:|:---:|:---:|:------:|:---:|
| `ravdess` | 1440 | 25.76 | 12.25 | -0.45 | 24.88 | 72.32 |
| `emodb` | 535 | 19.60 | 5.93 | 7.11 | 19.15 | 44.88 |

## Pooled Per-Dataset Correlation (across all models)

| Dataset | n | Pearson r | p-value | Accuracy |
|---------|:-:|:---------:|:-------:|:--------:|
| `ravdess` | 10080 | 0.0731 | 1.977e-13 | 0.8597 |
| `emodb` | 3745 | -0.0140 | 0.3932 | 0.8841 |

## Per Model x Dataset Correlation

| Model | Dataset | n | Pearson r | p-value | Accuracy |
|-------|---------|:-:|:---------:|:-------:|:--------:|
| `distilhubert` | `emodb` | 535 | -0.0430 | 0.3203 | 0.8748 |
| `hubert-base` | `emodb` | 535 | -0.0094 | 0.8278 | 0.9439 |
| `hubert-xlarge` | `emodb` | 535 | -0.0481 | 0.267 | 0.9327 |
| `wav2vec2-base` | `emodb` | 535 | -0.0310 | 0.475 | 0.9308 |
| `wav2vec2-large` | `emodb` | 535 | -0.0283 | 0.5134 | 0.6262 |
| `wavlm-base` | `emodb` | 535 | 0.0318 | 0.4632 | 0.9159 |
| `wavlm-large` | `emodb` | 535 | 0.0536 | 0.2157 | 0.9645 |
| `distilhubert` | `ravdess` | 1440 | 0.0922 | 0.0004585 | 0.8271 |
| `hubert-base` | `ravdess` | 1440 | 0.0707 | 0.007275 | 0.8625 |
| `hubert-xlarge` | `ravdess` | 1440 | 0.1177 | 7.467e-06 | 0.8667 |
| `wav2vec2-base` | `ravdess` | 1440 | 0.0363 | 0.1689 | 0.8521 |
| `wav2vec2-large` | `ravdess` | 1440 | 0.0669 | 0.01114 | 0.8194 |
| `wavlm-base` | `ravdess` | 1440 | 0.0916 | 0.0005028 | 0.8896 |
| `wavlm-large` | `ravdess` | 1440 | 0.0375 | 0.1553 | 0.9007 |

## Accuracy by SNR Quartile (pooled across models)

### `ravdess`

| Quartile | n | Accuracy |
|----------|:-:|:--------:|
| Q1 (noisiest) | 2520 | 0.8171 |
| Q2 | 2520 | 0.8675 |
| Q3 | 2520 | 0.8544 |
| Q4 (cleanest) | 2520 | 0.9000 |

### `emodb`

| Quartile | n | Accuracy |
|----------|:-:|:--------:|
| Q1 (noisiest) | 931 | 0.8776 |
| Q2 | 938 | 0.8955 |
| Q3 | 938 | 0.8998 |
| Q4 (cleanest) | 938 | 0.8635 |
