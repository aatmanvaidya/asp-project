# Emotion Recognition — Results Summary

Experiments: 1 models × 3 datasets
F1 (macro) is reported as mean ± 95% CI across 5-fold stratified cross-validation.

## Dataset: `cameo`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-base` | ❌ `safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O erro` | — | — | — | — |

## Dataset: `emodb`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-base` | 0.7813 | 0.7673 | 0.7862 | 0.7517 ± 0.3061 | 0.7567 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-base` | 0.8674 | 0.8667 | 0.8675 | 0.8618 ± 0.0342 | 0.8677 |

## Cross-Dataset Comparison — F1-Macro [95% CI]

| Model | `cameo` | `emodb` | `ravdess` |
|-------|:------:|:------:|:------:|
| `wav2vec2-base` | ❌ | 0.7517 ± 0.3061 | 0.8618 ± 0.0342 |
