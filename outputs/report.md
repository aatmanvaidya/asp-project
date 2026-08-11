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
| `wav2vec2-base` | 0.9327 | 0.9411 | 0.9283 | 0.9321 ± 0.0197 | 0.9326 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-base` | 0.8632 | 0.8673 | 0.8618 | 0.8576 ± 0.0429 | 0.8634 |

## Cross-Dataset Comparison — F1-Macro [95% CI]

| Model | `cameo` | `emodb` | `ravdess` |
|-------|:------:|:------:|:------:|
| `wav2vec2-base` | ❌ | 0.9321 ± 0.0197 | 0.8576 ± 0.0429 |
