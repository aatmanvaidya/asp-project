# Emotion Recognition — Results Summary

Experiments: 2 models × 2 datasets

## Dataset: `emodb`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-large` | ❌ `safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O erro` | — | — | — | — |
| `wavlm-large` | ❌ `safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O erro` | — | — | — | — |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `wav2vec2-large` | ❌ `safetensors._safetensors_rust.SafetensorError: Error while serializing: I/O erro` | — | — | — | — |
| `wavlm-large` | ❌ `OSError: [Errno 122] Disk quota exceeded` | — | — | — | — |

## Cross-Dataset Comparison — F1-Macro [95% CI]

| Model | `emodb` | `ravdess` |
|-------|:------:|:------:|
| `wav2vec2-large` | ❌ | ❌ |
| `wavlm-large` | ❌ | ❌ |
