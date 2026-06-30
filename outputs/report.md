# Emotion Recognition — Results Summary

Experiments: 5 models × 2 datasets

## Dataset: `emodb`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.8272 | 0.8207 | 0.8345 | 0.8199 | 0.8186 |
| `emotion2vec-base` | ❌ `  Attempt 2: emotion2vec/emotion2vec_plus_base does not appear to have a file na` | — | — | — | — |
| `hubert-xlarge` | 0.9136 | 0.9405 | 0.9123 | 0.9157 | 0.9091 |
| `wav2vec2-base` | 0.9383 | 0.9541 | 0.9361 | 0.9377 | 0.9344 |
| `wavlm-large` | 0.9877 | 0.9857 | 0.9881 | 0.9863 | 0.9877 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.7963 | 0.7878 | 0.7909 | 0.7862 | 0.7985 |
| `emotion2vec-base` | ❌ `  Attempt 2: emotion2vec/emotion2vec_plus_base does not appear to have a file na` | — | — | — | — |
| `hubert-xlarge` | 0.9028 | 0.8985 | 0.9044 | 0.9006 | 0.9029 |
| `wav2vec2-base` | 0.8981 | 0.8990 | 0.8956 | 0.8951 | 0.8995 |
| `wavlm-large` | 0.8981 | 0.8907 | 0.8910 | 0.8898 | 0.8988 |

## Cross-Dataset Comparison — F1-Macro

| Model | `emodb` | `ravdess` |
|-------|:------:|:------:|
| `distilhubert` | 0.8199 | 0.7862 |
| `emotion2vec-base` | ❌ | ❌ |
| `hubert-xlarge` | 0.9157 | 0.9006 |
| `wav2vec2-base` | 0.9377 | 0.8951 |
| `wavlm-large` | 0.9863 | 0.8898 |
