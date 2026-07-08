# Emotion Recognition — Results Summary

Experiments: 6 models × 2 datasets

## Dataset: `cameo`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.7905 | 0.7818 | 0.7764 | 0.7787 | 0.7900 |
| `emotion2vec-base` | ❌ `  Attempt 2: emotion2vec/emotion2vec_plus_base does not appear to have a file na` | — | — | — | — |
| `hubert-xlarge` | 0.9082 | 0.9053 | 0.9024 | 0.9036 | 0.9079 |
| `mfcc-cnn` | 0.6383 | 0.6258 | 0.6249 | 0.6239 | 0.6396 |
| `wav2vec2-base` | 0.8759 | 0.8707 | 0.8677 | 0.8690 | 0.8758 |
| `wavlm-large` | 0.8726 | 0.8663 | 0.8651 | 0.8639 | 0.8716 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.7963 | 0.7878 | 0.7909 | 0.7862 | 0.7985 |
| `emotion2vec-base` | ❌ `  Attempt 2: emotion2vec/emotion2vec_plus_base does not appear to have a file na` | — | — | — | — |
| `hubert-xlarge` | 0.9028 | 0.8985 | 0.9044 | 0.9006 | 0.9029 |
| `mfcc-cnn` | 0.6701 | 0.6839 | 0.6639 | 0.6635 | 0.6707 |
| `wav2vec2-base` | 0.8981 | 0.8990 | 0.8956 | 0.8951 | 0.8995 |
| `wavlm-large` | 0.8981 | 0.8907 | 0.8910 | 0.8898 | 0.8988 |

## Cross-Dataset Comparison — F1-Macro

| Model | `cameo` | `ravdess` |
|-------|:------:|:------:|
| `distilhubert` | 0.7787 | 0.7862 |
| `emotion2vec-base` | ❌ | ❌ |
| `hubert-xlarge` | 0.9036 | 0.9006 |
| `mfcc-cnn` | 0.6239 | 0.6635 |
| `wav2vec2-base` | 0.8690 | 0.8951 |
| `wavlm-large` | 0.8639 | 0.8898 |
