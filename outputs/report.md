# Emotion Recognition — Results Summary

Experiments: 7 models × 2 datasets

## Dataset: `emodb`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.8272 | 0.8207 | 0.8345 | 0.8199 | 0.8186 |
| `hubert-base` | 0.9383 | 0.9452 | 0.9427 | 0.9430 | 0.9379 |
| `hubert-xlarge` | 0.9136 | 0.9405 | 0.9123 | 0.9157 | 0.9091 |
| `wav2vec2-base` | 0.9383 | 0.9541 | 0.9361 | 0.9377 | 0.9344 |
| `wav2vec2-large` | 0.8395 | 0.8813 | 0.8069 | 0.8164 | 0.8325 |
| `wavlm-base` | 0.9630 | 0.9721 | 0.9610 | 0.9656 | 0.9624 |
| `wavlm-large` | 0.9877 | 0.9857 | 0.9881 | 0.9863 | 0.9877 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:----------:|:-------------:|
| `distilhubert` | 0.7963 | 0.7878 | 0.7909 | 0.7862 | 0.7985 |
| `hubert-base` | 0.8426 | 0.8502 | 0.8111 | 0.8133 | 0.8347 |
| `hubert-xlarge` | 0.9028 | 0.8985 | 0.9044 | 0.9006 | 0.9029 |
| `wav2vec2-base` | 0.8981 | 0.8990 | 0.8956 | 0.8951 | 0.8995 |
| `wav2vec2-large` | 0.8796 | 0.8739 | 0.8692 | 0.8700 | 0.8821 |
| `wavlm-base` | 0.8704 | 0.8687 | 0.8653 | 0.8642 | 0.8727 |
| `wavlm-large` | 0.8981 | 0.8907 | 0.8910 | 0.8898 | 0.8988 |

## Cross-Dataset Comparison — F1-Macro

| Model | `emodb` | `ravdess` |
|-------|:------:|:------:|
| `distilhubert` | 0.8199 | 0.7862 |
| `hubert-base` | 0.9430 | 0.8133 |
| `hubert-xlarge` | 0.9157 | 0.9006 |
| `wav2vec2-base` | 0.9377 | 0.8951 |
| `wav2vec2-large` | 0.8164 | 0.8700 |
| `wavlm-base` | 0.9656 | 0.8642 |
| `wavlm-large` | 0.9863 | 0.8898 |
