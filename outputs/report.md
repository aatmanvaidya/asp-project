# Emotion Recognition — Results Summary

Experiments: 7 models × 2 datasets
F1 (macro) is reported as mean ± 95% CI across 5-fold stratified cross-validation.

## Dataset: `emodb`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `distilhubert` | 0.8748 | 0.8884 | 0.8635 | 0.8705 ± 0.0314 | 0.8749 |
| `hubert-base` | 0.9439 | 0.9508 | 0.9450 | 0.9455 ± 0.0324 | 0.9435 |
| `hubert-xlarge` | 0.9327 | 0.9395 | 0.9321 | 0.9331 ± 0.0374 | 0.9321 |
| `wav2vec2-base` | 0.9308 | 0.9383 | 0.9310 | 0.9305 ± 0.0445 | 0.9293 |
| `wav2vec2-large` | 0.6262 | 0.5565 | 0.5827 | 0.5466 ± 0.5603 | 0.5665 |
| `wavlm-base` | 0.9159 | 0.9363 | 0.9142 | 0.9135 ± 0.0625 | 0.9125 |
| `wavlm-large` | 0.9645 | 0.9737 | 0.9665 | 0.9667 ± 0.0278 | 0.9646 |

## Dataset: `ravdess`

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) [95% CI] | F1 (weighted) |
|-------|:--------:|:-----------------:|:--------------:|:--------------------:|:-------------:|
| `distilhubert` | 0.8271 | 0.8207 | 0.8213 | 0.8188 ± 0.0318 | 0.8263 |
| `hubert-base` | 0.8625 | 0.8628 | 0.8659 | 0.8586 ± 0.0286 | 0.8649 |
| `hubert-xlarge` | 0.8667 | 0.8721 | 0.8670 | 0.8629 ± 0.0261 | 0.8660 |
| `wav2vec2-base` | 0.8521 | 0.8518 | 0.8529 | 0.8459 ± 0.0465 | 0.8515 |
| `wav2vec2-large` | 0.8194 | 0.8372 | 0.8200 | 0.8119 ± 0.1299 | 0.8200 |
| `wavlm-base` | 0.8896 | 0.8922 | 0.8823 | 0.8842 ± 0.0447 | 0.8898 |
| `wavlm-large` | 0.9007 | 0.9008 | 0.9003 | 0.8971 ± 0.0380 | 0.9002 |

## Cross-Dataset Comparison — F1-Macro [95% CI]

| Model | `emodb` | `ravdess` |
|-------|:------:|:------:|
| `distilhubert` | 0.8705 ± 0.0314 | 0.8188 ± 0.0318 |
| `hubert-base` | 0.9455 ± 0.0324 | 0.8586 ± 0.0286 |
| `hubert-xlarge` | 0.9331 ± 0.0374 | 0.8629 ± 0.0261 |
| `wav2vec2-base` | 0.9305 ± 0.0445 | 0.8459 ± 0.0465 |
| `wav2vec2-large` | 0.5466 ± 0.5603 | 0.8119 ± 0.1299 |
| `wavlm-base` | 0.9135 ± 0.0625 | 0.8842 ± 0.0447 |
| `wavlm-large` | 0.9667 ± 0.0278 | 0.8971 ± 0.0380 |
