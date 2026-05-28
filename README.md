# ASP — Speech Emotion Recognition

Comparing shallow MFCC-based classifiers against fine-tuned transformer models on the [RAVDESS](https://huggingface.co/datasets/xbgoose/ravdess) dataset.

## Project structure

```
cli.py              # main entry point
baselines/
  features.py       # MFCC extraction utilities
  model.py          # 1D CNN architecture
  train.py          # training loop, evaluation, output saving
data/               # dataset loaders (one file per dataset)
models/             # saved model weights
notebooks/          # exploratory notebooks
```

## Setup

```bash
# install uv if needed
curl -Ls https://astral.sh/uv/install.sh | sh

# install dependencies
uv sync
```

## Usage

```bash
python cli.py train mfcc --dataset <name> [--epochs 50] [--lr 1e-3] [--batch-size 32]
python cli.py train transformer --dataset <name> [--model-id facebook/wav2vec2-base]
python cli.py evaluate --dataset <name> --model models/mfcc_cnn.pt
```

`train transformer` and `evaluate` are not yet implemented.

## Adding a dataset

Create `data/<name>.py` with a single function:

```python
def load() -> list[dict]:
    # return a list of samples, one dict per audio file
    return [
        {"array": np.ndarray, "sr": int, "label": str},
        ...
    ]
```

Then pass `--dataset <name>` to any `cli.py` command.

## Outputs

After `train mfcc` completes:

| Path | Content |
|------|---------|
| `baselines/results/report.txt` | sklearn classification report |
| `baselines/results/confusion_matrix.png` | per-class confusion heatmap |
| `models/mfcc_cnn.pt` | saved model weights |

## Emotions (RAVDESS)

`neutral · calm · happy · sad · angry · fearful · disgust · surprised`
