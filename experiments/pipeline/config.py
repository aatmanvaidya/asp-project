"""Central configuration — edit constants here to change the full pipeline."""

SAMPLING_RATE = 16_000
MAX_DURATION  = 5.0
MAX_LENGTH    = int(SAMPLING_RATE * MAX_DURATION)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE    = 8
NUM_EPOCHS    = 30
LEARNING_RATE = 3e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1

EARLY_STOPPING_PATIENCE  = 5
EARLY_STOPPING_THRESHOLD = 0.001

HPO_TRIALS = 10  # Optuna trials per experiment (Stage 1 — single GPU only)
HPO_EPOCHS = 3   # epochs per HPO trial (short, just enough to rank configs)
SEED       = 42

MODELS: dict[str, str] = {
    "wav2vec2-base":    "facebook/wav2vec2-base",
    "hubert-xlarge":    "facebook/hubert-xlarge-ls960-ft",
    "emotion2vec-base": "emotion2vec/emotion2vec_plus_base",
    "wavlm-large":      "microsoft/wavlm-large",
    "distilhubert":     "ntu-spml/distilhubert",
}

DATASETS: dict[str, str] = {
    "ravdess": "xbgoose/ravdess",
    "emodb":   "renumics/emodb",
}

# Used when RAVDESS labels are integer-encoded and dataset lacks ClassLabel names
RAVDESS_EMOTION_NAMES = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
]
