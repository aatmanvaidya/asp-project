"""Central configuration — edit constants here to change the full pipeline."""

SAMPLING_RATE = 16_000
MAX_DURATION  = 5.0
MAX_LENGTH    = int(SAMPLING_RATE * MAX_DURATION)

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

BATCH_SIZE    = 8
NUM_EPOCHS    = 30
# datasets.map() batch size during feature extraction — kept small because raw
# audio arrays are fully decoded before truncation to MAX_LENGTH, and a large
# batch (the datasets default is 1000) can OOM on bigger splits like CAMEO.
PREPROCESS_BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1

EARLY_STOPPING_PATIENCE  = 5
EARLY_STOPPING_THRESHOLD = 0.001

HPO_TRIALS = 10  # Optuna trials per experiment (Stage 1 — single GPU only)
HPO_EPOCHS = 3   # epochs per HPO trial (short, just enough to rank configs)

# Large training sets (e.g. CAMEO's ~27k merged examples) make a full HPO
# trial take ~40min regardless of batch size, so 20 trials x 3 epochs is
# multiple GPU-days. Above this many train examples, HPO runs on a random
# subsample with fewer trials/epochs — Stage 2 training still uses 100% of
# the data, only hyperparameter search is shrunk.
HPO_LARGE_DATASET_THRESHOLD = 10_000
HPO_SUBSAMPLE_FRAC = 0.15
HPO_TRIALS_LARGE   = 8
HPO_EPOCHS_LARGE   = 2

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
    "cameo":   "amu-cai/CAMEO",
}

# Canonical emotion labels after CAMEO harmonization
CAMEO_CANONICAL_LABELS = [
    "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"
]

CAMEO_LABEL_ALIASES: dict[str, str | None] = {
    "angry": "anger",
    "happy": "happiness",
    "joy": "happiness",
    "sad": "sadness",
    "fearful": "fear",
    "scared": "fear",
    "surprised": "surprise",
    # drop non-standard emotions
    "calm": None,
    "contempt": None,
    "boredom": None,
    "bored": None,
    "enthusiasm": None,
    "excited": None,
    "poker": None,
    "amused": None,
    "sleepy": None,
}

# Used when RAVDESS labels are integer-encoded and dataset lacks ClassLabel names
RAVDESS_EMOTION_NAMES = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
]
