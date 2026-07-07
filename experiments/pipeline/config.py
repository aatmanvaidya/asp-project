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

# Stage 1 (HPO) always runs on a single GPU, so large train splits (e.g.
# CAMEO's ~27k samples) make HPO_TRIALS x HPO_EPOCHS passes very slow. Above
# HPO_SUBSAMPLE_THRESHOLD samples, HPO trials train on a stratified
# HPO_SUBSAMPLE_FRACTION subsample instead of the full train split — this only
# affects which config Optuna picks, not the final Stage-2 training data.
HPO_SUBSAMPLE_THRESHOLD = 5_000
HPO_SUBSAMPLE_FRACTION  = 0.20

MODELS: dict[str, str] = {
    "wav2vec2-base":    "facebook/wav2vec2-base",
    "hubert-xlarge":    "facebook/hubert-xlarge-ls960-ft",
    "wavlm-large":      "microsoft/wavlm-large",
    "distilhubert":     "ntu-spml/distilhubert",
    "hubert-base":      "facebook/hubert-base-ls960",
    "wavlm-base":       "microsoft/wavlm-base",
    "wav2vec2-large":   "facebook/wav2vec2-large",
}

DATASETS: dict[str, str] = {
    "ravdess": "xbgoose/ravdess",
    "emodb":   "renumics/emodb",
    "cameo":   "amu-cai/CAMEO",
}

# Used when RAVDESS labels are integer-encoded and dataset lacks ClassLabel names
RAVDESS_EMOTION_NAMES = [
    "neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised",
]

# CAMEO spans 13 sub-corpora whose emotion vocabularies only partially overlap
# (e.g. "poker" only in German/PAVOQUE, "sarcasm"/"excitement" only in
# English/EMNS, "anxiety"/"apology"/"assertiveness"/"concern"/"encouragement"
# only in English/JL-Corpus, "enthusiasm" only in Russian/RESD, "calm" only in
# RAVDESS). We restrict to the emotions common across most sub-corpora so the
# label space is meaningful in every language; samples with any other emotion
# are dropped.
CAMEO_CORE_EMOTIONS = {
    "anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise",
}
