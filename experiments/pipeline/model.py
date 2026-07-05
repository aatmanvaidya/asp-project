"""Model and feature-extractor loading with graceful fallbacks."""
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def load_feature_extractor(hf_name: str):
    """Load the feature extractor for the given HuggingFace model name."""
    try:
        fe = AutoFeatureExtractor.from_pretrained(hf_name, trust_remote_code=True)
        print(f"  Feature extractor : {type(fe).__name__}  (sr={fe.sampling_rate})")
        return fe
    except Exception as e:
        print(f"  AutoFeatureExtractor failed ({e}), falling back to wav2vec2-base extractor.")
        from transformers import Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        return fe


def load_model(hf_name: str, num_labels: int, id2label: dict, label2id: dict):
    """
    Load a model for sequence classification.

    Tries AutoModelForAudioClassification first (handles wav2vec2, hubert,
    wavlm, distilhubert), then falls back to
    Wav2Vec2ForSequenceClassification for models that predate the Auto API.
    """
    kwargs = dict(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    try:
        model = AutoModelForAudioClassification.from_pretrained(
            hf_name, trust_remote_code=True, **kwargs
        )
    except Exception as e1:
        print(f"  AutoModelForAudioClassification failed ({e1}), trying Wav2Vec2ForSequenceClassification.")
        try:
            from transformers import Wav2Vec2ForSequenceClassification
            model = Wav2Vec2ForSequenceClassification.from_pretrained(hf_name, **kwargs)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load model '{hf_name}'.\n  Attempt 1: {e1}\n  Attempt 2: {e2}"
            ) from e2

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model             : {type(model).__name__}  ({total:,} params, {trainable:,} trainable)")
    return model
