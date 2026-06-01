"""Offline unit tests for the MFCC baseline (features.py, model.py, train.py).

Run with:  uv run pytest baselines/test_baselines.py -v
No audio files or network access required.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# features.py
# ---------------------------------------------------------------------------

from baselines.features import N_MFCC, T_MAX, extract_mfcc, build_feature_matrix


def _sine_wave(duration_s: float = 1.0, sr: int = 22050) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


class TestExtractMfcc:
    def test_output_shape(self):
        audio = _sine_wave(1.0)
        result = extract_mfcc(audio, sr=22050)
        assert result.shape == (N_MFCC, T_MAX)

    def test_short_audio_is_padded(self):
        # ~0.1 s — far fewer than T_MAX frames
        audio = _sine_wave(0.1)
        result = extract_mfcc(audio, sr=22050)
        assert result.shape == (N_MFCC, T_MAX)
        # padded columns should be zero
        raw_frames = librosa_frame_count(len(audio))
        assert np.all(result[:, raw_frames:] == 0.0)

    def test_long_audio_is_truncated(self):
        # ~10 s — much more than T_MAX frames
        audio = _sine_wave(10.0)
        result = extract_mfcc(audio, sr=22050)
        assert result.shape == (N_MFCC, T_MAX)

    def test_output_dtype_is_float32(self):
        audio = _sine_wave(1.0)
        result = extract_mfcc(audio, sr=22050)
        assert result.dtype == np.float32

    def test_different_sample_rates(self):
        for sr in (8000, 16000, 22050):
            audio = _sine_wave(1.0, sr=sr)
            result = extract_mfcc(audio, sr=sr)
            assert result.shape == (N_MFCC, T_MAX)


def librosa_frame_count(n_samples: int, hop_length: int = 512) -> int:
    """Expected number of frames librosa produces for n_samples."""
    import librosa
    return librosa.feature.mfcc(
        y=_sine_wave(n_samples / 22050, sr=22050)[:n_samples],
        sr=22050,
        n_mfcc=N_MFCC,
        hop_length=hop_length,
    ).shape[1]


class TestBuildFeatureMatrix:
    def _make_records(self, n: int = 6) -> list[dict]:
        labels = ["angry", "happy", "sad"]
        return [
            {"array": _sine_wave(1.0), "sr": 22050, "label": labels[i % len(labels)]}
            for i in range(n)
        ]

    def test_shapes(self):
        records = self._make_records(6)
        X, y, label_names = build_feature_matrix(records)
        assert X.shape == (6, N_MFCC, T_MAX)
        assert y.shape == (6,)

    def test_label_names_sorted(self):
        records = self._make_records(6)
        _, _, label_names = build_feature_matrix(records)
        assert label_names == sorted(label_names)

    def test_label_indices_match(self):
        records = self._make_records(6)
        _, y, label_names = build_feature_matrix(records)
        assert set(y.tolist()).issubset(set(range(len(label_names))))

    def test_x_dtype(self):
        records = self._make_records(4)
        X, _, _ = build_feature_matrix(records)
        assert X.dtype == np.float32

    def test_y_dtype(self):
        records = self._make_records(4)
        _, y, _ = build_feature_matrix(records)
        assert y.dtype == np.int64

    def test_single_class(self):
        records = [{"array": _sine_wave(1.0), "sr": 22050, "label": "neutral"} for _ in range(3)]
        X, y, label_names = build_feature_matrix(records)
        assert label_names == ["neutral"]
        assert np.all(y == 0)


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------

from baselines.model import EmotionCNN


class TestEmotionCNN:
    def test_default_output_shape(self):
        model = EmotionCNN(n_mfcc=N_MFCC, n_classes=8)
        x = torch.randn(4, N_MFCC, T_MAX)
        out = model(x)
        assert out.shape == (4, 8)

    def test_custom_n_classes(self):
        model = EmotionCNN(n_mfcc=N_MFCC, n_classes=3)
        x = torch.randn(2, N_MFCC, T_MAX)
        out = model(x)
        assert out.shape == (2, 3)

    def test_single_sample(self):
        model = EmotionCNN(n_mfcc=N_MFCC, n_classes=5)
        x = torch.randn(1, N_MFCC, T_MAX)
        out = model(x)
        assert out.shape == (1, 5)

    def test_output_is_logits_not_probabilities(self):
        # CrossEntropyLoss expects raw logits — values should not be in [0,1] sum-to-1
        model = EmotionCNN(n_mfcc=N_MFCC, n_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(8, N_MFCC, T_MAX))
        row_sums = torch.softmax(out, dim=1).sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-5)

    def test_eval_mode_does_not_crash(self):
        model = EmotionCNN().eval()
        with torch.no_grad():
            out = model(torch.randn(2, N_MFCC, T_MAX))
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# train.py — minimal smoke test (tiny data, 1 epoch, CPU)
# ---------------------------------------------------------------------------

from baselines.train import run


class TestTrainRun:
    def test_smoke_runs_without_error(self, tmp_path, monkeypatch):
        """run() should complete without raising on tiny synthetic data."""
        # Redirect output dirs to a temp location so nothing is written to the repo
        import baselines.train as train_mod
        monkeypatch.setattr(train_mod, "RESULTS_DIR", str(tmp_path / "results"))
        monkeypatch.setattr(train_mod, "MODELS_DIR", str(tmp_path / "models"))

        n_classes = 3
        N = 30  # 10 samples per class
        label_names = ["angry", "happy", "sad"]
        rng = np.random.default_rng(0)
        X = rng.random((N, N_MFCC, T_MAX), dtype=np.float32)
        y = np.array([i % n_classes for i in range(N)], dtype=np.int64)

        run(X, y, label_names, epochs=1, batch_size=8)

    def test_smoke_saves_artifacts(self, tmp_path, monkeypatch):
        import baselines.train as train_mod
        results_dir = tmp_path / "results"
        models_dir = tmp_path / "models"
        monkeypatch.setattr(train_mod, "RESULTS_DIR", str(results_dir))
        monkeypatch.setattr(train_mod, "MODELS_DIR", str(models_dir))

        n_classes = 2
        N = 20
        rng = np.random.default_rng(1)
        X = rng.random((N, N_MFCC, T_MAX), dtype=np.float32)
        y = np.array([i % n_classes for i in range(N)], dtype=np.int64)

        run(X, y, ["neg", "pos"], epochs=1, batch_size=8)

        assert (results_dir / "report.txt").exists()
        assert (results_dir / "confusion_matrix.png").exists()
        assert (models_dir / "mfcc_cnn.pt").exists()
