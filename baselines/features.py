import numpy as np
import librosa

N_MFCC = 40
T_MAX = 130  # ~3 s at sr=22050, hop_length=512
HOP_LENGTH = 512


def extract_mfcc(array: np.ndarray, sr: int) -> np.ndarray:
    """Extract a fixed-length MFCC matrix from a single audio sample.

    Returns shape (N_MFCC, T_MAX), padded or truncated as needed.
    """
    mfccs = librosa.feature.mfcc(
        y=array.astype(np.float32), sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    if mfccs.shape[1] < T_MAX:
        mfccs = np.pad(mfccs, ((0, 0), (0, T_MAX - mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :T_MAX]
    return mfccs


def build_feature_matrix(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build X, y arrays from a list of audio records.

    Each record must have keys: "array" (np.ndarray), "sr" (int), "label" (str).

    Returns:
        X: float32 array of shape (N, N_MFCC, T_MAX)
        y: int64 label indices of shape (N,)
        label_names: sorted list of class name strings
    """
    label_names = sorted({r["label"] for r in records})
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    X = np.stack(
        [extract_mfcc(r["array"], r["sr"]) for r in records]
    ).astype(np.float32)
    y = np.array([label_to_idx[r["label"]] for r in records], dtype=np.int64)

    return X, y, label_names
