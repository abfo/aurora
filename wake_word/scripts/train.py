"""Train the Aurora wake word CNN and export it to ONNX.

Runs locally on CPU in a few minutes. Usage::

    python wake_word/scripts/train.py

Outputs:
    wake_word/models/aurora.onnx   - probability-output model for the detector
    wake_word/models/aurora.json   - threshold + feature params + metrics

The decision threshold is chosen to favour recall (false positives are
preferable to false negatives for this application), subject to a cap on the
false-alarm rate measured on the negative validation set.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

import _bootstrap  # noqa: F401
import numpy as np

from prepare_dataset import DATA_DIR, build_dataset, to_arrays
from wake_word import config

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _grouped_split(groups: np.ndarray, y: np.ndarray, val_frac: float, seed: int):
    """Split indices by group so a speaker/source is never in both sets."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    # Try to keep both classes present in val; fall back to simple split.
    val_groups = set(uniq[:n_val].tolist())
    val_mask = np.array([g in val_groups for g in groups])
    if val_mask.all() or not val_mask.any() or len(np.unique(y[~val_mask])) < 2:
        # Degenerate (e.g. tiny dataset): fall back to a stratified random split.
        idx = np.arange(len(y))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - val_frac))
        train_idx, val_idx = idx[:cut], idx[cut:]
        return train_idx, val_idx
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def _choose_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    max_hard_neg_rate: float,
) -> float:
    """Most permissive threshold that still rejects the hard negatives ("Alexa").

    The requirements are asymmetric: random speech triggering the wake word is
    acceptable (the realtime session just closes), but the household's Alexa
    devices must NOT set it off. So we treat the hard-negative trigger rate as
    the binding constraint and otherwise maximize recall: pick the *lowest*
    threshold whose hard-negative trigger rate stays <= ``max_hard_neg_rate``.
    """
    pos = probs[labels == 1]
    if len(pos) == 0:
        return config.DEFAULT_THRESHOLD

    hard = probs[(labels == 0) & (categories == "hard_negatives")]
    candidates = np.linspace(0.05, 0.95, 19)

    feasible = [
        thr for thr in candidates
        if (len(hard) == 0 or float(np.mean(hard >= thr)) <= max_hard_neg_rate)
    ]
    if feasible:
        best = float(min(feasible))   # lowest feasible thr => max recall, Alexa still rejected
    else:
        best = 0.9                    # cannot reject Alexa; be as strict as possible
    # Keep a safety margin against garbage/near-silence triggers.
    return round(min(max(best, config.MIN_AUTO_THRESHOLD), 0.9), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Aurora wake word model.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--augment-factor", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-hard-negative-rate", type=float, default=config.MAX_HARD_NEGATIVE_RATE,
                    help="Max allowed trigger rate on hard negatives ('Alexa') when picking the threshold.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=MODELS_DIR)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from wake_word.model import WakeWordCNN, export_onnx

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Building dataset...")
    samples = build_dataset(args.data_dir, augment=True, augment_factor=args.augment_factor, seed=args.seed)
    if len(samples) < 4:
        raise SystemExit("Not enough data. Add wavs under wake_word/data/ (see README).")
    X, y, groups, categories = to_arrays(samples)
    n_pos = int(y.sum())
    print(f"  {len(y)} windows, {n_pos} positive, {len(y) - n_pos} negative")
    if n_pos == 0 or n_pos == len(y):
        raise SystemExit("Need both positive and negative samples to train.")

    train_idx, val_idx = _grouped_split(groups, y, args.val_frac, args.seed)
    print(f"  train={len(train_idx)} val={len(val_idx)}")

    def _fit(train_X, train_y, epochs, val_X=None, val_y=None):
        """Train a fresh model; with a val set, early-stop on best val loss."""
        loader = DataLoader(TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
        p = float(train_y.sum().item())
        n = float(len(train_y) - p)
        pos_weight = torch.tensor([n / p]) if p > 0 else torch.tensor([1.0])
        m = WakeWordCNN()
        opt = torch.optim.Adam(m.parameters(), lr=args.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_val, best_state, best_epoch = float("inf"), None, epochs - 1
        for epoch in range(epochs):
            m.train()
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(m(xb).squeeze(-1), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            train_loss = total / len(train_X)
            if val_X is not None:
                m.eval()
                with torch.no_grad():
                    val_loss = loss_fn(m(val_X).squeeze(-1), val_y).item()
                if val_loss < best_val:
                    best_val, best_epoch = val_loss, epoch
                    best_state = {k: v.clone() for k, v in m.state_dict().items()}
                if epoch % 5 == 0 or epoch == epochs - 1:
                    print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            elif epoch % 5 == 0 or epoch == epochs - 1:
                print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}")
        if best_state is not None:
            m.load_state_dict(best_state)
        return m, best_val, best_epoch

    Xtr, ytr = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    Xva, yva = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

    # Train on the train split with early stopping on val loss. We keep the
    # best-val checkpoint as the shipped model: refitting on all data (including
    # val) was tried and badly mis-calibrated the scores, so we don't do it.
    print("Training (early-stopped on held-out voices)...")
    model, best_val, _best_epoch = _fit(Xtr, ytr, args.epochs, Xva, yva)

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xva).squeeze(-1)).numpy()
    val_labels = y[val_idx]
    val_categories = categories[val_idx]
    threshold = _choose_threshold(val_probs, val_labels, val_categories, args.max_hard_negative_rate)

    pos_p = val_probs[val_labels == 1]
    neg_p = val_probs[val_labels == 0]
    hard_p = val_probs[(val_labels == 0) & (val_categories == "hard_negatives")]
    recall = float(np.mean(pos_p >= threshold)) if len(pos_p) else 0.0
    fa = float(np.mean(neg_p >= threshold)) if len(neg_p) else 0.0
    alexa_fa = float(np.mean(hard_p >= threshold)) if len(hard_p) else 0.0
    print(f"Chosen threshold={threshold}  (held-out val recall={recall:.3f}  "
          f"FA rate={fa:.3f}  Alexa/hard-neg rate={alexa_fa:.3f})")

    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, "aurora.onnx")
    export_onnx(model, onnx_path)

    metadata = {
        "version": 1,
        "trained_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "threshold": threshold,
        "features": {
            "sample_rate": config.SAMPLE_RATE,
            "window_samples": config.WINDOW_SAMPLES,
            "n_fft": config.N_FFT,
            "hop_length": config.HOP_LENGTH,
            "n_mels": config.N_MELS,
            "num_frames": config.NUM_FRAMES,
        },
        "metrics": {
            "val_recall": round(recall, 4),
            "val_false_alarm_rate": round(fa, 4),
            "val_hard_negative_rate": round(alexa_fa, 4),
            "val_loss": round(best_val, 4),
            "num_windows": len(y),
            "num_positive": n_pos,
        },
    }
    meta_path = os.path.join(args.out_dir, "aurora.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported {onnx_path}")
    print(f"Wrote    {meta_path}")


if __name__ == "__main__":
    main()
