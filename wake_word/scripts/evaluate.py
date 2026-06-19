"""Evaluate the exported Aurora wake word model.

Runs the ONNX model over the audio in wake_word/data (or a held-out --data-dir),
and reports:
    - overall recall (true-positive rate) and false-alarm rate at the model threshold
    - a threshold sweep so you can see the recall / false-alarm trade-off
    - per-category breakdown, including how well "Alexa" is rejected

Usage::

    python wake_word/scripts/evaluate.py
    python wake_word/scripts/evaluate.py --data-dir path/to/heldout
"""

from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np

from prepare_dataset import DATA_DIR, build_dataset, to_arrays
from wake_word import config
from wake_word.detector import WakeWordDetector  # for threshold metadata reuse

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Aurora wake word model.")
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--model", default=os.path.join(MODELS_DIR, "aurora.onnx"))
    args = ap.parse_args()

    import onnxruntime as ort

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train it first (train.py).")

    # Reuse the detector only to read the saved threshold from metadata.
    detector = WakeWordDetector(args.model)
    threshold = detector.threshold

    print("Building evaluation set (no augmentation)...")
    samples = build_dataset(args.data_dir, augment=False)
    if not samples:
        raise SystemExit("No data found.")
    X, y, _groups, categories = to_arrays(samples)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    probs = sess.run(None, {name: X})[0].reshape(-1)

    pos = probs[y == 1]
    neg = probs[y == 0]

    print(f"\nModel: {args.model}")
    print(f"Threshold (from metadata): {threshold}")
    print(f"Samples: {len(y)} ({len(pos)} positive, {len(neg)} negative)\n")

    print("Threshold sweep:")
    print(f"  {'thr':>5}  {'recall':>7}  {'FA rate':>8}")
    for thr in np.linspace(0.1, 0.9, 9):
        recall = float(np.mean(pos >= thr)) if len(pos) else 0.0
        fa = float(np.mean(neg >= thr)) if len(neg) else 0.0
        marker = "  <- threshold" if abs(thr - threshold) < 0.05 else ""
        print(f"  {thr:5.2f}  {recall:7.3f}  {fa:8.3f}{marker}")

    recall = float(np.mean(pos >= threshold)) if len(pos) else 0.0
    fa = float(np.mean(neg >= threshold)) if len(neg) else 0.0
    print(f"\nAt threshold {threshold}:  recall={recall:.3f}  false-alarm rate={fa:.3f}")

    print("\nPer-category mean probability and trigger rate:")
    for cat in sorted(set(categories.tolist())):
        mask = categories == cat
        cat_probs = probs[mask]
        trig = float(np.mean(cat_probs >= threshold))
        label = "PASS" if (cat == "positives" and trig > 0.5) or (cat != "positives" and trig < 0.5) else "CHECK"
        kind = "should trigger" if cat == "positives" else "should reject"
        print(f"  {cat:16s} n={mask.sum():4d}  mean_p={cat_probs.mean():.3f}  trigger_rate={trig:.3f}  ({kind}) [{label}]")

    # Spotlight the Alexa rejection requirement.
    alexa_mask = np.array(["alexa" in c.lower() for c in categories])  # category is dir name
    if not alexa_mask.any():
        # hard_negatives holds Alexa; report it explicitly if present.
        hn = categories == "hard_negatives"
        if hn.any():
            trig = float(np.mean(probs[hn] >= threshold))
            print(f"\nHard-negative (incl. 'Alexa') trigger rate: {trig:.3f}  (lower is better)")


if __name__ == "__main__":
    main()
