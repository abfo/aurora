# Aurora Wake Word

A small, **fully local** wake word detector for the word **"Aurora"**. It replaces
the previous Picovoice Porcupine dependency. You own the whole pipeline: there is
no API key, no usage tier, and nothing that can be discontinued out from under you.

- **Frontend:** pure-numpy log-mel spectrogram ([features.py](features.py))
- **Model:** a tiny CNN ([model.py](model.py)) trained in PyTorch, exported to ONNX
- **Runtime:** [`WakeWordDetector`](detector.py) runs the ONNX model with
  `onnxruntime` + `numpy` only - works on the Windows laptop and the Raspberry Pi 4
- **Drop-in:** the detector mimics Porcupine's interface, so `main.py` is unchanged
  apart from how it constructs the detector.

The detector deliberately favours **false positives over false negatives**: a missed
"Aurora" is worse than an occasional spurious wake, because the OpenAI realtime
session closes quickly if it doesn't hear a real conversation. It is also trained
to **reject "Alexa"** (a hard-negative class), so the household's Alexa devices
don't set it off.

## How it works

```
mic 16 kHz mono int16
  -> ring buffer (~1.2 s)            detector.py
  -> log-mel spectrogram             features.py   (SAME code used in training)
  -> CNN (aurora.onnx) -> probability
  -> smoothing + threshold + refractory
  -> fires -> main.py starts the realtime conversation
```

`features.py` is shared by training and inference, which guarantees the model sees
identical features in both - the most common source of "works in training, fails
live" bugs.

## Runtime requirements

Already in the top-level `requirements.txt`: `numpy`, `onnxruntime`. The trained
model (`models/aurora.onnx` + `models/aurora.json`) is committed, so a fresh
checkout works with no extra steps. On the Raspberry Pi 4 (64-bit OS) `pip install
onnxruntime` provides an aarch64 wheel.

## Improving / retraining the model

> Training needs extra packages: `pip install -r wake_word/requirements-train.txt`
> (PyTorch etc. - install these on your PC or in Colab, **not** on the Pi).

### 1. Add data

Audio lives under `wake_word/data/` (git-ignored), organized by label:

| Folder            | What goes in it                                   | Label |
|-------------------|---------------------------------------------------|-------|
| `positives/`      | people saying "Aurora" / "Hey Aurora"             | wake  |
| `hard_negatives/` | "Alexa" + similar-sounding words (aura, Laura...) | not   |
| `negatives/`      | normal speech / household chatter / silence       | not   |
| `background/`     | noise / music / room tone (used for augmentation) | -     |

Any sample rate is fine (files are resampled to 16 kHz mono on load). Name files
`<speaker>_<n>.wav` (e.g. `kate_001.wav`) - the `<speaker>` prefix groups the
train/validation split so we measure how well the model generalizes across voices.

**Record real samples** (do this for each of the four household members - real
voices matter most):

```bash
python wake_word/scripts/record_samples.py --speaker kate --label positives --count 15
python wake_word/scripts/record_samples.py --speaker kate --label hard_negatives --count 8
```

**Bootstrap with synthetic voices** (OpenAI TTS, uses your existing `OPENAI_API_KEY`;
generates many voices of "Aurora", "Alexa" and general speech):

```bash
python wake_word/scripts/generate_tts.py --dry-run   # preview the plan + counts
python wake_word/scripts/generate_tts.py             # actually generate
```

You can also drop in public-domain noise/speech clips for `negatives/` and
`background/` to make the model more robust.

### 2. Train (locally, a few minutes on CPU)

```bash
python wake_word/scripts/train.py
```

This builds the dataset (with augmentation: gain, time-shift, background-noise
mixing), trains the CNN, picks a detection threshold that favours recall while
keeping the false-alarm rate low, and writes:

- `models/aurora.onnx` - the model the detector loads
- `models/aurora.json` - threshold, feature params, and validation metrics

Useful flags: `--epochs`, `--augment-factor`, `--max-fa-rate` (raise it to allow
more false positives / fewer misses), `--data-dir`.

### 3. Check quality

```bash
python wake_word/scripts/evaluate.py
```

Prints a recall / false-alarm threshold sweep and a per-category breakdown,
including how reliably **"Alexa" is rejected**. Confirm positives trigger and
hard-negatives don't.

### 4. Deploy

Commit the updated `models/aurora.onnx` and `models/aurora.json`, then `git pull`
on the Raspberry Pi. That's it - no retraining on the Pi.

## Training on Colab (optional)

For larger datasets, [`train_aurora.ipynb`](train_aurora.ipynb) mirrors `train.py`.
Upload a zip of your `data/` folder, run all cells, and download `aurora.onnx` /
`aurora.json` into `models/`. The notebook is tested locally (headless) before use.

## Tuning sensitivity

- Lower `WAKE_WORD_THRESHOLD` in `.env` (or retrain with a higher `--max-fa-rate`)
  to catch more wakes at the cost of more false positives.
- Detection smoothing / refractory behaviour lives in [config.py](config.py)
  (`SMOOTHING_WINDOW`, `TRIGGER_CONSECUTIVE`, `REFRACTORY_SECONDS`).
- If you change any feature constant in `config.py`, you must retrain.
