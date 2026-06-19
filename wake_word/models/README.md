# Trained models

The wake word detector loads `aurora.onnx` (and `aurora.json` for the threshold +
feature metadata) from this folder.

These files are produced by `wake_word/scripts/train.py` (or the Colab notebook) and
are committed to the repo so a checkout works without retraining. If they are not
present yet, train a model first:

```bash
pip install -r wake_word/requirements-train.txt
python wake_word/scripts/generate_tts.py     # bootstrap data (uses OPENAI_API_KEY)
python wake_word/scripts/train.py
```

See [../README.md](../README.md) for the full workflow.
