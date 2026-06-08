# iva-wakeword

The **wake-word training project** for [Iva](https://github.com/cloudomate/iva-hermes)
— the on-device "Okay Iva" voice assistant. It holds the training pipeline for
custom **microWakeWord** models and the trained model artifacts it produces.

This is dev-time work, run on a Mac with
[OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word). It is
**not** runtime code: the *runtime* that loads + scores wake words on the device
is the `pymicro-wakeword` package, and the active models ship bundled inside the
device app (`cloudomate/iva-hermes`). This repo is where you **create new
models**; the produced `.tflite` + `.json` are then copied into iva-hermes.

```
wakewords/                 # trained model artifacts (quantized .tflite + .json)
  okay_iva.{tflite,json}   #   the ACTIVE wake word
  hey_iva.{tflite,json}
  hey_hermes.{tflite,json}
training/                  # the training pipeline (run on a Mac)
  TRAINING.md              #   full pipeline + the empirical wake-word design lessons
  prep_*.py                #   dataset prep (synthetic Piper + real recordings)
  record_wakeword.py       #   capture real wake-word samples
  training_parameters_*.yaml
  train.sh
```

## Design lessons (the short version)

See `training/TRAINING.md` for the full pipeline and the hard-won lessons —
notably: syllable count dominates accuracy, and synthetic Piper audio often
mismatches the real speaker, so train on **real recordings** of the target voice.

## Shipping a new model to the device

After training, the model is consumed by the device app, not installed from
here. To roll a new wake word out:

1. Train it here → `wakewords/<name>.{tflite,json}`.
2. Copy those two files into `iva-hermes` at `src/iva/data/wakewords/` (they ship
   as package data) and release a new `iva-hermes`; **or** drop them into the
   device's `WAKE_MODELS_DIR` (default `/home/iva/wakewords/`) directly.

The daemon auto-loads every `*.json` in `WAKE_MODELS_DIR`; "Okay Iva" is active.
