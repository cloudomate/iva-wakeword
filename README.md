# iva-wakeword

Custom **microWakeWord** models for [Iva](https://github.com/cloudomate/iva-hermes)
— the on-device "Okay Iva" voice assistant — plus the training pipeline that
produces them.

This is dev-time work (run on a Mac with
[OHF-Voice/micro-wake-word](https://github.com/OHF-Voice/micro-wake-word)); the
quantized `.tflite` + `.json` outputs are **consumed at runtime** by the device
app (`cloudomate/iva-hermes`), which loads every `*.json` in the device's
`WAKE_MODELS_DIR` (default `/home/iva/wakewords/`).

This is a pip package — `pip install iva-wakeword` — that ships the models as
package data. The device app finds them via `iva_wakeword.models_dir()`:

```python
from iva_wakeword import models_dir
print(models_dir())   # -> .../iva_wakeword/models  (contains *.tflite + *.json)
```

```
src/iva_wakeword/models/   # trained models (quantized .tflite + .json), shipped as package data
  okay_iva.{tflite,json}   #   the ACTIVE wake word
  hey_iva.{tflite,json}
  hey_hermes.{tflite,json}
training/                  # the training pipeline (run on a Mac; NOT shipped in the wheel)
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

## Deploying models to a device

The device app pulls these models in `cloudomate/iva-hermes`'s
`deploy/install-device.sh` (copies `wakewords/*` to the device's
`WAKE_MODELS_DIR`). After training a new model, commit it here and re-run that
installer (or copy the `.tflite` + `.json` to `/home/iva/wakewords/`).
