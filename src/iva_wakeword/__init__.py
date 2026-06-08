"""Wake-word models (microWakeWord, quantized .tflite + .json) for the Iva
voice assistant. The device app (`iva-hermes`) depends on this package and loads
every ``*.json`` in ``models_dir()`` at runtime; "Okay Iva" is the active model.

The training pipeline that produces these lives in this repo's ``training/``
(not shipped in the wheel)."""
import os

__version__ = "0.1.0"


def models_dir() -> str:
    """Absolute path to the bundled wake-word models directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
