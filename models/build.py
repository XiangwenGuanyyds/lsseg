# models/build.py
from utils import MODELS


def build_model(cfg):
    """Build and move the configured model to its target device."""
    model_type = cfg.MODEL["type"]
    model_cls = MODELS.get(model_type)
    if model_cls is None:
        raise KeyError(
            f"Unknown model '{model_type}'. Available models: {MODELS.keys()}"
        )
    return model_cls(cfg).to(cfg.DEVICE)
