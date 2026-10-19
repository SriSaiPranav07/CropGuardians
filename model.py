"""
model.py - Backward Compatibility Shim
Provides legacy predict_image function expected by legacy Flask routes.
"""

import io
from PIL import Image
from src.model_utils import (
    load_model_and_classes,
    preprocess_image,
    predict_leaf_disease
)
from src.disease_info import get_all_supported_classes

_CACHED_MODEL = None
_CACHED_CLASSES = None


def get_or_load_legacy_model():
    global _CACHED_MODEL, _CACHED_CLASSES
    if _CACHED_MODEL is None:
        model, classes = load_model_and_classes()
        if model is not None and classes:
            _CACHED_MODEL = model
            _CACHED_CLASSES = classes
        else:
            _CACHED_CLASSES = get_all_supported_classes()
    return _CACHED_MODEL, _CACHED_CLASSES


def predict_image(img_bytes: bytes) -> int:
    """Legacy interface: accepts raw image bytes, returns predicted class index."""
    model, classes = get_or_load_legacy_model()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = preprocess_image(img)

    if model is not None:
        res = predict_leaf_disease(model, tensor, classes)
        pred_class = res["class_name"]
        try:
            return classes.index(pred_class)
        except ValueError:
            return 0
    return 0
