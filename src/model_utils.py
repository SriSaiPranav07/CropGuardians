"""
model_utils.py - Lightweight MobileNetV2 Architecture, Preprocessing, and Inference
Implements MobileNetV2 transfer learning optimized for edge / mobile-browser serving.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import time
import json
from typing import List, Tuple, Dict, Any, Union, Optional
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models, optimizers

IMG_SIZE: Tuple[int, int] = (224, 224)
INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)
CONFIDENCE_THRESHOLD: float = 0.60


def build_mobilenetv2_model(
    num_classes: int,
    alpha: float = 0.35,
    learning_rate: float = 1e-4,
    freeze_base: bool = True
) -> tf.keras.Model:
    """
    Constructs a lightweight MobileNetV2 transfer learning model.
    alpha=0.35 yields a highly compact architecture (~1.7M parameters, ~7MB)
    meeting Zelvora mobile-browser and Streamlit Community Cloud resource constraints.
    """
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights="imagenet",
        alpha=alpha
    )
    base_model.trainable = not freeze_base

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="leaf_image_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(0.30, name="head_dropout1")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.20, name="head_dropout2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"CropGuardians_MobileNetV2_a{alpha}")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model


def validate_image_file(image_source: Union[str, bytes, io.BytesIO, Image.Image]) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded file format and minimum dimensions.
    Returns (is_valid, error_message, PIL.Image).
    """
    try:
        if isinstance(image_source, Image.Image):
            img = image_source.copy()
        elif isinstance(image_source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(image_source))
        elif isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            img = Image.open(image_source)
        elif isinstance(image_source, str):
            if not os.path.exists(image_source):
                return False, f"Image path '{image_source}' does not exist.", None
            img = Image.open(image_source)
        else:
            return False, "Unsupported image source type.", None

        # Verify format
        if img.format and img.format.upper() not in ["JPEG", "JPG", "PNG", "WEBP"]:
            return False, f"Unsupported format '{img.format}'. Please upload a JPG, JPEG, or PNG image.", None

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Check dimensions (minimum 32x32)
        width, height = img.size
        if width < 32 or height < 32:
            return False, f"Image resolution ({width}x{height}) is too small. Minimum resolution is 32x32 pixels.", None

        return True, None, img
    except Exception as e:
        return False, f"Unable to parse image file: {str(e)}", None


def preprocess_image(
    image_source: Union[str, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = IMG_SIZE
) -> np.ndarray:
    """
    Loads, resizes, and pre-processes an input leaf image for MobileNetV2.
    Applies official tf.keras.applications.mobilenet_v2.preprocess_input (scales to [-1, 1]).
    Returns batch-dimension tensor of shape (1, target_size[0], target_size[1], 3).
    """
    is_valid, err_msg, img = validate_image_file(image_source)
    if not is_valid or img is None:
        raise ValueError(err_msg or "Invalid image file")

    # Resize with high quality bilinear/bicubic resampling
    img = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocess_input expects RGB in range [0, 255] and maps to [-1, 1]
    img_preprocessed = preprocess_input(img_array)

    # Add batch dimension: (1, 224, 224, 3)
    return np.expand_dims(img_preprocessed, axis=0)


def predict_leaf_disease(
    model: tf.keras.Model,
    preprocessed_tensor: np.ndarray,
    class_names: List[str],
    confidence_threshold: float = CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """
    Performs inference and latency measurement on a preprocessed leaf tensor.
    Returns predicted class, confidence percentage, top-3 candidates, and low-confidence alert flag.
    """
    start_time = time.perf_counter()
    raw_preds = model.predict(preprocessed_tensor, verbose=0)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    pred_idx = int(np.argmax(raw_preds))
    confidence = float(raw_preds[pred_idx])
    pred_class = class_names[pred_idx] if pred_idx < len(class_names) else "Unknown"

    # Sorted top-3 candidate probabilities
    sorted_indices = np.argsort(raw_preds)[::-1]
    top_candidates = []
    for idx in sorted_indices[:min(3, len(class_names))]:
        top_candidates.append({
            "class_name": class_names[idx],
            "probability": float(raw_preds[idx]),
            "percentage": round(float(raw_preds[idx]) * 100.0, 2)
        })

    is_low_confidence = bool(confidence < confidence_threshold)

    return {
        "class_name": pred_class,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100.0, 2),
        "is_low_confidence": is_low_confidence,
        "latency_ms": round(latency_ms, 1),
        "top_candidates": top_candidates,
        "all_probabilities": [float(p) for p in raw_preds]
    }


def save_trained_artifacts(
    model: tf.keras.Model,
    class_names: List[str],
    model_dir: str = "models",
    model_filename: str = "mobilenetv2_cropguardians.keras",
    classes_filename: str = "class_names.json"
) -> Tuple[str, str]:
    """Saves model in .keras format and classes in JSON order."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    classes_path = os.path.join(model_dir, classes_filename)

    model.save(model_path)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    return model_path, classes_path


def load_model_and_classes(
    model_path: str = "models/mobilenetv2_cropguardians.keras",
    classes_path: str = "models/class_names.json"
) -> Tuple[Optional[tf.keras.Model], List[str]]:
    """
    Safely loads trained .keras model and associated class list.
    Returns (model, class_names). Handles missing files gracefully.
    """
    if not os.path.exists(model_path):
        return None, []

    if not os.path.exists(classes_path):
        return None, []

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path, compile=False)
    return model, class_names


def get_model_file_size_mb(model_path: str) -> float:
    """Returns size of model file on disk in Megabytes."""
    if os.path.exists(model_path):
        return round(os.path.getsize(model_path) / (1024 * 1024), 2)
    return 0.0
