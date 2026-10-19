"""
test_preprocessing.py - Unit Tests for Image Preprocessing and Validation
"""

import io
import pytest
import numpy as np
from PIL import Image
from src.model_utils import (
    validate_image_file,
    preprocess_image,
    IMG_SIZE
)


def test_validate_valid_rgb_image():
    # Create valid synthetic RGB image
    img = Image.new("RGB", (300, 300), color=(50, 150, 50))
    is_valid, err_msg, loaded_img = validate_image_file(img)
    assert is_valid is True
    assert err_msg is None
    assert loaded_img is not None
    assert loaded_img.size == (300, 300)


def test_validate_bytes_image():
    img = Image.new("RGB", (100, 100), color=(100, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    is_valid, err_msg, loaded_img = validate_image_file(buf.getvalue())
    assert is_valid is True
    assert err_msg is None
    assert loaded_img is not None


def test_validate_too_small_image():
    # Under minimum 32x32 dimensions
    tiny_img = Image.new("RGB", (20, 20), color=(0, 0, 0))
    is_valid, err_msg, _ = validate_image_file(tiny_img)
    assert is_valid is False
    assert "too small" in err_msg.lower()


def test_validate_invalid_data():
    is_valid, err_msg, _ = validate_image_file(b"not an image file")
    assert is_valid is False
    assert err_msg is not None


def test_preprocess_image_output_shape_and_range():
    img = Image.new("RGB", (400, 350), color=(128, 200, 64))
    tensor = preprocess_image(img, target_size=IMG_SIZE)

    # Check batch dimension: (1, 224, 224, 3)
    assert tensor.shape == (1, 224, 224, 3)

    # MobileNetV2 preprocess_input maps pixel values [0, 255] to [-1.0, 1.0]
    assert np.all(tensor >= -1.05)
    assert np.all(tensor <= 1.05)


def test_preprocess_raises_on_invalid_input():
    with pytest.raises(ValueError):
        preprocess_image(b"corrupt bytes")
