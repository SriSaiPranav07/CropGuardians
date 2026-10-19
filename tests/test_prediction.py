"""
test_prediction.py - Unit Tests for Prediction Logic, Model Loading, and Disease Info Lookup
"""

import os
import pytest
import numpy as np
from src.disease_info import (
    get_disease_info,
    get_all_supported_classes,
    DISEASE_REGISTRY,
    DEFAULT_UNKNOWN_DISEASE
)
from src.model_utils import (
    load_model_and_classes,
    predict_leaf_disease,
    CONFIDENCE_THRESHOLD
)


def test_disease_info_retrieval_supported_classes():
    classes = get_all_supported_classes()
    assert len(classes) >= 10, "Must support at least 10 classes as per Zelvora requirements"

    for class_id in classes:
        info = get_disease_info(class_id)
        assert "crop" in info
        assert "disease" in info
        assert "symptoms" in info and len(info["symptoms"]) > 0
        assert "treatment" in info and len(info["treatment"]) > 0
        assert "prevention" in info and len(info["prevention"]) > 0
        assert "growth_care" in info
        assert isinstance(info["is_healthy"], bool)


def test_disease_info_fallback_unknown():
    info = get_disease_info("Completely_Unknown_NonExistent_Class")
    assert info["crop"] == DEFAULT_UNKNOWN_DISEASE["crop"]
    assert "Unidentified" in info["disease"]


def test_missing_model_files_handled_gracefully():
    # Attempting to load non-existent model files should return (None, []) without crashing
    model, classes = load_model_and_classes("non_existent_model.keras", "non_existent_classes.json")
    assert model is None
    assert classes == []


def test_prediction_output_structure_and_confidence():
    class_names = ["Apple___Apple_scab", "Corn___healthy", "Tomato___Late_blight"]

    # Mock model predicting class 0 with 0.85 confidence
    class MockModel:
        def predict(self, tensor, verbose=0):
            return np.array([[0.85, 0.10, 0.05]], dtype=np.float32)

    mock_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_leaf_disease(MockModel(), mock_tensor, class_names, confidence_threshold=CONFIDENCE_THRESHOLD)

    assert res["class_name"] == "Apple___Apple_scab"
    assert res["confidence"] == pytest.approx(0.85, rel=1e-3)
    assert res["confidence_pct"] == 85.0
    assert res["is_low_confidence"] is False
    assert len(res["top_candidates"]) == 3
    assert res["top_candidates"][0]["class_name"] == "Apple___Apple_scab"
    assert res["latency_ms"] >= 0.0


def test_low_confidence_flag():
    class_names = ["Apple___Apple_scab", "Corn___healthy", "Tomato___Late_blight"]

    # Mock model with low confidence (0.42 < 0.60 threshold)
    class LowConfMockModel:
        def predict(self, tensor, verbose=0):
            return np.array([[0.42, 0.38, 0.20]], dtype=np.float32)

    mock_tensor = np.zeros((1, 224, 224, 3), dtype=np.float32)
    res = predict_leaf_disease(LowConfMockModel(), mock_tensor, class_names, confidence_threshold=0.60)

    assert res["confidence_pct"] == 42.0
    assert res["is_low_confidence"] is True


def test_backward_compatibility_shims():
    import utils
    import model

    assert hasattr(utils, "disease_dic")
    assert len(utils.disease_dic) > 0

    # Ensure predict_image runs with fallback
    fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x20\x00\x00\x00\x20\x08\x02\x00\x00\x00\xfc\x18\xed\xa3\x00\x00\x00\x19IDATx^\xed\xc1\x01\x01\x00\x00\x00\x82 \xff\xafnH@\x01\x00\x00\x00\x00IEND\xaeB`\x82"
    pred_idx = model.predict_image(fake_png)
    assert isinstance(pred_idx, int)
