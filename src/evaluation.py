"""
evaluation.py - Model Performance Assessment and Metrics Computation
Computes genuine accuracy, precision, recall, F1-score, and confusion matrix.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np


def evaluate_dataset_predictions(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Computes genuine precision, recall, f1-score, accuracy, and confusion matrix
    from ground truth and predicted labels.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    per_class_metrics = {}
    for i, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i]) if i < len(per_class_supp) else 0
        }

    return {
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100.0, 2),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "total_test_samples": len(y_true),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }


def evaluate_model_on_dataset(
    model: Any,
    test_ds: Any,
    class_names: List[str]
) -> Dict[str, Any]:
    """Runs batch predictions on a test dataset and evaluates metrics.
    Requires TensorFlow (only used during local training)."""
    y_true_all = []
    y_pred_all = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_batch = np.argmax(preds, axis=1)
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            y_true_batch = np.argmax(labels.numpy(), axis=1)
        else:
            y_true_batch = labels.numpy()

        y_true_all.extend(y_true_batch.tolist())
        y_pred_all.extend(y_pred_batch.tolist())

    return evaluate_dataset_predictions(y_true_all, y_pred_all, class_names)


def plot_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = "models/confusion_matrix.png"
):
    """Generates an annotated confusion matrix plot and optionally saves it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    # Shorter labels for cleaner visual
    short_labels = [c.replace("___", " - ").replace("_", " ") for c in class_names]

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=short_labels,
        yticklabels=short_labels,
        title="Crop Disease Classification Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7
            )

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def save_evaluation_report(
    report: Dict[str, Any],
    save_path: str = "models/evaluation_metrics.json"
) -> str:
    """Saves metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return save_path


def load_evaluation_report(
    report_path: str = "models/evaluation_metrics.json"
) -> Optional[Dict[str, Any]]:
    """Loads metrics dictionary if file exists."""
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None
