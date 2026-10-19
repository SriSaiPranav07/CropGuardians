"""
train_model.py - MobileNetV2 Crop Disease Classifier Training Pipeline
Trains a lightweight transfer-learning CNN on PlantVillage dataset splits
with data augmentation, early stopping, and genuine performance reporting.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import sys

# Configure UTF-8 stdout on Windows if supported
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.data_loader import (
    scan_dataset_directory,
    stratified_split,
    build_tf_datasets,
    generate_synthetic_samples_if_empty,
    SUPPORTED_CLASSES
)
from src.model_utils import (
    build_mobilenetv2_model,
    save_trained_artifacts,
    get_model_file_size_mb
)
from src.evaluation import (
    evaluate_model_on_dataset,
    plot_confusion_matrix_figure,
    save_evaluation_report
)


def run_training(
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    alpha: float = 0.35,
    val_split: float = 0.15,
    test_split: float = 0.15,
    output_dir: str = "models",
    synthetic_fallback: bool = True
):
    print("=" * 70)
    print("[*] CropGuardians AI - MobileNetV2 Training Pipeline")
    print("=" * 70)

    # Verify or generate dataset directory
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        if synthetic_fallback:
            print(f"[!] Data directory '{data_dir}' is empty or missing.")
            print("[+] Generating verifiable sample dataset for local pipeline verification...")
            data_dir = generate_synthetic_samples_if_empty(target_dir=data_dir, samples_per_class=6)
        else:
            raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    print(f"\n[1/6] Scanning and deduplicating dataset in '{data_dir}'...")
    file_paths, labels, class_names = scan_dataset_directory(
        root_dir=data_dir,
        selected_classes=SUPPORTED_CLASSES,
        deduplicate=True
    )
    num_classes = len(class_names)
    print(f"[OK] Found {len(file_paths)} unique images across {num_classes} classes.")
    for idx, cname in enumerate(class_names):
        count = labels.count(idx)
        print(f"   [{idx:02d}] {cname}: {count} samples")

    if num_classes < 2:
        raise ValueError("At least 2 classes are required to train a classifier.")

    print(f"\n[2/6] Performing stratified split (Train={1-val_split-test_split:.0%}, Val={val_split:.0%}, Test={test_split:.0%})...")
    splits = stratified_split(file_paths, labels, val_ratio=val_split, test_ratio=test_split, seed=42)
    print(f"[OK] Train: {len(splits['train'][0])} images")
    print(f"[OK] Val:   {len(splits['val'][0])} images")
    print(f"[OK] Test:  {len(splits['test'][0])} images")

    print(f"\n[3/6] Building tf.data input pipelines with data augmentation...")
    tf_datasets = build_tf_datasets(
        splits=splits,
        num_classes=num_classes,
        batch_size=batch_size,
        target_size=(224, 224),
        apply_aug=True
    )

    print(f"\n[4/6] Initializing MobileNetV2 (alpha={alpha}, input=(224,224,3))...")
    model = build_mobilenetv2_model(
        num_classes=num_classes,
        alpha=alpha,
        learning_rate=learning_rate,
        freeze_base=True
    )
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"[OK] Model created: Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, "mobilenetv2_cropguardians.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n[5/6] Training model for up to {epochs} epochs (Batch size={batch_size})...")
    history = model.fit(
        tf_datasets["train"],
        validation_data=tf_datasets["val"],
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model and exact class ordering
    model_path, classes_path = save_trained_artifacts(
        model=model,
        class_names=class_names,
        model_dir=output_dir,
        model_filename="mobilenetv2_cropguardians.keras",
        classes_filename="class_names.json"
    )
    file_size_mb = get_model_file_size_mb(model_path)
    print(f"\n[OK] Saved model in .keras format to: {model_path} ({file_size_mb} MB)")
    print(f"[OK] Saved class names to: {classes_path}")

    print("\n[6/6] Computing genuine evaluation metrics on independent test partition...")
    if "test" in tf_datasets and len(splits["test"][0]) > 0:
        eval_metrics = evaluate_model_on_dataset(model, tf_datasets["test"], class_names)
        eval_metrics["model_architecture"] = f"MobileNetV2 (alpha={alpha})"
        eval_metrics["model_file_size_mb"] = file_size_mb
        eval_metrics["total_parameters"] = int(total_params)
        eval_metrics["trainable_parameters"] = int(trainable_params)
        eval_metrics["epochs_trained"] = len(history.history["loss"])
        eval_metrics["training_loss"] = [float(x) for x in history.history["loss"]]
        eval_metrics["val_loss"] = [float(x) for x in history.history["val_loss"]]
        eval_metrics["training_accuracy"] = [float(x) for x in history.history["accuracy"]]
        eval_metrics["val_accuracy"] = [float(x) for x in history.history["val_accuracy"]]

        report_json_path = os.path.join(output_dir, "evaluation_metrics.json")
        save_evaluation_report(eval_metrics, report_json_path)

        cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix_figure(np.array(eval_metrics["confusion_matrix"]), class_names, cm_plot_path)

        print("\n" + "=" * 50)
        print("[*] FINAL MEASURED EVALUATION REPORT")
        print("=" * 50)
        print(f"Overall Test Accuracy:    {eval_metrics['accuracy_pct']}%")
        print(f"Macro-Averaged Precision: {eval_metrics['macro_precision']}")
        print(f"Macro-Averaged Recall:    {eval_metrics['macro_recall']}")
        print(f"Macro-Averaged F1-Score:  {eval_metrics['macro_f1']}")
        print(f"Confusion matrix saved:   {cm_plot_path}")
        print(f"Metrics record saved:     {report_json_path}")
        print("=" * 50)
    else:
        print("[!] Test partition was empty; skipping independent test evaluation.")

    print("\n[OK] Training pipeline completed successfully.")
    return model, class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Crop Disease Classifier")
    parser.add_argument("--data-dir", type=str, default="data/sample_leaves", help="Path to PlantVillage dataset root directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 alpha width multiplier (0.35 for edge/mobile)")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        alpha=args.alpha,
        output_dir=args.output_dir
    )
