"""
data_loader.py - Dataset Ingestion, Deduplication, Stratification, and Augmentation
Handles PlantVillage dataset loading with stratified splits and data leakage prevention.
"""

import os
import hashlib
from typing import List, Tuple, Dict, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image

SUPPORTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy"
]


def create_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Creates lightweight real-time data augmentation layers for leaf images.
    Mimics natural variations in orientation, solar exposure, and distance.
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="aug_random_flip"),
        layers.RandomRotation(0.15, fill_mode="reflect", name="aug_random_rot"),
        layers.RandomZoom(0.12, fill_mode="reflect", name="aug_random_zoom"),
        layers.RandomContrast(0.10, name="aug_random_contrast"),
    ], name="leaf_augmentation_pipeline")


def compute_file_hash(filepath: str) -> str:
    """Computes MD5 hash of image content to detect duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_dataset_directory(
    root_dir: str,
    selected_classes: Optional[List[str]] = None,
    deduplicate: bool = True
) -> Tuple[List[str], List[int], List[str]]:
    """
    Scans directory structure for class folders.
    Optionally deduplicates files by content hash to prevent train/test leakage.
    Returns (file_paths, labels, class_names).
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset root directory '{root_dir}' does not exist.")

    available_folders = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ]

    if selected_classes:
        class_names = sorted([c for c in available_folders if c in selected_classes])
    else:
        class_names = sorted(available_folders)

    if not class_names:
        raise ValueError(f"No valid class folders found in '{root_dir}'.")

    file_paths: List[str] = []
    labels: List[int] = []
    seen_hashes = set()
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext in valid_exts:
                fpath = os.path.join(class_folder, fname)
                if deduplicate:
                    try:
                        fhash = compute_file_hash(fpath)
                        if fhash in seen_hashes:
                            continue  # Skip duplicate to prevent split leakage
                        seen_hashes.add(fhash)
                    except Exception:
                        continue

                file_paths.append(fpath)
                labels.append(class_idx)

    return file_paths, labels, class_names


def stratified_split(
    file_paths: List[str],
    labels: List[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Performs stratified partitioning into Train, Validation, and Test sets
    preserving class representation in each partition.
    """
    np.random.seed(seed)
    class_to_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for label, idxs in class_to_indices.items():
        shuffled = np.random.permutation(idxs).tolist()
        n_total = len(shuffled)
        n_val = int(round(n_total * val_ratio))
        n_test = int(round(n_total * test_ratio))

        val_part = shuffled[:n_val]
        test_part = shuffled[n_val:n_val + n_test]
        train_part = shuffled[n_val + n_test:]

        # Guarantee at least 1 sample in train if class is tiny
        if not train_part and val_part:
            train_part.append(val_part.pop())

        val_indices.extend(val_part)
        test_indices.extend(test_part)
        train_indices.extend(train_part)

    def extract_subset(indices: List[int]) -> Tuple[List[str], List[int]]:
        return [file_paths[i] for i in indices], [labels[i] for i in indices]

    return {
        "train": extract_subset(train_indices),
        "val": extract_subset(val_indices),
        "test": extract_subset(test_indices),
    }


def parse_image_and_label(filepath: tf.Tensor, label: tf.Tensor, num_classes: int, target_size=(224, 224)):
    """Decodes, resizes, and applies MobileNetV2 preprocessing in tf.data pipeline."""
    img_bytes = tf.io.read_file(filepath)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    # Preprocess for MobileNetV2: scale from [0, 255] to [-1, 1]
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    one_hot_label = tf.one_hot(label, depth=num_classes)
    return img, one_hot_label


def build_tf_datasets(
    splits: Dict[str, Tuple[List[str], List[int]]],
    num_classes: int,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    apply_aug: bool = True
) -> Dict[str, tf.data.Dataset]:
    """
    Constructs high-performance tf.data pipelines with prefetching.
    """
    aug_layers = create_augmentation_pipeline() if apply_aug else None
    datasets = {}

    for split_name, (fpaths, labels_list) in splits.items():
        if not fpaths:
            continue

        ds = tf.data.Dataset.from_tensor_slices((fpaths, labels_list))

        if split_name == "train":
            ds = ds.shuffle(buffer_size=min(len(fpaths), 1000), seed=42)

        ds = ds.map(
            lambda x, y: parse_image_and_label(x, y, num_classes, target_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if split_name == "train" and aug_layers:
            ds = ds.map(
                lambda x, y: (aug_layers(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        datasets[split_name] = ds

    return datasets


def generate_synthetic_samples_if_empty(
    target_dir: str = "data/sample_leaves",
    samples_per_class: int = 5
) -> str:
    """
    Generates synthetic leaf images for local smoke testing and unit tests
    if no raw PlantVillage dataset has been downloaded yet.
    """
    os.makedirs(target_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    for cname in SUPPORTED_CLASSES[:10]:
        cdir = os.path.join(target_dir, cname)
        os.makedirs(cdir, exist_ok=True)
        if len(os.listdir(cdir)) < samples_per_class:
            for s in range(samples_per_class):
                # Synthesize a leaf-like image with natural green tones and texture variations
                arr = rng.integers(30, 80, size=(224, 224, 3), dtype=np.uint8)
                if "healthy" in cname.lower():
                    arr[:, :, 1] = rng.integers(140, 220, size=(224, 224), dtype=np.uint8) # Lush green
                else:
                    arr[:, :, 1] = rng.integers(90, 150, size=(224, 224), dtype=np.uint8)  # Dull green
                    # Add brownish or yellowish lesion spots
                    arr[60:120, 60:120, 0] = rng.integers(160, 210, size=(60, 60), dtype=np.uint8)
                    arr[60:120, 60:120, 1] = rng.integers(120, 160, size=(60, 60), dtype=np.uint8)

                img = Image.fromarray(arr)
                img.save(os.path.join(cdir, f"sample_{s+1}.jpg"), format="JPEG", quality=90)

    return target_dir
