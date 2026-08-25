#!/usr/bin/env python3
"""
ResNet50 Feature Extraction Pipeline for NEW Fashion Dataset.

Processes ONLY:
  garment-processing-api/dataset/fashion_dataset/Images/

Outputs saved ONLY to:
  garment-processing-api/dataset/fashion_dataset/embeddings.npy
  garment-processing-api/dataset/fashion_dataset/image_ids.npy
"""

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image


def resolve_fashion_dataset_dir() -> Path:
    """Locates fashion_dataset directory containing metadata.csv and Images/."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset" / "fashion_dataset",
        script_dir.parent.parent / "dataset" / "fashion_dataset",
        script_dir.parent.parent / "garment-processing-api" / "dataset" / "fashion_dataset",
        Path.cwd() / "dataset" / "fashion_dataset",
        Path.cwd() / "garment-processing-api" / "dataset" / "fashion_dataset",
    ]
    for candidate in candidates:
        if (candidate / "metadata.csv").exists() and (candidate / "Images").exists():
            return candidate
    raise FileNotFoundError("Could not locate fashion_dataset directory with metadata.csv and Images/.")


def generate_embeddings():
    start_time = time.time()
    print("=" * 70)
    print("  RESNET50 FEATURE EXTRACTION PIPELINE - NEW STANDALONE GARMENT DATASET  ")
    print("=" * 70)

    dataset_dir = resolve_fashion_dataset_dir()
    metadata_csv_path = dataset_dir / "metadata.csv"
    images_dir = dataset_dir / "Images"
    embeddings_output_path = dataset_dir / "embeddings.npy"
    image_ids_output_path = dataset_dir / "image_ids.npy"

    print(f"[*] Dataset Directory    : {dataset_dir.resolve()}")
    print(f"[*] Metadata CSV         : {metadata_csv_path.resolve()}")
    print(f"[*] Images Directory     : {images_dir.resolve()}")
    print(f"[*] Output Embeddings    : {embeddings_output_path.resolve()}")
    print(f"[*] Output Image IDs     : {image_ids_output_path.resolve()}")

    # 1. Read metadata.csv
    df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    total_rows = len(df)
    print(f"[*] Total metadata rows  : {total_rows:,}")

    # 2. Build filename -> relative_path map for fast, exact lookup
    print("[*] Indexing image files in Images/ directory...")
    image_files_map = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                full_p = Path(root) / f
                rel_p = full_p.relative_to(images_dir)
                image_files_map[f] = (full_p, rel_p)

    print(f"[*] Total image files indexed: {len(image_files_map):,}")

    # 3. Match metadata rows to image paths deterministically
    ordered_items = []
    missing_items = []

    for idx, row in df.iterrows():
        item_id = str(row["id"]).strip()
        pname = str(row["product_name"]).strip()
        if pname in image_files_map:
            full_path, rel_path = image_files_map[pname]
            ordered_items.append({
                "row_index": idx,
                "id": item_id,
                "product_name": pname,
                "full_path": full_path,
                "relative_path": str(rel_path.as_posix())
            })
        else:
            missing_items.append((item_id, pname))

    if missing_items:
        print(f"[!] ERROR: {len(missing_items)} metadata rows could not be matched to image files!")
        for mid, mpname in missing_items[:5]:
            print(f"    Missing ID: {mid} | product_name: {mpname}")
        sys.exit(1)

    print(f"[*] Verified 1-to-1 match for all {len(ordered_items):,} items.")

    # 4. Load ResNet50 Model (weights='imagenet', include_top=False, pooling='avg')
    print("\n[*] Loading ResNet50 model (weights='imagenet', include_top=False, pooling='avg')...")
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    print("[*] ResNet50 model loaded successfully.")

    # 5. Extract Feature Embeddings in Batches
    batch_size = 32
    all_embeddings = []
    processed_ids = []

    batch_imgs = []
    batch_item_ids = []

    print(f"\n[*] Processing {len(ordered_items):,} images in batches of {batch_size}...")

    for i, item in enumerate(ordered_items, 1):
        try:
            with Image.open(item["full_path"]) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((224, 224), Image.Resampling.LANCZOS)
                img_arr = image.img_to_array(img_resized)

            batch_imgs.append(img_arr)
            batch_item_ids.append(item["id"])

        except Exception as err:
            print(f"[!] Error processing image {item['full_path']}: {err}")
            sys.exit(1)

        if len(batch_imgs) == batch_size or i == len(ordered_items):
            batch_tensor = np.array(batch_imgs, dtype=np.float32)
            batch_preprocessed = preprocess_input(batch_tensor)

            raw_feats = model.predict(batch_preprocessed, verbose=0)  # Shape: (B, 2048)

            # Apply L2 Normalization to each vector in batch
            norms = np.linalg.norm(raw_feats, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_feats = raw_feats / norms

            all_embeddings.append(norm_feats)
            processed_ids.extend(batch_item_ids)

            batch_imgs = []
            batch_item_ids = []

        if i % 500 == 0 or i == len(ordered_items):
            print(f"  Progress: {i:,} / {len(ordered_items):,} processed ({time.time() - start_time:.1f}s)")

    # 6. Assemble Final Matrix & Save Outputs
    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    image_ids_array = np.array(processed_ids, dtype=object)

    np.save(embeddings_output_path, embeddings_matrix)
    np.save(image_ids_output_path, image_ids_array)

    total_time = time.time() - start_time
    print(f"\n[+] Embeddings generated successfully in {total_time:.2f}s!")
    print(f"[+] Saved embeddings shape : {embeddings_matrix.shape}")
    print(f"[+] Saved image_ids shape  : {image_ids_array.shape}")

    # 7. Comprehensive Verification
    print("\n" + "=" * 70)
    print("                      VERIFICATION REPORT                      ")
    print("=" * 70)

    # Check 1: Processed count
    print(f"1. Processed Images Count : {len(processed_ids):,} / {total_rows:,} (Matched)")
    assert len(processed_ids) == total_rows, "Processed count mismatch!"

    # Check 2: Embedding shape & dtype
    print(f"2. Embeddings Shape       : {embeddings_matrix.shape} (Expected: ({total_rows}, 2048))")
    assert embeddings_matrix.shape == (total_rows, 2048), "Embeddings shape mismatch!"
    print(f"3. Embeddings Dtype       : {embeddings_matrix.dtype} (Expected: float32)")
    assert embeddings_matrix.dtype == np.float32, "Embeddings dtype mismatch!"

    # Check 3: Image IDs shape
    print(f"4. Image IDs Shape        : {image_ids_array.shape} (Expected: ({total_rows},))")
    assert image_ids_array.shape == (total_rows,), "Image IDs shape mismatch!"

    # Check 4: NaN and Inf validation
    has_nan = np.isnan(embeddings_matrix).any()
    has_inf = np.isinf(embeddings_matrix).any()
    print(f"5. NaN Check              : {'FAIL (NaNs found!)' if has_nan else 'PASS (No NaNs)'}")
    print(f"6. Inf Check              : {'FAIL (Infs found!)' if has_inf else 'PASS (No Infs)'}")
    assert not has_nan, "Embeddings matrix contains NaN values!"
    assert not has_inf, "Embeddings matrix contains Inf values!"

    # Check 5: L2 Norm Validation
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    mean_norm = float(np.mean(norms))
    print(f"7. L2 Norm Statistics    : Min={min_norm:.6f}, Max={max_norm:.6f}, Mean={mean_norm:.6f} (Expected: ~1.0)")
    assert np.allclose(norms, 1.0, atol=1e-4), "L2 norms diverge from 1.0!"

    # Check 6: ID Mapping Sample Alignment
    print("\n--- Sample ID Mapping Alignment Check ---")
    print(f"First 3 Mappings:")
    for idx in range(3):
        item = ordered_items[idx]
        print(f"  Row {idx}: ID={image_ids_array[idx]} | product_name={item['product_name']} | rel_path={item['relative_path']}")
        assert image_ids_array[idx] == item["id"], f"ID mismatch at index {idx}"

    print(f"\nLast 3 Mappings:")
    for idx in range(total_rows - 3, total_rows):
        item = ordered_items[idx]
        print(f"  Row {idx}: ID={image_ids_array[idx]} | product_name={item['product_name']} | rel_path={item['relative_path']}")
        assert image_ids_array[idx] == item["id"], f"ID mismatch at index {idx}"

    print("\n" + "=" * 70)
    print("       ALL VERIFICATION CHECKS PASSED PERFECTLY!       ")
    print("=" * 70)


if __name__ == "__main__":
    generate_embeddings()
