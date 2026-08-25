#!/usr/bin/env python3
"""
FAISS Index Generation Pipeline for NEW Fashion Dataset.

Inputs:
  garment-processing-api/dataset/fashion_dataset/embeddings.npy
  garment-processing-api/dataset/fashion_dataset/image_ids.npy
  garment-processing-api/dataset/fashion_dataset/metadata.csv

Outputs:
  garment-processing-api/dataset/fashion_dataset/faiss_index.bin
  garment-processing-api/dataset/fashion_dataset/faiss_metadata.pkl
"""

import os
import sys
import time
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import faiss


def resolve_fashion_dataset_dir() -> Path:
    """Locates fashion_dataset directory containing embeddings.npy, image_ids.npy, metadata.csv, and Images/."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset" / "fashion_dataset",
        script_dir.parent.parent / "dataset" / "fashion_dataset",
        script_dir.parent.parent / "garment-processing-api" / "dataset" / "fashion_dataset",
        Path.cwd() / "dataset" / "fashion_dataset",
        Path.cwd() / "garment-processing-api" / "dataset" / "fashion_dataset",
    ]
    for candidate in candidates:
        if (
            (candidate / "embeddings.npy").exists()
            and (candidate / "image_ids.npy").exists()
            and (candidate / "metadata.csv").exists()
            and (candidate / "Images").exists()
        ):
            return candidate
    raise FileNotFoundError("Could not locate fashion_dataset directory with required input files.")


def build_faiss_index():
    start_time = time.time()
    print("=" * 70)
    print("  FAISS INDEX GENERATION PIPELINE - NEW STANDALONE GARMENT DATASET  ")
    print("=" * 70)

    dataset_dir = resolve_fashion_dataset_dir()
    embeddings_path = dataset_dir / "embeddings.npy"
    image_ids_path = dataset_dir / "image_ids.npy"
    metadata_csv_path = dataset_dir / "metadata.csv"
    images_dir = dataset_dir / "Images"
    index_output_path = dataset_dir / "faiss_index.bin"
    metadata_output_path = dataset_dir / "faiss_metadata.pkl"

    print(f"[*] Dataset Directory    : {dataset_dir.resolve()}")
    print(f"[*] Embeddings Path      : {embeddings_path.resolve()}")
    print(f"[*] Image IDs Path       : {image_ids_path.resolve()}")
    print(f"[*] Metadata CSV Path    : {metadata_csv_path.resolve()}")
    print(f"[*] Index Output Path    : {index_output_path.resolve()}")
    print(f"[*] Metadata Output Path : {metadata_output_path.resolve()}")

    # 1. Load and verify embeddings.npy
    print("\n[*] Loading embeddings.npy...")
    embeddings = np.load(embeddings_path)
    num_embeddings, dim = embeddings.shape
    print(f"    - Embeddings shape: {embeddings.shape}")
    print(f"    - Embeddings dtype: {embeddings.dtype}")

    assert embeddings.shape == (3585, 2048), f"Expected shape (3585, 2048), got {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Expected dtype float32, got {embeddings.dtype}"

    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    print(f"    - NaN check: {'FAIL' if has_nan else 'PASS (No NaNs)'}")
    print(f"    - Inf check: {'FAIL' if has_inf else 'PASS (No Infs)'}")
    assert not has_nan, "Embeddings contain NaN values!"
    assert not has_inf, "Embeddings contain Inf values!"

    # 2. Load and verify image_ids.npy
    print("\n[*] Loading image_ids.npy...")
    image_ids_arr = np.load(image_ids_path, allow_pickle=True)
    image_ids = [str(img_id).strip() for img_id in image_ids_arr]
    print(f"    - Image IDs count: {len(image_ids)}")
    assert len(image_ids) == 3585, f"Expected 3585 IDs, got {len(image_ids)}"

    # 3. Load and verify metadata.csv
    print("\n[*] Loading metadata.csv...")
    df = pd.read_csv(metadata_csv_path, dtype={"id": str})
    print(f"    - Metadata CSV rows: {len(df)}")
    assert len(df) == 3585, f"Expected 3585 records in metadata.csv, got {len(df)}"

    # Verify IDs map 1-to-1 with metadata rows
    csv_ids = [str(row_id).strip() for row_id in df["id"]]
    assert image_ids == csv_ids, "image_ids.npy does not map 1-to-1 with metadata.csv row IDs!"
    print("    - Verified 1-to-1 ID mapping between image_ids.npy and metadata.csv")

    # 4. Build filename -> relative_path mapping
    print("\n[*] Indexing image subfolder relative paths...")
    filename_to_relpath = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                full_p = Path(root) / f
                rel_p = full_p.relative_to(images_dir)
                filename_to_relpath[f] = str(rel_p.as_posix())

    print(f"    - Indexed {len(filename_to_relpath):,} relative image paths")

    # 5. Build FAISS IndexFlatIP (dimension=2048)
    print("\n[*] Building FAISS IndexFlatIP index (dim=2048)...")
    # Ensure float32 and L2-normalized
    embeddings_matrix = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_matrix)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)
    print(f"    - Added {index.ntotal:,} vectors to FAISS index.")
    assert index.ntotal == 3585, f"Expected 3585 vectors in index, got {index.ntotal}"
    assert index.d == 2048, f"Expected index dimension 2048, got {index.d}"

    # 6. Save FAISS binary index
    print(f"[*] Saving FAISS binary index to '{index_output_path}'...")
    faiss.write_index(index, str(index_output_path))
    assert index_output_path.exists(), "Binary index file was not saved!"
    print(f"    - Saved faiss_index.bin ({os.path.getsize(index_output_path) / (1024*1024):.2f} MB)")

    # 7. Construct and save rich metadata pickle
    print("\n[*] Building faiss_metadata.pkl dictionary...")
    csv_lookup_mapping = {}
    row_positions = {}

    for pos, row in df.iterrows():
        img_id = str(row["id"]).strip()
        pname = str(row["product_name"]).strip()
        rel_path = filename_to_relpath.get(pname, pname)
        img_url = f"/images/{rel_path}"

        meta_entry = {
            "id": img_id,
            "category": str(row["category"]) if pd.notna(row["category"]) else "",
            "occasion": str(row["occasion"]) if pd.notna(row["occasion"]) else "",
            "type": str(row["type"]) if pd.notna(row["type"]) else "",
            "style": str(row["style"]) if pd.notna(row["style"]) else "",
            "color": str(row["color"]) if pd.notna(row["color"]) else "",
            "gender": str(row["gender"]) if pd.notna(row["gender"]) else "",
            "product_name": pname,
            "relative_path": rel_path,
            "image_url": img_url,
        }

        csv_lookup_mapping[img_id] = meta_entry
        row_positions[img_id] = pos

    faiss_metadata = {
        "image_ids": image_ids,
        "row_positions": row_positions,
        "csv_lookup_mapping": csv_lookup_mapping,
    }

    print(f"[*] Saving metadata pickle to '{metadata_output_path}'...")
    with open(metadata_output_path, "wb") as f:
        pickle.dump(faiss_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    assert metadata_output_path.exists(), "Metadata pickle file was not saved!"
    print(f"    - Saved faiss_metadata.pkl ({os.path.getsize(metadata_output_path) / (1024*1024):.2f} MB)")

    elapsed = time.time() - start_time
    print(f"\n[+] FAISS Indexing completed in {elapsed:.2f} seconds.")

    # 8. Complete Verification Report
    print("\n" + "=" * 70)
    print("                      VERIFICATION REPORT                      ")
    print("=" * 70)

    # Reload from disk to verify
    loaded_index = faiss.read_index(str(index_output_path))
    with open(metadata_output_path, "rb") as f:
        loaded_meta = pickle.load(f)

    print(f"1. FAISS Index Dimension      : {loaded_index.d} (Expected: 2048)")
    assert loaded_index.d == 2048, "Loaded FAISS dimension mismatch!"

    print(f"2. FAISS Vector Count (ntotal): {loaded_index.ntotal:,} (Expected: 3,585)")
    assert loaded_index.ntotal == 3585, "Loaded FAISS vector count mismatch!"

    print(f"3. Metadata image_ids Count   : {len(loaded_meta['image_ids']):,} (Expected: 3,585)")
    assert len(loaded_meta['image_ids']) == 3585, "Metadata image_ids count mismatch!"

    print(f"4. Metadata row_positions Count: {len(loaded_meta['row_positions']):,} (Expected: 3,585)")
    assert len(loaded_meta['row_positions']) == 3585, "Metadata row_positions count mismatch!"

    print(f"5. csv_lookup_mapping Count   : {len(loaded_meta['csv_lookup_mapping']):,} (Expected: 3,585)")
    assert len(loaded_meta['csv_lookup_mapping']) == 3585, "csv_lookup_mapping count mismatch!"

    print("\n--- First 3 Metadata Mappings ---")
    for idx in range(3):
        item_id = loaded_meta['image_ids'][idx]
        entry = loaded_meta['csv_lookup_mapping'][item_id]
        print(f"  ID: {entry['id']:<5} | Category: {entry['category']:<10} | Type: {entry['type']:<6} | URL: {entry['image_url']}")

    print("\n--- Last 3 Metadata Mappings ---")
    for idx in range(3582, 3585):
        item_id = loaded_meta['image_ids'][idx]
        entry = loaded_meta['csv_lookup_mapping'][item_id]
        print(f"  ID: {entry['id']:<5} | Category: {entry['category']:<10} | Type: {entry['type']:<6} | URL: {entry['image_url']}")

    print("\n--- Random Item Verification ---")
    rand_idx = random.randint(0, 3584)
    rand_id = loaded_meta['image_ids'][rand_idx]
    rand_entry = loaded_meta['csv_lookup_mapping'][rand_id]
    print(f"  Random Index {rand_idx}: ID={rand_id}")
    print(f"  Product Name  : {rand_entry['product_name']}")
    print(f"  Relative Path : {rand_entry['relative_path']}")
    print(f"  Image URL     : {rand_entry['image_url']}")

    # 9. Verify OLD root-level files remain untouched
    old_root_dir = dataset_dir.parent
    old_files = [
        old_root_dir / "embeddings.npy",
        old_root_dir / "image_ids.npy",
        old_root_dir / "faiss_index.bin",
        old_root_dir / "faiss_metadata.pkl"
    ]
    print("\n--- Old Root-Level Dataset Files Safety Check ---")
    for old_file in old_files:
        exists = old_file.exists()
        print(f"  {old_file.name:<20} : {'EXISTS & UNTOUCHED' if exists else 'MISSING!'}")
        assert exists, f"Old root-level file {old_file.name} is missing!"

    print("\n" + "=" * 70)
    print("      ALL FAISS INDEXING VERIFICATION CHECKS PASSED!      ")
    print("=" * 70)


if __name__ == "__main__":
    build_faiss_index()