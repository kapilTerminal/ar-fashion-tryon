#!/usr/bin/env python3
"""
FAISS Indexing Pipeline for AI Fashion Recommendation System.

This script:
1. Verifies required dependencies (faiss, numpy, pandas, pickle).
2. Resolves and loads:
   - `dataset/embeddings.npy`
   - `dataset/image_ids.npy`
   - `dataset/cleaned_styles.csv`
3. Normalizes every embedding using L2 normalization before indexing.
4. Builds a FAISS IndexFlatIP index (Inner Product for cosine similarity).
5. Adds every normalized embedding into the index.
6. Saves the index to `dataset/faiss_index.bin`.
7. Saves detailed metadata to `dataset/faiss_metadata.pkl`:
   - `image_ids`: List of image IDs in indexing order
   - `row_positions`: Dict mapping image_id -> row position index
   - `csv_lookup_mapping`: Dict mapping image_id -> detailed product metadata
8. Prints summary metrics (count, dimension, index type, indexed vectors, file paths, time taken).
9. Verifies the saved index with 3 query embeddings (first, middle, random item) and prints Top-5 nearest neighbors.
"""

import os
import sys
import time
import pickle
import random
from pathlib import Path

# ==============================================================================
# Step 1: Missing Dependency Detection & Friendly Reporting
# ==============================================================================
missing_dependencies = []

try:
    import numpy as np
except ImportError:
    missing_dependencies.append("numpy")

try:
    import pandas as pd
except ImportError:
    missing_dependencies.append("pandas")

try:
    import faiss
except ImportError:
    missing_dependencies.append("faiss-cpu (or faiss-gpu)")

if missing_dependencies:
    print("=" * 70)
    print("ERROR: Missing required Python dependencies!")
    print(f"The following required package(s) could not be imported: {', '.join(missing_dependencies)}")
    print("\nPlease install missing packages using pip/uv:")
    print("  pip install faiss-cpu numpy pandas")
    print("  uv pip install faiss-cpu numpy pandas")
    print("=" * 70)
    sys.exit(1)


# ==============================================================================
# Step 2: Dataset Path Resolution Helper
# ==============================================================================
def resolve_dataset_dir() -> Path:
    """
    Locates dataset directory containing embeddings.npy, image_ids.npy, and cleaned_styles.csv.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset",
        script_dir.parent / "garment-processing-api" / "dataset",
        script_dir.parent.parent / "dataset",
        script_dir.parent.parent / "garment-processing-api" / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "garment-processing-api" / "dataset",
    ]
    for candidate in candidates:
        if (
            (candidate / "embeddings.npy").exists()
            and (candidate / "image_ids.npy").exists()
            and (candidate / "cleaned_styles.csv").exists()
        ):
            return candidate
            
    # Check individual missing files for precise reporting
    print("=" * 70)
    print("ERROR: Dataset files could not be located!")
    print("Expected dataset directory containing:")
    print("  - embeddings.npy")
    print("  - image_ids.npy")
    print("  - cleaned_styles.csv")
    print("\nChecked candidate paths:")
    for candidate in candidates:
        print(f"  - {candidate.resolve()}")
    print("=" * 70)
    sys.exit(1)


# ==============================================================================
# Step 3: Main FAISS Index Building Pipeline
# ==============================================================================
def build_faiss_index():
    start_time = time.time()
    print("=" * 70)
    print("FAISS Indexing Pipeline - AI Fashion Recommendation System")
    print("=" * 70)

    # 1. Locate dataset directory
    dataset_dir = resolve_dataset_dir()
    print(f"[*] Dataset directory resolved: {dataset_dir}")

    embeddings_path = dataset_dir / "embeddings.npy"
    image_ids_path = dataset_dir / "image_ids.npy"
    styles_csv_path = dataset_dir / "cleaned_styles.csv"
    index_output_path = dataset_dir / "faiss_index.bin"
    metadata_output_path = dataset_dir / "faiss_metadata.pkl"

    # 2. Load input dataset files safely
    print("[*] Loading dataset files...")
    try:
        embeddings = np.load(embeddings_path)
        print(f"    - Loaded embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    except Exception as e:
        print(f"[!] Critical Error: Failed to load embeddings file '{embeddings_path}': {e}")
        sys.exit(1)

    try:
        image_ids_arr = np.load(image_ids_path, allow_pickle=True)
        image_ids = [str(img_id) for img_id in image_ids_arr]
        print(f"    - Loaded image IDs count: {len(image_ids)}")
    except Exception as e:
        print(f"[!] Critical Error: Failed to load image IDs file '{image_ids_path}': {e}")
        sys.exit(1)

    try:
        df_styles = pd.read_csv(styles_csv_path)
        print(f"    - Loaded cleaned styles CSV rows: {len(df_styles)}")
    except Exception as e:
        print(f"[!] Critical Error: Failed to read CSV file '{styles_csv_path}': {e}")
        sys.exit(1)

    # Dimension and count validation
    num_embeddings, dim = embeddings.shape
    if num_embeddings != len(image_ids):
        print(f"[!] Error: Mismatch between embeddings count ({num_embeddings}) and image IDs count ({len(image_ids)})!")
        sys.exit(1)

    # 3. Perform L2 Normalization
    print("[*] Normalizing embeddings using L2 normalization...")
    # Ensure float32 matrix for FAISS compatibility
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    print("    - L2 normalization completed.")

    # 4. Build FAISS IndexFlatIP (Inner Product for Cosine Similarity)
    print(f"[*] Building FAISS IndexFlatIP index (dimension={dim})...")
    index = faiss.IndexFlatIP(dim)
    index_type_name = type(index).__name__

    # 5. Add normalized vectors to index
    index.add(embeddings)
    total_indexed = index.ntotal
    print(f"    - Successfully added {total_indexed} vectors to FAISS index.")

    # 6. Save FAISS binary index
    print(f"[*] Saving FAISS binary index to '{index_output_path}'...")
    faiss.write_index(index, str(index_output_path))
    print(f"    - Binary index saved ({os.path.getsize(index_output_path) / (1024*1024):.2f} MB)")

    # 7. Construct and save rich metadata
    print("[*] Building metadata dictionary...")
    # Ensure string representation for IDs in CSV lookup mapping
    if 'id' in df_styles.columns:
        df_styles['id_str'] = df_styles['id'].astype(str)
    else:
        # Fallback if first column is ID
        df_styles['id_str'] = df_styles.iloc[:, 0].astype(str)

    csv_lookup_mapping = {}
    desired_cols = [
        "id", "gender", "masterCategory", "subCategory", 
        "articleType", "baseColour", "usage", "productDisplayName"
    ]
    available_cols = [col for col in desired_cols if col in df_styles.columns]

    for _, row in df_styles.iterrows():
        img_id = str(row['id_str'])
        meta_entry = {col: (None if pd.isna(row[col]) else row[col]) for col in available_cols}
        csv_lookup_mapping[img_id] = meta_entry

    row_positions = {img_id: idx for idx, img_id in enumerate(image_ids)}

    metadata = {
        "image_ids": image_ids,
        "row_positions": row_positions,
        "csv_lookup_mapping": csv_lookup_mapping,
    }

    print(f"[*] Saving metadata pickle to '{metadata_output_path}'...")
    with open(metadata_output_path, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    - Metadata pickle saved ({os.path.getsize(metadata_output_path) / (1024*1024):.2f} MB)")

    elapsed_time = time.time() - start_time

    # 8. Print Summary Report
    print("\n" + "=" * 70)
    print("FAISS INDEXING SUMMARY REPORT")
    print("=" * 70)
    print(f"  Number of Embeddings Loaded : {num_embeddings:,}")
    print(f"  Embedding Dimension         : {dim}")
    print(f"  FAISS Index Type            : {index_type_name}")
    print(f"  Total Vectors Indexed       : {total_indexed:,}")
    print(f"  Binary Index Path           : {index_output_path.resolve()}")
    print(f"  Metadata Pickle Path        : {metadata_output_path.resolve()}")
    print(f"  Execution Time              : {elapsed_time:.2f} seconds")
    print("=" * 70 + "\n")

    # 9. Verify Saved Index with 3 Query Embeddings
    verify_faiss_index(index_output_path, metadata_output_path, embeddings, image_ids)


def verify_faiss_index(index_path: Path, metadata_path: Path, embeddings: np.ndarray, image_ids: list):
    """
    Verifies saved index by loading it from disk and executing 3 sample queries:
    - First item (index 0)
    - Middle item (index N // 2)
    - Random item
    """
    print("=" * 70)
    print("FAISS INDEX VERIFICATION & SIMILARITY SEARCH TEST")
    print("=" * 70)

    print(f"[*] Loading index from: {index_path}")
    loaded_index = faiss.read_index(str(index_path))

    print(f"[*] Loading metadata from: {metadata_path}")
    with open(metadata_path, "rb") as f:
        loaded_metadata = pickle.load(f)

    meta_image_ids = loaded_metadata["image_ids"]
    csv_lookup = loaded_metadata["csv_lookup_mapping"]

    N = len(image_ids)
    test_indices = [
        ("FIRST ITEM (Index 0)", 0),
        (f"MIDDLE ITEM (Index {N // 2})", N // 2),
        ("RANDOM ITEM", random.randint(0, N - 1)),
    ]

    top_k = 5

    for query_label, query_idx in test_indices:
        query_id = image_ids[query_idx]
        query_vec = embeddings[query_idx : query_idx + 1]  # shape (1, dim)
        query_info = csv_lookup.get(query_id, {})
        query_name = query_info.get("productDisplayName", "Unknown Product")
        query_cat = query_info.get("articleType", "Unknown Category")

        print(f"\n---> Query Target [{query_label}]:")
        print(f"     Image ID     : {query_id}")
        print(f"     Product Name : {query_name}")
        print(f"     Article Type : {query_cat}")

        # Search Top-K
        scores, neighbor_indices = loaded_index.search(query_vec, top_k)
        
        print(f"     Top-{top_k} Nearest Neighbors:")
        print(f"     {'Rank':<5} {'Neighbor ID':<15} {'Score':<10} {'Article Type':<18} {'Product Display Name'}")
        print("     " + "-" * 75)

        for rank_idx in range(top_k):
            neighbor_pos = neighbor_indices[0][rank_idx]
            score = scores[0][rank_idx]
            neighbor_id = meta_image_ids[neighbor_pos]
            neighbor_meta = csv_lookup.get(neighbor_id, {})
            neighbor_name = neighbor_meta.get("productDisplayName", "Unknown")
            neighbor_art = neighbor_meta.get("articleType", "Unknown")

            print(f"     #{rank_idx + 1:<4} {neighbor_id:<15} {score:<10.4f} {neighbor_art:<18} {neighbor_name[:35]}")

    print("=" * 70)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    build_faiss_index()
