#!/usr/bin/env python3
"""
ResNet50 Feature Extraction Pipeline for AI Fashion Recommendation System.

This script:
1. Reads `dataset/cleaned_styles.csv`.
2. Loads corresponding images from `dataset/images/`.
3. Loads Keras ResNet50 (include_top=False, pooling='avg', weights='imagenet').
4. Preprocesses images (Resize 224x224, RGB conversion, TensorFlow preprocess_input).
5. Generates 2048-dimensional deep feature embeddings.
6. Handles initial limit (MAX_IMAGES) or processes full dataset when MAX_IMAGES is None.
7. Saves outputs to `dataset/embeddings.npy` and `dataset/image_ids.npy`.
8. Gracefully handles corrupted or missing images and logs their IDs.
9. Prints comprehensive summary report including total found, processed, failed, matrix shape, and time taken.
"""

import sys
import time
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
    from PIL import Image
except ImportError:
    missing_dependencies.append("pillow (PIL)")

try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.preprocessing import image
except ImportError:
    missing_dependencies.append("tensorflow")

if missing_dependencies:
    print("=" * 70)
    print("ERROR: Missing required Python dependencies!")
    print(f"The following required package(s) could not be imported: {', '.join(missing_dependencies)}")
    print("\nPlease install the missing dependencies using pip:")
    print("  pip install " + " ".join([pkg.split()[0] for pkg in missing_dependencies]))
    print("\nOr execute using the project virtual environment:")
    print("  garment-processing-api\\.venv\\Scripts\\python.exe garment-processing-api/scripts/generate_embeddings.py")
    print("=" * 70)
    sys.exit(1)


# ==============================================================================
# Configuration Parameters
# ==============================================================================
# Maximum number of images to process for this run.
# Set MAX_IMAGES = 1000 for initial test run, or MAX_IMAGES = None to process the entire dataset.
MAX_IMAGES = None

# Batch size for GPU/CPU inference performance and memory management
BATCH_SIZE = 32


# ==============================================================================
# Step 2: Dataset Directory Resolution Helper
# ==============================================================================
def resolve_dataset_dir() -> Path:
    """
    Locates dataset directory containing cleaned_styles.csv and images/.
    Checks script-relative paths, working directory, and backend folder paths.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset",
        script_dir.parent.parent / "dataset",
        script_dir.parent.parent / "garment-processing-api" / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "garment-processing-api" / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "cleaned_styles.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find dataset/cleaned_styles.csv in expected directory locations.")


def find_image_path(images_dir: Path, item_id: str) -> Path | None:
    """
    Finds image file matching item_id across supported file extensions.
    """
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img_path = images_dir / f"{item_id}{ext}"
        if img_path.exists():
            return img_path
    return None


# ==============================================================================
# Step 3: Main Embedding Generation Pipeline
# ==============================================================================
def main():
    start_time = time.time()

    print("=" * 70)
    print("      RESNET50 FEATURE EXTRACTION PIPELINE - FASHION EMBEDDINGS      ")
    print("=" * 70)

    # Resolve dataset paths
    dataset_dir = resolve_dataset_dir()
    csv_path = dataset_dir / "cleaned_styles.csv"
    images_dir = dataset_dir / "images"
    embeddings_output_path = dataset_dir / "embeddings.npy"
    image_ids_output_path = dataset_dir / "image_ids.npy"

    print(f"Reading dataset from     : {csv_path}")
    print(f"Loading images from      : {images_dir}")
    print(f"Output embeddings path   : {embeddings_output_path}")
    print(f"Output image IDs path    : {image_ids_output_path}")

    # Read cleaned_styles.csv
    df = pd.read_csv(csv_path, dtype={"id": str})
    
    # Extract item IDs
    all_item_ids = df["id"].astype(str).str.strip().tolist()
    total_found_in_csv = len(all_item_ids)

    # Determine subset size based on MAX_IMAGES
    if MAX_IMAGES is not None and MAX_IMAGES > 0:
        target_item_ids = all_item_ids[:MAX_IMAGES]
        print(f"Processing limit set to  : MAX_IMAGES = {MAX_IMAGES} (out of {total_found_in_csv:,} total records)")
    else:
        target_item_ids = all_item_ids
        print(f"Processing limit set to  : ALL ({total_found_in_csv:,} records)")

    total_target_count = len(target_item_ids)

    # Step 4: Load ResNet50 Model
    print("\nLoading TensorFlow Keras ResNet50 model (weights='imagenet', include_top=False, pooling='avg')...")
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    print("ResNet50 model loaded successfully.")

    # Tracking metrics
    processed_image_ids = []
    all_embeddings_list = []
    failed_images = []

    batch_images = []
    batch_ids = []

    print(f"\nExtracting features for {total_target_count:,} images in batches of {BATCH_SIZE}...")

    # Step 5: Process Images in Batches
    for idx, item_id in enumerate(target_item_ids, 1):
        img_path = find_image_path(images_dir, item_id)
        
        if not img_path:
            failed_images.append((item_id, "File not found"))
            continue

        try:
            # Load image, convert to RGB, and resize to 224x224 target size
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
                img_arr = image.img_to_array(img_resized)

            batch_images.append(img_arr)
            batch_ids.append(item_id)

        except Exception as err:
            # Catch corrupted or unreadable image errors gracefully
            failed_images.append((item_id, f"Corrupted/unreadable image ({type(err).__name__}: {err})"))
            continue

        # Perform inference when batch is full or at last item
        if len(batch_images) == BATCH_SIZE or idx == total_target_count:
            if batch_images:
                # Convert batch list to numpy array shape: (batch_size, 224, 224, 3)
                batch_tensor = np.array(batch_images, dtype=np.float32)
                # Apply ResNet50 specific preprocessing (BGR conversion, mean subtraction)
                batch_preprocessed = preprocess_input(batch_tensor)
                
                # Predict 2048-dim embeddings shape: (batch_size, 2048)
                batch_embeddings = model.predict(batch_preprocessed, verbose=0)
                
                all_embeddings_list.append(batch_embeddings)
                processed_image_ids.extend(batch_ids)

                # Reset batch buffers
                batch_images = []
                batch_ids = []

        # Progress reporting every 200 images
        if idx % 200 == 0 or idx == total_target_count:
            elapsed_sec = time.time() - start_time
            print(f"  Progress: {idx:,} / {total_target_count:,} items processed ({elapsed_sec:.1f}s elapsed)")

    # Step 6: Combine Batch Results & Save Outputs
    if all_embeddings_list:
        final_embeddings = np.vstack(all_embeddings_list)
    else:
        final_embeddings = np.empty((0, 2048), dtype=np.float32)

    final_image_ids = np.array(processed_image_ids, dtype=object)

    # Save to .npy files
    np.save(embeddings_output_path, final_embeddings)
    np.save(image_ids_output_path, final_image_ids)

    end_time = time.time()
    total_time_sec = end_time - start_time

    # Step 7: Print Summary Report
    print("\n" + "=" * 70)
    print("                    EXTRACTION SUMMARY REPORT                    ")
    print("=" * 70)
    print(f"Total images found (in dataset subset) : {total_target_count:,}")
    print(f"Total images processed successfully   : {len(processed_image_ids):,}")
    print(f"Failed images                         : {len(failed_images):,}")
    
    if failed_images:
        print("\n--- Failed Image IDs ---")
        for fid, reason in failed_images[:20]:  # Show up to first 20 failures
            print(f"  ID: {fid:<15} | Reason: {reason}")
        if len(failed_images) > 20:
            print(f"  ... and {len(failed_images) - 20} more failed image(s).")

    print("\n--- Matrix & Output File Details ---")
    print(f"Embedding matrix shape                : {final_embeddings.shape}")
    print(f"Saved embeddings vector file          : {embeddings_output_path}")
    print(f"Saved image IDs file                  : {image_ids_output_path}")
    
    if total_time_sec >= 60:
        mins = int(total_time_sec // 60)
        secs = total_time_sec % 60
        time_str = f"{mins}m {secs:.2f}s ({total_time_sec:.2f} total seconds)"
    else:
        time_str = f"{total_time_sec:.2f} seconds"

    print(f"Time taken                            : {time_str}")
    print("=" * 70)


if __name__ == "__main__":
    main()
