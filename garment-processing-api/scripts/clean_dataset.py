#!/usr/bin/env python3
"""
Dataset Cleaning Script for AI Fashion Recommendation + Virtual Try-On System.

Filters `dataset/styles.csv`:
1. Keeps records with masterCategory == 'Apparel'.
2. Keeps specified articleTypes: Tshirts, Shirts, Jackets, Sweatshirts, Hoodies, Kurtas, Tops, Dresses, Jeans, Trousers, Shorts, Skirts, Track Pants.
3. Removes records whose corresponding image does not exist in `dataset/images/`.
4. Saves cleaned records to `dataset/cleaned_styles.csv`.
5. Prints detailed statistics.
"""

import csv
from collections import Counter
from pathlib import Path
import sys

ALLOWED_ARTICLE_TYPES = {
    "Tshirts",
    "Shirts",
    "Jackets",
    "Sweatshirts",
    "Hoodies",
    "Kurtas",
    "Tops",
    "Dresses",
    "Jeans",
    "Trousers",
    "Shorts",
    "Skirts",
    "Track Pants",
}

def resolve_dataset_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "dataset",
        script_dir.parent / "garment-processing-api" / "dataset",
        Path.cwd() / "dataset",
        Path.cwd() / "garment-processing-api" / "dataset",
    ]
    for candidate in candidates:
        if (candidate / "styles.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find dataset/styles.csv in expected directory locations.")

def check_image_exists(image_dir: Path, item_id: str) -> bool:
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        if (image_dir / f"{item_id}{ext}").exists():
            return True
    return False

def clean_dataset():
    dataset_dir = resolve_dataset_dir()
    styles_csv = dataset_dir / "styles.csv"
    images_dir = dataset_dir / "images"
    output_csv = dataset_dir / "cleaned_styles.csv"

    print(f"Reading original dataset from: {styles_csv}")
    print(f"Checking images in: {images_dir}")

    rows = []
    fieldnames = []

    with open(styles_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(row)

    original_count = len(rows)
    cleaned_rows = []

    for row in rows:
        # Filter 1: masterCategory == Apparel
        master_cat = (row.get("masterCategory") or "").strip()
        if master_cat != "Apparel":
            continue

        # Filter 2: allowed articleType
        article_type = (row.get("articleType") or "").strip()
        if article_type not in ALLOWED_ARTICLE_TYPES:
            continue

        # Filter 3: image existence check
        item_id = (row.get("id") or "").strip()
        if not item_id or not check_image_exists(images_dir, item_id):
            continue

        cleaned_rows.append(row)

    remaining_count = len(cleaned_rows)
    removed_count = original_count - remaining_count

    # Save to dataset/cleaned_styles.csv ignoring extraneous fields (e.g. key None)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"\nCleaned dataset written to: {output_csv}\n")

    # Compute Statistics
    article_type_counter = Counter((r.get("articleType") or "").strip() for r in cleaned_rows)
    gender_counter = Counter((r.get("gender") or "").strip() for r in cleaned_rows)
    usage_counter = Counter((r.get("usage") or "").strip() for r in cleaned_rows)

    # Print Summary Report
    print("=" * 60)
    print("                DATASET CLEANING REPORT                ")
    print("=" * 60)
    print(f"Original number of records : {original_count:,}")
    print(f"Remaining number of records: {remaining_count:,}")
    print(f"Number removed             : {removed_count:,}")
    print("=" * 60)

    print("\n--- Count by articleType ---")
    for article_type in sorted(ALLOWED_ARTICLE_TYPES):
        count = article_type_counter.get(article_type, 0)
        print(f"  {article_type:<20}: {count:,}")

    print("\n--- Count by gender ---")
    for gender, count in gender_counter.most_common():
        g_name = gender if gender else "(Missing/Unspecified)"
        print(f"  {g_name:<20}: {count:,}")

    print("\n--- Count by usage ---")
    for usage, count in usage_counter.most_common():
        u_name = usage if usage else "(Missing/Unspecified)"
        print(f"  {u_name:<20}: {count:,}")

    print("=" * 60)

if __name__ == "__main__":
    try:
        clean_dataset()
    except Exception as e:
        import traceback
        print(f"Error during dataset cleaning: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
