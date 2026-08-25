import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path("garment-processing-api").resolve()))

import numpy as np
import pandas as pd
import faiss
from services.hybrid_recommendation_service import get_hybrid_recommendation_service
from models.user_profile import UserProfile

service = get_hybrid_recommendation_service()

# ----------------------------------------------------------------------
# 1. DATA INTEGRITY CHECKS (Section 14)
# ----------------------------------------------------------------------
def test_data_integrity():
    print("\n[TEST] Running Data Integrity Checks...")
    csv_path = Path("garment-processing-api/dataset/fashion_dataset/metadata.csv")
    index_path = Path("garment-processing-api/dataset/fashion_dataset/faiss_index.bin")
    emb_path = Path("garment-processing-api/dataset/fashion_dataset/embeddings.npy")

    df = pd.read_csv(csv_path)
    assert len(df) == 3585, f"Expected 3585 records in metadata.csv, got {len(df)}"

    idx = faiss.read_index(str(index_path))
    assert idx.ntotal == 3585, f"Expected 3585 vectors in faiss_index.bin, got {idx.ntotal}"

    embs = np.load(emb_path)
    assert embs.shape == (3585, 2048), f"Expected shape (3585, 2048), got {embs.shape}"
    print("  [PASS] Data integrity checks passed (metadata.csv=3585, faiss_index=3585, embeddings=(3585, 2048)).")


# ----------------------------------------------------------------------
# 2. CONTROLLED REGRESSION TESTS (Section 12)
# ----------------------------------------------------------------------
def test_regression_test_a_men_casual():
    print("\n[TEST A] Running Men + Casual + Black + Casual...")
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Men", occasion="Casual", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    res = service.recommend(profile, top_k=10)
    recs = res["recommendations"]

    assert len(recs) == 10
    genders = [r["metadata"]["gender"].lower() for r in recs]

    # No Women or Girls
    assert not any(g in ["women", "female", "girls", "girl"] for g in genders), "No opposite gender in Men results"

    # Men rank above Unisex
    men_ranks = [i for i, g in enumerate(genders) if g in ["men", "male"]]
    unisex_ranks = [i for i, g in enumerate(genders) if g == "unisex"]
    if men_ranks and unisex_ranks:
        assert min(men_ranks) < min(unisex_ranks), "Men garments should rank above Unisex garments"

    # Verify candidate visual prototype similarity scores are valid floats
    for r in recs:
        assert isinstance(r["similarity"], float)
        assert 0.0 <= r["similarity"] <= 1.0

    print("  [PASS] TEST A Passed.")


def test_regression_test_b_women_casual():
    print("\n[TEST B] Running Women + Casual + Black + Casual...")
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Women", occasion="Casual", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    res = service.recommend(profile, top_k=10)
    recs = res["recommendations"]

    assert len(recs) == 10
    genders = [r["metadata"]["gender"].lower() for r in recs]

    # No Men or Boys
    assert not any(g in ["men", "male", "boys", "boy"] for g in genders), "No opposite gender in Women results"

    # Women rank above Unisex
    women_ranks = [i for i, g in enumerate(genders) if g in ["women", "female"]]
    unisex_ranks = [i for i, g in enumerate(genders) if g == "unisex"]
    if women_ranks and unisex_ranks:
        assert min(women_ranks) < min(unisex_ranks), "Women garments should rank above Unisex garments"

    print("  [PASS] TEST B Passed.")


def test_regression_test_c_men_formal():
    print("\n[TEST C] Running Men + Formal + Black + Casual...")
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Men", occasion="Formal", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    res = service.recommend(profile, top_k=10)
    recs = res["recommendations"]

    assert len(recs) == 10
    for r in recs:
        g = r["metadata"]["gender"].lower()
        occ = r["metadata"]["occasion"].lower()

        # No Women/Girls
        assert g not in ["women", "female", "girls", "girl"]

        # Formal-eligible only (office or formal_event in occasion string)
        assert any(t in occ for t in ["office", "formal_event", "formal"])

    # Verify deterministic ranking across two separate calls
    res2 = service.recommend(profile, top_k=10)
    ids1 = [r["id"] for r in recs]
    ids2 = [r["id"] for r in res2["recommendations"]]
    assert ids1 == ids2, "Recommendation ranking must be 100% deterministic"

    print("  [PASS] TEST C Passed.")


def test_regression_test_d_women_formal():
    print("\n[TEST D] Running Women + Formal + Black + Casual...")
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Women", occasion="Formal", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    res = service.recommend(profile, top_k=10)
    recs = res["recommendations"]

    assert len(recs) == 10
    for r in recs:
        g = r["metadata"]["gender"].lower()
        occ = r["metadata"]["occasion"].lower()

        # No Men/Boys
        assert g not in ["men", "male", "boys", "boy"]

        # Formal-eligible only
        assert any(t in occ for t in ["office", "formal_event", "formal"])

    print("  [PASS] TEST D Passed.")


def test_regression_test_e_men_sports():
    print("\n[TEST E] Running Men + Sports + Black + Sporty...")
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Men", occasion="Sports", preferred_color="Black", preferred_style="Sporty",
        body_type="Rectangle", skin_tone="Medium"
    )
    res = service.recommend(profile, top_k=10)
    recs = res["recommendations"]

    assert len(recs) == 10
    for r in recs:
        g = r["metadata"]["gender"].lower()
        occ = r["metadata"]["occasion"].lower()

        # No Women/Girls
        assert g not in ["women", "female", "girls", "girl"]

        # Sports-eligible only
        assert "sports" in occ

    print("  [PASS] TEST E Passed.")


# ----------------------------------------------------------------------
# 3. NEGATIVE & BOUNDARY TESTS (Section 13)
# ----------------------------------------------------------------------
def test_negative_tests():
    print("\n[TEST NEGATIVE] Running Negative & Boundary Tests...")
    # 1. Opposite gender never appears
    profile_men = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Men", occasion="Casual", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    candidates_men = service.generate_metadata_candidates(profile_men)
    for meta in candidates_men:
        assert meta["gender"].lower() not in ["women", "female", "girls", "girl"]

    # 2. Casual-only / Sports-only cannot pass Formal filtering
    profile_formal = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=np.zeros(2048, dtype=np.float32),
        gender="Men", occasion="Formal", preferred_color="Black", preferred_style="Casual",
        body_type="Rectangle", skin_tone="Medium"
    )
    candidates_formal = service.generate_metadata_candidates(profile_formal)
    for meta in candidates_formal:
        raw_occ = meta["occasion"].lower()
        # Must contain office or formal_event
        assert "office" in raw_occ or "formal_event" in raw_occ

    # 3. Random embeddings do NOT affect recommendation ranking
    rand_emb1 = np.random.randn(2048).astype(np.float32)
    rand_emb2 = np.random.randn(2048).astype(np.float32)

    p1 = UserProfile(image_path="uploads/users/t.jpg", embedding=rand_emb1, gender="Men", occasion="Casual", preferred_color="Black", preferred_style="Casual", body_type="Rectangle", skin_tone="Medium")
    p2 = UserProfile(image_path="uploads/users/t.jpg", embedding=rand_emb2, gender="Men", occasion="Casual", preferred_color="Black", preferred_style="Casual", body_type="Rectangle", skin_tone="Medium")

    r1 = service.recommend(p1, top_k=10)["recommendations"]
    r2 = service.recommend(p2, top_k=10)["recommendations"]

    ids1 = [item["id"] for item in r1]
    ids2 = [item["id"] for item in r2]
    assert ids1 == ids2, "Embedding vector must have ZERO effect on recommendation ranking"

    print("  [PASS] Negative & Boundary Tests Passed.")


if __name__ == "__main__":
    print("=" * 80)
    print("      RUNNING COMPLETE RANKING REDESIGN REGRESSION SUITE      ")
    print("=" * 80)
    test_data_integrity()
    test_regression_test_a_men_casual()
    test_regression_test_b_women_casual()
    test_regression_test_c_men_formal()
    test_regression_test_d_women_formal()
    test_regression_test_e_men_sports()
    test_negative_tests()
    print("\n" + "=" * 80)
    print("          ALL REGRESSION & INTEGRITY TESTS PASSED 100%!       ")
    print("=" * 80)
