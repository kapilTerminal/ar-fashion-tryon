import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path("garment-processing-api").resolve()))

from services.hybrid_recommendation_service import get_hybrid_recommendation_service
from models.user_profile import UserProfile
import numpy as np

service = get_hybrid_recommendation_service()

dummy_emb = np.random.randn(2048).astype(np.float32)
dummy_emb /= np.linalg.norm(dummy_emb)

def run_test(gender_str: str):
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=dummy_emb,
        gender=gender_str,
        occasion="Casual",
        preferred_color="Black",
        preferred_style="Casual",
        body_type="Rectangle",
        skin_tone="Medium"
    )

    print("=" * 80)
    print(f"--- TEST FOR USER GENDER: '{gender_str}' (Occasion=Casual, Color=Black, Style=Casual) ---")

    rec_res = service.recommend(profile, top_k=10)
    recs = rec_res["recommendations"]

    exact_matches = 0
    unisex_matches = 0
    exact_scores = []
    unisex_scores = []

    print(f"\nTop 10 Recommendations for {gender_str}:")
    for idx, r in enumerate(recs, 1):
        meta = r["metadata"]
        g = meta["gender"]
        score = r["final_score"]
        if (gender_str.lower() in ["men", "male"] and g.lower() in ["men", "male"]) or \
           (gender_str.lower() in ["women", "female"] and g.lower() in ["women", "female"]):
            exact_matches += 1
            exact_scores.append(score)
            tag = "[EXACT GENDER]"
        elif g.lower() == "unisex":
            unisex_matches += 1
            unisex_scores.append(score)
            tag = "[UNISEX]"
        else:
            tag = f"[{g.upper()}]"

        print(f"#{idx:<2} | ID: {r['id']:<5} | Product: {r['productDisplayName']:<45} | Gender: {g:<8} {tag:<16} | Sim: {r['similarity']} | Final: {score}")

    print(f"\nSummary for {gender_str} Query: Exact Gender Items = {exact_matches}/10, Unisex Items = {unisex_matches}/10")
    if exact_scores and unisex_scores:
        max_unisex = max(unisex_scores)
        min_exact = min(exact_scores)
        print(f"Min Exact-Gender Score: {min_exact} vs Max Unisex Score: {max_unisex}")
        assert min_exact > max_unisex, f"Expected min exact score ({min_exact}) > max unisex score ({max_unisex})"

if __name__ == "__main__":
    run_test("Men")
    print("\n")
    run_test("Women")
    print("=" * 80)
    print("       ALL GENDER RELEVANCE VERIFICATION CHECKS PASSED!       ")
    print("=" * 80)
