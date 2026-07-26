"""
Verification script for Stage 4B: Hybrid Recommendation Engine.

1. Creates a sample user profile with image and parameters.
2. Extracts 2048D ResNet50 user embedding.
3. Queries FAISS index for Top K = 200 candidates.
4. Applies hard filtering (Gender & Occasion).
5. Computes hybrid scores and ranks top 10 recommendations.
6. Prints pipeline metrics (Top K retrieved, filtered candidates count, Top 10 recommendations, similarity scores, final scores, execution time).
7. Verifies every recommendation exists in cleaned_styles.csv.
8. Tests the FastAPI /recommend endpoint.
"""
import sys
import os
import io
import time
import json
from pathlib import Path
from PIL import Image
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set stdout/stderr encoding to utf-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app import app
from services.user_profile_service import create_user_profile
from services.hybrid_recommendation_service import get_hybrid_recommendation_service
from models.user_profile import UserProfile


def verify_stage_4b():
    print("\n=======================================================")
    print("--- STAGE 4B: HYBRID RECOMMENDATION ENGINE VERIFICATION ---")
    print("=======================================================")

    start_time = time.time()

    # 1. Create a sample user profile with a test image
    print("\n1. Creating sample user profile & extracting 2048D ResNet50 embedding...")
    img = Image.new("RGB", (300, 400), color=(50, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    test_image_bytes = buf.getvalue()

    user_profile: UserProfile = create_user_profile(
        image_input=test_image_bytes,
        gender="male",
        occasion="casual",
        preferred_color="blue",
        preferred_style="streetwear",
        body_type="regular",
        skin_tone="cool",
        original_filename="sample_user.jpg"
    )

    print(f"   [OK] User profile created.")
    print(f"   [OK] Image Path: {user_profile.image_path}")
    print(f"   [OK] Embedding Shape: {list(user_profile.get_embedding_numpy().shape)}")

    # 2. Test HybridRecommendationService directly
    print("\n2. Executing Hybrid Recommendation Pipeline...")
    service = get_hybrid_recommendation_service()

    pipeline_start = time.time()
    result = service.recommend(user_profile, top_k=10)
    pipeline_time_ms = (time.time() - pipeline_start) * 1000

    total_retrieved = result["total_candidates_retrieved"]
    candidates_after_filtering = result["candidates_after_filtering"]
    recommendations = result["recommendations"]

    print(f"   [OK] Top K retrieved from FAISS: {total_retrieved}")
    print(f"   [OK] Candidates after hard filtering: {candidates_after_filtering}")
    print(f"   [OK] Recommendations returned: {len(recommendations)}")
    print(f"   [OK] Pipeline Execution Time: {pipeline_time_ms:.2f} ms")

    # 3. Print Top 10 Recommendations with scores
    print("\n------------------ TOP 10 RECOMMENDATIONS ------------------")
    for rank, item in enumerate(recommendations, start=1):
        meta = item["metadata"]
        print(
            f"Rank {rank:2d} | ID: {item['id']:<7} | Name: {item['productDisplayName'][:40]:<40} | "
            f"Sim: {item['similarity']:.4f} | Occ: {item['occasion_score']:.2f} | Style: {item['style_score']:.2f} | "
            f"Color: {item['color_score']:.2f} | Rule: {item['rule_score']:.2f} | Final Score: {item['final_score']:.4f}"
        )

    # 4. Verify every recommendation exists in cleaned_styles.csv
    print("\n3. Verifying recommendations exist in cleaned_styles.csv...")
    csv_path = PROJECT_ROOT / "dataset" / "cleaned_styles.csv"
    df_cleaned = pd.read_csv(csv_path)
    cleaned_ids = set(df_cleaned["id"].astype(str))

    all_exist = True
    for item in recommendations:
        item_id = str(item["id"])
        exists = item_id in cleaned_ids
        if not exists:
            print(f"   [FAIL] Recommendation ID {item_id} NOT found in cleaned_styles.csv!")
            all_exist = False

    assert all_exist, "One or more recommendation IDs were missing from cleaned_styles.csv!"
    print(f"   [OK] All {len(recommendations)} recommendations verified present in cleaned_styles.csv.")

    # 5. Test API Endpoint POST /recommend
    print("\n4. Testing FastAPI POST /recommend endpoint...")
    client = TestClient(app)
    api_response = client.post(
        "/recommend",
        files={"person_image": ("sample_user.jpg", test_image_bytes, "image/jpeg")},
        data={
            "gender": "male",
            "occasion": "casual",
            "preferred_color": "blue",
            "preferred_style": "streetwear",
            "body_type": "regular",
            "skin_tone": "cool"
        }
    )

    print(f"   [OK] API Response Status Code: {api_response.status_code}")
    assert api_response.status_code == 200, f"Expected 200, got {api_response.status_code}: {api_response.text}"

    resp_json = api_response.json()
    assert "user_profile" in resp_json, "Response missing 'user_profile' field"
    assert "recommendations" in resp_json, "Response missing 'recommendations' field"
    assert len(resp_json["recommendations"]) == 10, f"Expected 10 recommendations, got {len(resp_json['recommendations'])}"

    total_elapsed = time.time() - start_time

    print("\n================ VERIFICATION SUMMARY ================")
    print(f"Total Verification Execution Time: {total_elapsed:.2f} seconds")
    print(f"Top K Retrieved: {total_retrieved}")
    print(f"Candidates After Filtering: {candidates_after_filtering}")
    print(f"Top 10 Recommendations Count: {len(resp_json['recommendations'])}")
    print(f"Sample Top 1 Recommendation ID: {resp_json['recommendations'][0]['id']}")
    print(f"Sample Top 1 Product Name: {resp_json['recommendations'][0]['productDisplayName']}")
    print(f"Sample Top 1 Similarity: {resp_json['recommendations'][0]['similarity']}")
    print(f"Sample Top 1 Final Score: {resp_json['recommendations'][0]['final_score']}")
    print("=====================================================")
    print("SUCCESS: Stage 4B Hybrid Recommendation Engine fully verified!\n")


if __name__ == "__main__":
    verify_stage_4b()
