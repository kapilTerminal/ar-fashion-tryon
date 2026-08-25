import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path("garment-processing-api").resolve()))

from services.hybrid_recommendation_service import get_hybrid_recommendation_service
from models.user_profile import UserProfile
import numpy as np

service = get_hybrid_recommendation_service()

dummy_emb = np.zeros(2048, dtype=np.float32)

tests = [
    {"name": "TEST 1: Men | Casual | Black | Casual", "gender": "Men", "occasion": "Casual", "color": "Black", "style": "Casual"},
    {"name": "TEST 2: Women | Casual | Black | Casual", "gender": "Women", "occasion": "Casual", "color": "Black", "style": "Casual"},
    {"name": "TEST 3: Men | Formal | Black | Casual", "gender": "Men", "occasion": "Formal", "color": "Black", "style": "Casual"},
    {"name": "TEST 4: Women | Formal | Black | Casual", "gender": "Women", "occasion": "Formal", "color": "Black", "style": "Casual"},
    {"name": "TEST 5: Men | Sports | Black | Sporty", "gender": "Men", "occasion": "Sports", "color": "Black", "style": "Sporty"},
]

print("=" * 130)
print("             RECOMMENDATION REDESIGN ARCHITECTURE - 5 CONTROLLED TESTS            ")
print("=" * 130)

for t in tests:
    profile = UserProfile(
        image_path="uploads/users/test.jpg",
        embedding=dummy_emb,
        gender=t["gender"],
        occasion=t["occasion"],
        preferred_color=t["color"],
        preferred_style=t["style"],
        body_type="Rectangle",
        skin_tone="Medium"
    )

    rec_res = service.recommend(profile, top_k=10)
    recs = rec_res["recommendations"]

    print(f"\n>>> {t['name']} (Candidates Filtered: {rec_res['candidates_after_filtering']})")
    print(f"    {'Rank':<5} {'ID':<6} {'Product Name':<42} {'Gender':<8} {'Occasion':<15} {'Category':<10} {'Style':<8} {'Color':<6} {'OccSc':<6} {'GendSc':<6} {'StylSc':<6} {'ColSc':<6} {'RuleSc':<6} {'FinalScore':<10}")
    print("    " + "-" * 145)

    for idx, r in enumerate(recs, 1):
        m = r["metadata"]
        print(f"    #{idx:<4} {r['id']:<6} {r['productDisplayName']:<42} {m['gender']:<8} {m['occasion']:<15} {m['category']:<10} {m['style']:<8} {m['color']:<6} {r['occasion_score']:<6.4f} {r['gender_score']:<6.4f} {r['style_score']:<6.4f} {r['color_score']:<6.4f} {r['rule_score']:<6.4f} {r['final_score']:<10.4f}")

print("\n" + "=" * 130)
print("                         ALL 5 CONTROLLED TESTS COMPLETED                         ")
print("=" * 130)
