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
    {"num": 1, "gender": "Men", "occasion": "Casual", "color": "Black", "style": "Casual"},
    {"num": 2, "gender": "Women", "occasion": "Casual", "color": "Black", "style": "Casual"},
    {"num": 3, "gender": "Men", "occasion": "Formal", "color": "Black", "style": "Casual"},
    {"num": 4, "gender": "Women", "occasion": "Formal", "color": "Black", "style": "Casual"},
    {"num": 5, "gender": "Men", "occasion": "Sports", "color": "Black", "style": "Sporty"},
]

print("=" * 140)
print("                       RECOMMENDATION RANKING QUALITY AUDIT                        ")
print("=" * 140)

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

    print(f"\n>>> TEST {t['num']}: {t['gender']} | {t['occasion']} | {t['color']} | {t['style']} (Candidates: {rec_res['candidates_after_filtering']})")
    print(f"    {'Rk':<3} {'ID':<6} {'Product Name':<42} {'Gend':<7} {'Occasion':<25} {'Category/Type':<18} {'Style':<8} {'Color':<6} {'OccSc':<6} {'GendSc':<6} {'StylSc':<6} {'ColSc':<6} {'RuleSc':<6} {'FinalScore':<10}")
    print("    " + "-" * 155)

    categories = set()
    exact_genders = 0
    exact_colors = 0
    exact_occasions = 0

    for idx, r in enumerate(recs, 1):
        m = r["metadata"]
        cat_type = f"{m['category']}/{m['type']}"
        categories.add(m['category'])

        if (t['gender'].lower() in ['men', 'male'] and m['gender'].lower() in ['men', 'male']) or \
           (t['gender'].lower() in ['women', 'female'] and m['gender'].lower() in ['women', 'female']):
            exact_genders += 1

        if t['color'].lower() in m['color'].lower():
            exact_colors += 1

        if m['occasion'].lower() == t['occasion'].lower():
            exact_occasions += 1

        print(f"    #{idx:<2} {r['id']:<6} {r['productDisplayName']:<42} {m['gender']:<7} {m['occasion']:<25} {cat_type:<18} {m['style']:<8} {m['color']:<6} {r['occasion_score']:<6.4f} {r['gender_score']:<6.4f} {r['style_score']:<6.4f} {r['color_score']:<6.4f} {r['rule_score']:<6.4f} {r['final_score']:<10.4f}")

    print(f"\n    Metrics Summary for Test {t['num']}:")
    print(f"    - Unique Categories   : {len(categories)} ({', '.join(sorted(list(categories)))})")
    print(f"    - Exact Gender Matches : {exact_genders} / 10")
    print(f"    - Exact Color Matches  : {exact_colors} / 10")
    print(f"    - Exact Occasion String: {exact_occasions} / 10 (Single vs Multi-Occasion)")

print("\n" + "=" * 140)
