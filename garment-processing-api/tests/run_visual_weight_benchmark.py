"""
Deterministic Quality Benchmark & Visual Weight Validation Script.

Compares recommendations across visual similarity weights:
- 0% (Before / Metadata only)
- 10%
- 15% (Initial hypothesis)

For 5 target user profiles:
1. Men + Casual + Black + Casual
2. Women + Formal + Black + Formal
3. Men + Sports + Blue + Sporty
4. Women + Party + Red + Party
5. Unisex + Casual + White + Casual
"""
import sys
import os
from pathlib import Path
import numpy as np

# Ensure garment-processing-api root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.user_profile import UserProfile
from services.hybrid_recommendation_service import get_hybrid_recommendation_service

PROFILES = [
    {
        "name": "Profile 1: Men + Casual + Black + Casual",
        "gender": "Men",
        "occasion": "Casual",
        "color": "Black",
        "style": "Casual",
    },
    {
        "name": "Profile 2: Women + Formal + Black + Formal",
        "gender": "Women",
        "occasion": "Formal",
        "color": "Black",
        "style": "Formal",
    },
    {
        "name": "Profile 3: Men + Sports + Blue + Sporty",
        "gender": "Men",
        "occasion": "Sports",
        "color": "Blue",
        "style": "Sporty",
    },
    {
        "name": "Profile 4: Women + Party + Red + Party",
        "gender": "Women",
        "occasion": "Party",
        "color": "Red",
        "style": "Party",
    },
    {
        "name": "Profile 5: Unisex + Casual + White + Casual",
        "gender": "Unisex",
        "occasion": "Casual",
        "color": "White",
        "style": "Casual",
    },
]


def run_benchmark():
    service = get_hybrid_recommendation_service()

    print("=" * 100)
    print("        FASHION RECOMMENDATION SYSTEM — VISUAL WEIGHT & BEFORE/AFTER QUALITY BENCHMARK        ")
    print("=" * 100)

    # Weight Configurations
    weight_configs = {
        "0% (BEFORE - Metadata Only)": {
            "occasion": 0.30, "visual": 0.00, "style": 0.20, "category": 0.20,
            "gender": 0.15, "color": 0.10, "style_pref": 0.00, "body_skin": 0.05
        },
        "10% (Visual Weight = 10%)": {
            "occasion": 0.32, "visual": 0.10, "style": 0.16, "category": 0.11,
            "gender": 0.11, "color": 0.10, "style_pref": 0.05, "body_skin": 0.05
        },
        "15% (AFTER - Visual Weight = 15%)": {
            "occasion": 0.30, "visual": 0.15, "style": 0.15, "category": 0.10,
            "gender": 0.10, "color": 0.10, "style_pref": 0.05, "body_skin": 0.05
        },
    }

    benchmark_results = {}

    for p in PROFILES:
        print("\n" + "#" * 100)
        print(f"PROFILE: {p['name']}")
        print(f"Preferences: Gender={p['gender']}, Occasion={p['occasion']}, Color={p['color']}, Style={p['style']}")
        print("#" * 100)

        user_profile = UserProfile(
            image_path="uploads/users/benchmark_user.jpg",
            embedding=np.zeros(2048, dtype=np.float32),
            gender=p["gender"],
            occasion=p["occasion"],
            preferred_color=p["color"],
            preferred_style=p["style"],
            body_type="regular",
            skin_tone="medium"
        )

        p_results = {}

        for config_name, w_dict in weight_configs.items():
            print(f"\n--- CONFIG: {config_name} ---")
            res = service.recommend(user_profile, top_k=10, weights=w_dict)
            recs = res["recommendations"]

            # Calculate compliance & metrics
            gender_pass = 0
            occ_pass = 0
            color_pass = 0
            style_pass = 0
            categories_seen = set()
            color_cat_pairs = set()

            allowed_genders = service.get_allowed_genders(p["gender"])
            allowed_usages = service.get_allowed_usages(p["occasion"])

            print(f"Fallback Stage Used: {'Stage 1 (Strict)' if res['candidates_after_filtering'] > 0 else 'Fallback'}")
            print(f"Candidates Filtered: {res['candidates_after_filtering']}")
            print(f"{'Rank':<5} | {'ID':<6} | {'Name':<35} | {'Category':<12} | {'Type':<8} | {'Style':<10} | {'Color':<10} | {'Gender':<8} | {'Occasion':<20} | {'VisSim':<6} | {'FinalScore':<10}")
            print("-" * 150)

            for item in recs:
                meta = item["metadata"]
                g_ok = meta.get("gender", "").lower() in allowed_genders
                o_ok = service._matches_occasion(meta, allowed_usages)
                c_ok = service.compute_color_score(p["color"], meta.get("color", "")) >= 0.70
                s_ok = service.compute_style_score(p["style"], meta) >= 0.85

                if g_ok: gender_pass += 1
                if o_ok: occ_pass += 1
                if c_ok: color_pass += 1
                if s_ok: style_pass += 1

                cat = meta.get("category", "")
                col = meta.get("color", "")
                categories_seen.add(cat)
                color_cat_pairs.add((cat, col))

                vis_str = f"{item.get('visual_similarity', 0.0):.3f}"
                score_str = f"{item['final_score']:.4f}"
                pname = (meta.get("product_name") or item["productDisplayName"])[:33]

                print(f"{item['rank']:<5} | {str(item['id']):<6} | {pname:<35} | {cat:<12} | {meta.get('type',''):<8} | {meta.get('style',''):<10} | {col:<10} | {meta.get('gender',''):<8} | {meta.get('occasion',''):<20} | {vis_str:<6} | {score_str:<10}")

            metrics = {
                "top_10": recs,
                "gender_compliance": f"{gender_pass}/10 ({gender_pass*10}%)",
                "occasion_compliance": f"{occ_pass}/10 ({occ_pass*10}%)",
                "color_match_rate": f"{color_pass}/10 ({color_pass*10}%)",
                "style_match_rate": f"{style_pass}/10 ({style_pass*10}%)",
                "distinct_categories": len(categories_seen),
                "distinct_color_category_pairs": len(color_cat_pairs),
                "fallback_stage": "Stage 1 (Strict)" if res['candidates_after_filtering'] > 0 else "Fallback"
            }
            p_results[config_name] = metrics

            print(f"\nSummary Metrics for {config_name}:")
            print(f"  Gender Compliance     : {metrics['gender_compliance']}")
            print(f"  Occasion Compliance   : {metrics['occasion_compliance']}")
            print(f"  Color Match Rate      : {metrics['color_match_rate']}")
            print(f"  Style Match Rate       : {metrics['style_match_rate']}")
            print(f"  Category Diversity    : {metrics['distinct_categories']} distinct categories in Top 10")
            print(f"  Color-Category Pairs  : {metrics['distinct_color_category_pairs']} distinct combinations in Top 10")

        benchmark_results[p["name"]] = p_results

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()
