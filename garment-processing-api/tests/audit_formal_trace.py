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

profile = UserProfile(
    image_path="uploads/users/test.jpg",
    embedding=dummy_emb,
    gender="Men",
    occasion="Formal",
    preferred_color="Black",
    preferred_style="Casual",
    body_type="Rectangle",
    skin_tone="Medium"
)

print("=" * 80)
print("--- AUDIT TRACE FOR TEST INPUT ---")
print("User Gender        :", profile.gender)
print("User Occasion      :", profile.occasion)
print("User Color         :", profile.preferred_color)
print("User Style         :", profile.preferred_style)

allowed_genders = service.get_allowed_genders(profile.gender)
allowed_usages = service.get_allowed_usages(profile.occasion)

print("\nAllowed Genders    :", allowed_genders)
print("Allowed Usages     :", allowed_usages)

gender_and_cat = [
    meta for meta in service.styles_dict.values()
    if meta.get("gender", "").lower() in allowed_genders
]
print("\nTotal Catalog Size          :", len(service.styles_dict))
print("Gender Candidates Count     :", len(gender_and_cat))

occ_cands = [
    meta for meta in gender_and_cat
    if service._matches_occasion(meta, allowed_usages)
]
print("Occasion Candidates Count   :", len(occ_cands))

# Check items in metadata.csv with occasion Formal
formal_items = [
    meta for meta in service.styles_dict.values()
    if "formal" in (meta.get("occasion") or "").lower()
]
print(f"Total Items in metadata.csv with occasion containing 'formal': {len(formal_items)}")
for f in formal_items[:10]:
    print(f"  Formal item: ID={f['id']:<5} | product_name={f['product_name']:<50} | gender={f['gender']:<8} | occasion={f['occasion']}")

rec_res = service.recommend(profile, top_k=10)
print("\nTop 10 Recommendations returned by service.recommend():")
for idx, r in enumerate(rec_res['recommendations'], 1):
    meta = r['metadata']
    print(f"#{idx:<2} | ID: {r['id']:<5} | Product: {r['productDisplayName']:<45} | Category: {meta['category']:<10} | Occasion: {meta['occasion']:<15} | Type: {meta['type']:<8} | Gender: {meta['gender']:<8} | Sim: {r['similarity']} | Final: {r['final_score']}")

print("=" * 80)
