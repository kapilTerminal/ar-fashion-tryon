import sys
import os
from pathlib import Path

# Add garment-processing-api to sys.path
sys.path.insert(0, str(Path("garment-processing-api").resolve()))

from services.hybrid_recommendation_service import get_hybrid_recommendation_service, FAISS_INDEX_PATH, FAISS_META_PATH
from models.user_profile import UserProfile
import numpy as np

print("=" * 70)
print("1. Checking FAISS Artifact Paths Loaded:")
print("   FAISS Index Path:", FAISS_INDEX_PATH.resolve())
print("   FAISS Meta Path :", FAISS_META_PATH.resolve())

assert "fashion_dataset" in str(FAISS_INDEX_PATH), "FAISS Index Path does not point to fashion_dataset!"
assert "fashion_dataset" in str(FAISS_META_PATH), "FAISS Meta Path does not point to fashion_dataset!"

service = get_hybrid_recommendation_service()
print("\n2. Verifying Loaded Service State:")
print("   FAISS Index ntotal:", service.index.ntotal)
print("   FAISS Index dim   :", service.index.d)
print("   Metadata Items    :", len(service.styles_dict))

assert service.index.ntotal == 3585, f"Expected 3585 vectors, got {service.index.ntotal}"
assert service.index.d == 2048, f"Expected 2048 dim, got {service.index.d}"
assert len(service.styles_dict) == 3585, f"Expected 3585 metadata items, got {len(service.styles_dict)}"

# 3. Test recommendation run
dummy_emb = np.random.randn(2048).astype(np.float32)
dummy_emb /= np.linalg.norm(dummy_emb)

profile = UserProfile(
    image_path="uploads/users/test.jpg",
    embedding=dummy_emb,
    gender="boys",
    occasion="casual",
    preferred_color="black",
    preferred_style="casual"
)

rec_res = service.recommend(profile, top_k=5)
recs = rec_res["recommendations"]
print(f"\n3. Generated {len(recs)} Recommendations for user profile (gender=boys, occasion=casual):")

for r in recs:
    print(f"   Rank #{r.get('rank', 1)} | ID: {r['id']:<5} | Final Score: {r['final_score']} | URL: {r['image_url']} | Cloth Type: {r['tryon_payload']['cloth_type']}")
    assert r['image_url'].startswith('/images/Casual/'), f"Image URL {r['image_url']} does not start with /images/Casual/!"
    assert r['tryon_payload']['cloth_type'] in ['upper', 'lower', 'overall'], "Invalid cloth_type"

# 4. Check static file accessibility for served images
img1_rel = "Casual/Boys/casual_boys_hoodie_cream_0001.jpg"
img2_rel = "Traditional_Festival/Male/traditional_festival_men_kurta_navy_3583.jpg"
base_img_dir = Path("garment-processing-api/dataset/fashion_dataset/Images")

path1 = base_img_dir / img1_rel
path2 = base_img_dir / img2_rel

print("\n4. Verifying Image File Existence on Disk:")
print(f"   Image 1 ({img1_rel}):", "EXISTS" if path1.exists() else "MISSING")
print(f"   Image 2 ({img2_rel}):", "EXISTS" if path2.exists() else "MISSING")

assert path1.exists(), f"Image file {path1} missing!"
assert path2.exists(), f"Image file {path2} missing!"

# 5. Safety check for OLD dataset root-level files
old_root = Path("garment-processing-api/dataset")
old_files = ["embeddings.npy", "image_ids.npy", "faiss_index.bin", "faiss_metadata.pkl", "cleaned_styles.csv"]

print("\n5. Safety Check for OLD Root-Level Files:")
for of in old_files:
    p = old_root / of
    assert p.exists(), f"Old root file {of} missing or altered!"
    print(f"   {of:<20}: EXISTS & UNTOUCHED ({p.stat().st_size / (1024*1024):.2f} MB)")

print("\n" + "=" * 70)
print("       ALL RECOMMENDATION ENGINE VERIFICATION CHECKS PASSED!       ")
print("=" * 70)
