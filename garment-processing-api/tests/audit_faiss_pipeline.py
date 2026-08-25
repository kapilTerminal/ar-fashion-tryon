import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path("garment-processing-api").resolve()))

from services.hybrid_recommendation_service import get_hybrid_recommendation_service
from models.user_profile import UserProfile
import numpy as np
import pandas as pd
import faiss

service = get_hybrid_recommendation_service()

print("=" * 80)
print("--- FAISS PIPELINE COMPREHENSIVE AUDIT ---")

# 1. Load actual embeddings.npy
emb_path = Path("garment-processing-api/dataset/fashion_dataset/embeddings.npy")
embeddings = np.load(emb_path)

print(f"\n1. Loaded Dataset Embeddings ('embeddings.npy'):")
print(f"   Shape            : {embeddings.shape}")
print(f"   Dtype            : {embeddings.dtype}")
print(f"   Non-zero values  : {np.count_nonzero(embeddings)} / {embeddings.size}")
print(f"   Min value        : {embeddings.min():.6f}")
print(f"   Max value        : {embeddings.max():.6f}")
print(f"   L2 Norm of #0    : {np.linalg.norm(embeddings[0]):.6f}")

# 2. Inspect FAISS index
index = service.index
meta = service.faiss_metadata

print(f"\n2. FAISS Index Details:")
print(f"   Dimension (d)    : {index.d}")
print(f"   Total Vectors    : {index.ntotal}")
print(f"   Index Type       : {type(index)}")
print(f"   Metadata Keys    : {list(meta.keys())}")
print(f"   Image IDs Count  : {len(meta.get('image_ids', []))}")
print(f"   Row Pos Count    : {len(meta.get('row_positions', {}))}")

# 3. Direct FAISS Search Test with a Real Image Embedding (e.g., embeddings[0])
query_real = embeddings[0].astype(np.float32)
query_real /= np.linalg.norm(query_real)

print(f"\n3. Testing Direct FAISS Search with REAL Image Vector (Item #0, ID={meta['image_ids'][0]}):")
raw_distances, raw_indices = index.search(query_real.reshape(1, -1), 10)

print("   Top 10 Raw FAISS Search Results:")
for i in range(10):
    idx = raw_indices[0][i]
    dist = raw_distances[0][i]
    gid = meta['image_ids'][idx]
    print(f"   Rank #{i+1:<2} | Pos: {idx:<5} | ID: {gid:<5} | Raw Cosine Sim: {dist:.6f}")

# 4. Compare with Random Vector
query_rand = np.random.randn(2048).astype(np.float32)
query_rand /= np.linalg.norm(query_rand)

raw_dist_rand, _ = index.search(query_rand.reshape(1, -1), 10)
print(f"\n4. Testing Direct FAISS Search with RANDOM Vector:")
print(f"   Top 10 Raw Dot Products for Random Vector:")
for i in range(10):
    print(f"   Rank #{i+1:<2} | Raw Dot Product: {raw_dist_rand[0][i]:.6f} | Clipped [0,1]: {max(0.0, float(raw_dist_rand[0][i])):.6f}")

# 5. Full End-to-End Recommendation Test with Real Query Embedding
print("\n5. End-to-End Recommendation Test with REAL Image Vector Query:")
profile_real = UserProfile(
    image_path="uploads/users/test.jpg",
    embedding=query_real,
    gender="Men",
    occasion="Casual",
    preferred_color="Black",
    preferred_style="Casual",
    body_type="Rectangle",
    skin_tone="Medium"
)

rec_res = service.recommend(profile_real, top_k=10)
print(f"   Total Candidates Filtered : {rec_res['candidates_after_filtering']}")
print(f"   Total Candidates Retrieved: {rec_res['total_candidates_retrieved']}")

print("\n   Top 10 Recommendations with REAL Image Query Vector:")
print(f"   {'Rank':<5} {'ID':<6} {'Product Name':<42} {'Gender':<8} {'Occasion':<15} {'RawSim':<8} {'SimScore':<8} {'OccSc':<6} {'StylSc':<6} {'ColSc':<6} {'RuleSc':<6} {'FinalScore':<10}")
print("-" * 145)

for idx, r in enumerate(rec_res["recommendations"], 1):
    m = r["metadata"]
    print(f"   #{idx:<4} {r['id']:<6} {r['productDisplayName']:<42} {m['gender']:<8} {m['occasion']:<15} {r['similarity']:<8.4f} {r['similarity_score']:<8.4f} {r['occasion_score']:<6.4f} {r['style_score']:<6.4f} {r['color_score']:<6.4f} {r['rule_score']:<6.4f} {r['final_score']:<10.4f}")

print("=" * 80)
