"""
Recommendation Service Module.

Handles candidate filtering by occasion, color, and style,
calculates Cosine Similarity against 2048-dimensional ResNet50 embeddings,
and selects the top-K recommended garments ready for Virtual Try-On.
"""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from models.user_profile import UserProfile
from services.hybrid_recommendation_service import get_hybrid_recommendation_service, HybridRecommendationService

logger = logging.getLogger(__name__)


def recommend_hybrid_garments(user_profile: UserProfile, top_k: int = 10) -> Dict[str, Any]:
    """
    Delegate recommendation generation to HybridRecommendationService (Stage 4B).
    """
    hybrid_service = get_hybrid_recommendation_service()
    return hybrid_service.recommend(user_profile, top_k=top_k)



def _generate_synthetic_embedding(seed_val: int) -> np.ndarray:
    """
    Generate a deterministic normalized 2048D embedding vector for catalog items.
    Used when pre-computed embeddings are stored in memory or JSON.
    """
    rng = np.random.RandomState(seed_val)
    vec = rng.randn(2048).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# -------------------- Sample Clothing Dataset --------------------
# Extensible catalog dataset with structured metadata and 2048D feature embeddings
GARMENT_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "garm_101",
        "name": "Oversized Black Streetwear Hoodie",
        "category": "upper",
        "occasion": "casual",
        "color": "black",
        "style": "streetwear",
        "image_url": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1556905055-8f358a7a47b2?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(101)
    },
    {
        "id": "garm_102",
        "name": "Classic White Oxford Shirt",
        "category": "upper",
        "occasion": "formal",
        "color": "white",
        "style": "business",
        "image_url": "https://images.unsplash.com/photo-1598033129183-c4f50c736f10?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1598033129183-c4f50c736f10?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(102)
    },
    {
        "id": "garm_103",
        "name": "Slim Fit Navy Blue Blazer",
        "category": "upper",
        "occasion": "formal",
        "color": "blue",
        "style": "business",
        "image_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(103)
    },
    {
        "id": "garm_104",
        "name": "Black Urban Graphic T-Shirt",
        "category": "upper",
        "occasion": "casual",
        "color": "black",
        "style": "streetwear",
        "image_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(104)
    },
    {
        "id": "garm_105",
        "name": "Red Party Evening Top",
        "category": "upper",
        "occasion": "party",
        "color": "red",
        "style": "glamour",
        "image_url": "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1518895949257-7621c3c786d7?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(105)
    },
    {
        "id": "garm_106",
        "name": "Black Athletic Performance Tee",
        "category": "upper",
        "occasion": "sport",
        "color": "black",
        "style": "athletic",
        "image_url": "https://images.unsplash.com/photo-1518310383802-640c2de311b2?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1518310383802-640c2de311b2?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(106)
    },
    {
        "id": "garm_107",
        "name": "Minimalist White Crewneck Sweater",
        "category": "upper",
        "occasion": "casual",
        "color": "white",
        "style": "minimalist",
        "image_url": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(107)
    },
    {
        "id": "garm_108",
        "name": "Black Tailored Dress Trousers",
        "category": "lower",
        "occasion": "formal",
        "color": "black",
        "style": "business",
        "image_url": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(108)
    },
    {
        "id": "garm_109",
        "name": "Black Minimalist Denim Jacket",
        "category": "upper",
        "occasion": "casual",
        "color": "black",
        "style": "streetwear",
        "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(109)
    },
    {
        "id": "garm_110",
        "name": "Blue Casual Polo Shirt",
        "category": "upper",
        "occasion": "casual",
        "color": "blue",
        "style": "minimalist",
        "image_url": "https://images.unsplash.com/photo-1625910513413-7fc259599540?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1625910513413-7fc259599540?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(110)
    },
    {
        "id": "garm_111",
        "name": "Black Sequined Party Top",
        "category": "upper",
        "occasion": "party",
        "color": "black",
        "style": "glamour",
        "image_url": "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(111)
    },
    {
        "id": "garm_112",
        "name": "Black Urban Cargo Pants",
        "category": "lower",
        "occasion": "casual",
        "color": "black",
        "style": "streetwear",
        "image_url": "https://images.unsplash.com/photo-1517445312882-bc9910d016b7?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1517445312882-bc9910d016b7?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(112)
    },
    {
        "id": "garm_113",
        "name": "Emerald Green Formal Suit Jacket",
        "category": "upper",
        "occasion": "formal",
        "color": "green",
        "style": "business",
        "image_url": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(113)
    },
    {
        "id": "garm_114",
        "name": "White Summer Linen Shirt",
        "category": "upper",
        "occasion": "casual",
        "color": "white",
        "style": "minimalist",
        "image_url": "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1602810318383-e386cc2a3ccf?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(114)
    },
    {
        "id": "garm_115",
        "name": "Black Oversized Zip Hoodie",
        "category": "upper",
        "occasion": "casual",
        "color": "black",
        "style": "streetwear",
        "image_url": "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?w=800&auto=format&fit=crop",
        "cutout_url": "https://images.unsplash.com/photo-1509967419530-da38b4704bc6?w=800&auto=format&fit=crop",
        "embedding": _generate_synthetic_embedding(115)
    }
]


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute Cosine Similarity between two 1D embedding vectors.
    Since input vectors are L2-normalized, similarity = dot product vec_a . vec_b
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clip to valid cosine range [-1.0, 1.0]
    return max(-1.0, min(1.0, sim))


def get_recommendations(
    person_embedding: np.ndarray,
    occasion: str,
    preferred_color: str,
    style_preference: str,
    top_k: int = 5
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter garments by occasion, color, and style, then rank using Cosine Similarity.

    Args:
        person_embedding: 2048D ResNet50 feature vector of person image.
        occasion: Target occasion string (e.g., 'casual', 'formal').
        preferred_color: Preferred color string (e.g., 'black', 'white').
        style_preference: Preferred style string (e.g., 'streetwear', 'business').
        top_k: Number of top recommendations to return (default 5).

    Returns:
        Tuple of (ranked_recommendations_list, total_candidate_count).
    """
    occ_clean = occasion.strip().lower()
    color_clean = preferred_color.strip().lower()
    style_clean = style_preference.strip().lower()

    logger.info(f"Filtering candidates for occasion='{occ_clean}', color='{color_clean}', style='{style_clean}'")

    # Step 1: Strict filtering by occasion AND color
    candidates = [
        item for item in GARMENT_CATALOG
        if item["occasion"].lower() == occ_clean and item["color"].lower() == color_clean
    ]

    # Step 2: Fallback if strict filter yields fewer than top_k items
    if len(candidates) < top_k:
        logger.info(f"Strict filter yielded {len(candidates)} items (< {top_k}). Relaxing color filter...")
        # Include items matching occasion OR color
        candidates = [
            item for item in GARMENT_CATALOG
            if item["occasion"].lower() == occ_clean or item["color"].lower() == color_clean
        ]

    # Step 3: Global fallback if still fewer than top_k items
    if len(candidates) < top_k:
        logger.info("Relaxing all filters to ensure Top-5 candidates exist...")
        candidates = list(GARMENT_CATALOG)

    total_candidates = len(candidates)

    # Step 4: Calculate Cosine Similarity & rank
    scored_candidates = []
    for item in candidates:
        sim = calculate_cosine_similarity(person_embedding, item["embedding"])

        # Add bonus weighting for exact style match (0.05 bonus)
        if item["style"].lower() == style_clean:
            sim = min(1.0, sim + 0.05)

        scored_candidates.append({
            "garment_id": item["id"],
            "name": item["name"],
            "category": item["category"],
            "occasion": item["occasion"],
            "color": item["color"],
            "style": item["style"],
            "similarity_score": round(float(sim), 4),
            "garment_url": item["image_url"],
            "cutout_url": item["cutout_url"],
            "tryon_payload": {
                "cloth_type": item["category"] if item["category"] in ["upper", "lower", "overall"] else "upper",
                "process_garment": False
            }
        })

    # Step 5: Sort descending by similarity score
    scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=False)
    # Re-sort descending
    scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

    # Assign 1-indexed ranks and take Top K
    top_recommendations = []
    for idx, rec in enumerate(scored_candidates[:top_k], start=1):
        rec["rank"] = idx
        top_recommendations.append(rec)

    logger.info(f"Successfully generated top {len(top_recommendations)} recommendations (Top score: {top_recommendations[0]['similarity_score']})")
    return top_recommendations, total_candidates
