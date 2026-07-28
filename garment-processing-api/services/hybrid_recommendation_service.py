"""
Hybrid Recommendation Service for AI Fashion System (Stage 4B).

Combines FAISS 2048D similarity search, metadata hard filtering,
rule-based attribute scoring, and hybrid weighted ranking.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import faiss

from models.user_profile import UserProfile

logger = logging.getLogger(__name__)

# File paths relative to garment-processing-api
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FAISS_INDEX_PATH = DATASET_DIR / "faiss_index.bin"
FAISS_META_PATH = DATASET_DIR / "faiss_metadata.pkl"
CSV_PATH = DATASET_DIR / "cleaned_styles.csv"

# Global lazy singleton instance
_hybrid_service_instance: Optional["HybridRecommendationService"] = None


class HybridRecommendationService:
    """
    Hybrid recommendation engine combining vector search, hard filtering,
    multi-attribute rule scoring, and weighted rank fusion.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = FAISS_META_PATH,
        csv_path: Path = CSV_PATH
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.csv_path = csv_path

        self.index: Optional[faiss.Index] = None
        self.faiss_metadata: Optional[Dict[str, Any]] = None
        self.styles_df: Optional[pd.DataFrame] = None
        self.styles_dict: Dict[str, Dict[str, Any]] = {}
        self.is_loaded: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load FAISS index, metadata pickle, and cleaned styles CSV.
        """
        if self.is_loaded:
            return

        logger.info("Loading FAISS index, metadata, and cleaned dataset for HybridRecommendationService...")

        # 1. Load FAISS index
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index binary not found at: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index: ntotal={self.index.ntotal}, dimension={self.index.d}")

        # 2. Load FAISS metadata pickle
        if not self.meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata pickle not found at: {self.meta_path}")
        with open(self.meta_path, "rb") as f:
            self.faiss_metadata = pickle.load(f)
        logger.info(f"Loaded FAISS metadata pickle (image_ids count: {len(self.faiss_metadata.get('image_ids', []))})")

        # 3. Load Cleaned Styles CSV
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Cleaned styles CSV not found at: {self.csv_path}")
        self.styles_df = pd.read_csv(self.csv_path)

        # Build fast O(1) dictionary lookup by string garment id
        self.styles_dict = {}
        for _, row in self.styles_df.iterrows():
            gid = str(int(row["id"]))
            self.styles_dict[gid] = {
                "id": gid,
                "productDisplayName": str(row["productDisplayName"]) if pd.notna(row["productDisplayName"]) else "Garment",
                "gender": str(row["gender"]) if pd.notna(row["gender"]) else "Unisex",
                "masterCategory": str(row["masterCategory"]) if pd.notna(row["masterCategory"]) else "Apparel",
                "subCategory": str(row["subCategory"]) if pd.notna(row["subCategory"]) else "",
                "articleType": str(row["articleType"]) if pd.notna(row["articleType"]) else "",
                "baseColour": str(row["baseColour"]) if pd.notna(row["baseColour"]) else "",
                "usage": str(row["usage"]) if pd.notna(row["usage"]) else "",
                "season": str(row["season"]) if pd.notna(row["season"]) else "",
                "year": str(int(row["year"])) if pd.notna(row["year"]) and str(row["year"]).replace('.', '').isdigit() else "",
            }

        logger.info(f"Successfully loaded {len(self.styles_dict)} items into garment metadata catalog.")
        self.is_loaded = True

    def faiss_search(
        self,
        user_embedding: np.ndarray,
        candidate_metadata: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 200,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank catalog items with the existing FAISS cosine-similarity index.

        When candidate_metadata is supplied, similarity is calculated only for
        that metadata-filtered pool.  This avoids using a full person-image
        embedding to choose the global garment candidate set while preserving
        the existing ResNet50/FAISS artifacts.

        Args:
            user_embedding: 2048-dimensional feature embedding array.
            candidate_metadata: Optional metadata-first candidate pool.
            top_k: Number of global candidates to retrieve when no pool is supplied.

        Returns:
            List of tuples: (garment_metadata_dict, similarity_score).
        """
        vec = np.array(user_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if candidate_metadata is not None:
            row_positions = self.faiss_metadata.get("row_positions", {})
            candidates = []
            for meta in candidate_metadata:
                row_position = row_positions.get(str(meta["id"]))
                if row_position is None:
                    continue
                # IndexFlatIP stores the normalized catalog vectors.  Their dot
                # product with the normalized query is cosine similarity.
                catalog_vector = self.index.reconstruct(int(row_position))
                similarity = float(np.clip(np.dot(vec[0], catalog_vector), 0.0, 1.0))
                candidates.append((meta, similarity))
            return candidates

        distances, indices = self.index.search(vec, top_k)

        candidates = []
        image_ids_list = self.faiss_metadata["image_ids"]

        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(image_ids_list):
                continue
            gid = str(image_ids_list[idx])
            meta = self.styles_dict.get(gid)
            if meta:
                # Clip cosine similarity to [0.0, 1.0]
                sim = float(np.clip(dist, 0.0, 1.0))
                candidates.append((meta, sim))

        return candidates

    @staticmethod
    def get_allowed_genders(user_gender: str) -> set:
        """
        Map user gender to allowed garment genders.
        Male -> Men + Unisex
        Female -> Women + Unisex
        """
        g_clean = user_gender.strip().lower()
        if g_clean in ["male", "men", "man", "boys", "boy"]:
            return {"men", "unisex", "boys"}
        elif g_clean in ["female", "women", "woman", "girls", "girl"]:
            return {"women", "unisex", "girls"}
        return {"men", "women", "unisex", "boys", "girls"}

    @staticmethod
    def get_allowed_usages(user_occasion: str) -> set:
        """
        Map user occasion to allowed garment usages.
        Office -> Formal, Smart Casual
        Wedding -> Ethnic, Formal
        Casual -> Casual, Travel
        Travel -> Travel, Casual
        Sports -> Sports
        Party -> Party, Casual
        """
        occ_clean = user_occasion.strip().lower()
        if occ_clean in ["office", "work", "business"]:
            return {"formal", "smart casual"}
        elif occ_clean in ["wedding", "ethnic", "traditional"]:
            return {"ethnic", "formal"}
        elif occ_clean in ["casual", "daily", "streetwear"]:
            return {"casual", "travel"}
        elif occ_clean in ["travel", "vacation", "trip"]:
            return {"travel", "casual"}
        elif occ_clean in ["sports", "sport", "gym", "active", "workout"]:
            return {"sports"}
        elif occ_clean in ["party", "nightout", "event", "evening"]:
            return {"party", "casual"}
        return {occ_clean}

    @staticmethod
    def _matches_category(meta: Dict[str, Any], category: Optional[str]) -> bool:
        """Match an optional requested category against catalog category fields."""
        if not category or not category.strip():
            return True
        requested = category.strip().lower()
        values = (
            meta.get("masterCategory", "").lower(),
            meta.get("subCategory", "").lower(),
            meta.get("articleType", "").lower(),
        )
        return any(requested in value or value in requested for value in values if value)

    @staticmethod
    def _style_is_available(style: str) -> bool:
        """Return whether the catalog has a defined rule for this style value."""
        return style.strip().lower() in {
            "streetwear", "casual", "business", "formal", "office",
            "ethnic", "traditional", "athletic", "sports", "sportswear",
            "glamour", "party",
        }

    def _matches_style(self, meta: Dict[str, Any], style: str) -> bool:
        """Use the same metadata style rules for hard candidate generation."""
        return self.compute_style_score(style, meta) >= 0.9

    def generate_metadata_candidates(self, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """
        Build candidates before similarity ranking.

        Gender is immutable.  Style may be relaxed first, and occasion only if
        no gender-safe results exist.  Color remains a soft ranking preference,
        so it is never used to exclude a garment or force a fallback.
        """
        allowed_genders = self.get_allowed_genders(user_profile.gender)
        allowed_usages = self.get_allowed_usages(user_profile.occasion)
        style = (user_profile.preferred_style or "").strip().lower()
        style_is_available = self._style_is_available(style)

        gender_and_category = [
            meta for meta in self.styles_dict.values()
            if meta["gender"].lower() in allowed_genders
            and self._matches_category(meta, user_profile.category)
        ]
        occasion_candidates = [
            meta for meta in gender_and_category
            if meta["usage"].lower() in allowed_usages or not meta["usage"]
        ]
        strict_candidates = [
            meta for meta in occasion_candidates
            if not style_is_available or self._matches_style(meta, style)
        ]
        if strict_candidates:
            return strict_candidates

        # Fallback 1: remove style only; never remove gender/category/occasion.
        if style_is_available and occasion_candidates:
            logger.info("No strict candidates; relaxing style while retaining gender, category, and occasion.")
            return occasion_candidates

        # Fallback 2 (color) is intentionally a no-op: color is always soft.
        # Fallback 3: relax occasion only when no strict gender-safe candidate exists.
        if gender_and_category:
            logger.info("No occasion candidates; relaxing occasion while retaining gender and category.")
            return gender_and_category

        return []

    @staticmethod
    def compute_color_score(preferred_color: str, base_colour: str) -> float:
        """
        Preferred color match: 1.0 if matched, 0.0 otherwise.
        """
        if not preferred_color or not base_colour:
            return 0.0
        p_col = preferred_color.strip().lower()
        b_col = base_colour.strip().lower()
        if p_col in b_col or b_col in p_col:
            return 1.0
        return 0.0

    @staticmethod
    def compute_style_score(preferred_style: str, meta: Dict[str, Any]) -> float:
        """
        Compute style match score using articleType, subCategory, usage, and productDisplayName.
        """
        p_style = preferred_style.strip().lower()
        art_type = meta.get("articleType", "").lower()
        sub_cat = meta.get("subCategory", "").lower()
        usage = meta.get("usage", "").lower()
        title = meta.get("productDisplayName", "").lower()

        # Direct string match
        if p_style in art_type or p_style in sub_cat or p_style in usage or p_style in title:
            return 1.0

        # Category/Style heuristic rules
        if p_style in ["streetwear", "casual"]:
            if usage == "casual" or sub_cat in ["topwear", "bottomwear"]:
                return 0.9
        elif p_style in ["business", "formal", "office"]:
            if usage in ["formal", "smart casual"] or art_type in ["shirts", "blazers", "suits", "trousers"]:
                return 0.9
        elif p_style in ["ethnic", "traditional"]:
            if usage == "ethnic" or art_type in ["kurtas", "sarees", "kurtis", "lehenga choli", "sherwani"]:
                return 0.9
        elif p_style in ["athletic", "sports", "sportswear"]:
            if usage == "sports" or art_type in ["track pants", "tshirts", "t-shirts", "sports shoes"]:
                return 0.9
        elif p_style in ["glamour", "party"]:
            if usage == "party" or art_type in ["dresses", "tops"]:
                return 0.9

        return 0.5

    @staticmethod
    def compute_body_type_score(body_type: str, meta: Dict[str, Any]) -> float:
        """
        Body Type scoring rules:
        Slim -> Slim Fit / Regular Fit
        Athletic -> Sports, Track Pants, Performance Wear
        Regular -> No penalty (1.0)
        """
        b_type = body_type.strip().lower()
        art_type = meta.get("articleType", "").lower()
        title = meta.get("productDisplayName", "").lower()
        usage = meta.get("usage", "").lower()

        if b_type == "slim":
            if "slim" in title or "slim" in art_type:
                return 1.0
            if meta.get("subCategory", "").lower() in ["topwear", "bottomwear"]:
                return 0.8
            return 0.5
        elif b_type in ["athletic", "muscular"]:
            if usage == "sports" or art_type in ["track pants", "tshirts", "t-shirts", "sports shoes"]:
                return 1.0
            return 0.6
        # Regular or unspecified
        return 1.0

    @staticmethod
    def compute_skin_tone_score(skin_tone: str, base_colour: str) -> float:
        """
        Skin Tone color matching rules:
        Warm -> Brown, Beige, Olive, Maroon, Red, Gold, Yellow, Orange, Mustard, Rust
        Cool -> Blue, Purple, Grey, Gray, Silver, Navy Blue, Teal, Pink
        Neutral -> No penalty (1.0)
        """
        s_tone = skin_tone.strip().lower()
        b_color = base_colour.strip().lower()

        warm_colors = {"brown", "beige", "olive", "maroon", "red", "gold", "yellow", "orange", "mustard", "rust"}
        cool_colors = {"blue", "purple", "grey", "gray", "silver", "navy blue", "teal", "pink"}

        if s_tone == "warm":
            if any(w_col in b_color for w_col in warm_colors):
                return 1.0
            return 0.5
        elif s_tone == "cool":
            if any(c_col in b_color for c_col in cool_colors):
                return 1.0
            return 0.5
        # Neutral or unspecified
        return 1.0

    def compute_hybrid_scores(
        self,
        candidates: List[Tuple[Dict[str, Any], float]],
        user_profile: UserProfile
    ) -> List[Dict[str, Any]]:
        """
        Compute multi-attribute scores and calculate final hybrid weighted score:
        Final Score = 0.25 * Similarity + 0.25 * Occasion + 0.20 * Style
                      + 0.15 * Color + 0.15 * Body/Skin suitability.
        """
        allowed_usages = self.get_allowed_usages(user_profile.occasion)
        scored_items = []

        for meta, similarity in candidates:
            # 1. Occasion Score
            occ_score = 1.0 if (meta.get("usage", "").lower() in allowed_usages or not meta.get("usage")) else 0.5

            # 2. Color Score
            color_score = self.compute_color_score(user_profile.preferred_color, meta.get("baseColour", ""))

            # 3. Style Score
            style_score = self.compute_style_score(user_profile.preferred_style, meta)

            # 4. Body Type & Skin Tone Rule Score
            b_score = self.compute_body_type_score(user_profile.body_type, meta)
            s_score = self.compute_skin_tone_score(user_profile.skin_tone, meta.get("baseColour", ""))
            rule_score = round(0.5 * b_score + 0.5 * s_score, 4)

            # 5. Hybrid Weighted Score
            final_score = (
                0.25 * similarity
                + 0.25 * occ_score
                + 0.20 * style_score
                + 0.15 * color_score
                + 0.15 * rule_score
            )
            final_score = round(float(final_score), 4)

            scored_items.append({
                "id": meta["id"],
                "productDisplayName": meta["productDisplayName"],
                "similarity": round(float(similarity), 4),
                "occasion_score": round(float(occ_score), 4),
                "style_score": round(float(style_score), 4),
                "color_score": round(float(color_score), 4),
                "rule_score": round(float(rule_score), 4),
                "final_score": final_score,
                "metadata": {
                    "gender": meta["gender"],
                    "masterCategory": meta["masterCategory"],
                    "subCategory": meta["subCategory"],
                    "articleType": meta["articleType"],
                    "baseColour": meta["baseColour"],
                    "usage": meta["usage"],
                    "season": meta["season"],
                    "year": meta["year"]
                }
            })

        # Sort descending by final_score
        scored_items.sort(key=lambda x: x["final_score"], reverse=True)
        return scored_items

    def recommend(self, user_profile: UserProfile, top_k: int = 10) -> Dict[str, Any]:
        """
        Complete Hybrid Recommendation Pipeline:
        1. Metadata-first candidate generation
        2. FAISS cosine ranking within that pool
        3. Multi-attribute rule scoring
        4. Hybrid weighted ranking
        5. Return up to Top 10 recommendations.

        Returns:
            Dict containing total_candidates_retrieved, candidates_after_filtering, and recommendations list.
        """
        user_emb = user_profile.get_embedding_numpy()

        metadata_candidates = self.generate_metadata_candidates(user_profile)
        candidates_after_filtering = len(metadata_candidates)

        # FAISS similarity ranks only metadata-eligible garments.  The existing
        # person-image score remains a reduced-weight signal until a later
        # garment-region embedding redesign.
        retrieved_candidates = self.faiss_search(
            user_emb,
            candidate_metadata=metadata_candidates,
        )
        total_retrieved = len(retrieved_candidates)

        # Step 5 & 6: Compute Hybrid Scores & Rank Top 10
        ranked_results = self.compute_hybrid_scores(retrieved_candidates, user_profile)
        top_recommendations = ranked_results[:top_k]

        return {
            "total_candidates_retrieved": total_retrieved,
            "candidates_after_filtering": candidates_after_filtering,
            "recommendations": top_recommendations
        }


def get_hybrid_recommendation_service() -> HybridRecommendationService:
    """Get or create singleton HybridRecommendationService instance."""
    global _hybrid_service_instance
    if _hybrid_service_instance is None:
        _hybrid_service_instance = HybridRecommendationService()
    return _hybrid_service_instance
