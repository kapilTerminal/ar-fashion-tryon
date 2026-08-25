"""
Hybrid Recommendation Service for AI Fashion System (Stage 4B - Metadata-Based Ranking Architecture).

Combines metadata hard filtering, dataset-backed statistical category suitability,
occasion specificity scoring, gender relevance, style matching, color matching,
and body/skin suitability into a deterministic weighted rank fusion model.
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

# File paths relative to garment-processing-api (Updated for fashion_dataset)
BASE_DIR = Path(__file__).resolve().parent.parent
FASHION_DATASET_DIR = BASE_DIR / "dataset" / "fashion_dataset"
FAISS_INDEX_PATH = FASHION_DATASET_DIR / "faiss_index.bin"
FAISS_META_PATH = FASHION_DATASET_DIR / "faiss_metadata.pkl"
CSV_PATH = FASHION_DATASET_DIR / "metadata.csv"

# Named Scoring Weights (Configurable Default Target Weights: Sum = 1.00)
WEIGHT_OCCASION: float = 0.30
WEIGHT_VISUAL: float = 0.15
WEIGHT_STYLE: float = 0.15
WEIGHT_CATEGORY: float = 0.10
WEIGHT_GENDER: float = 0.10
WEIGHT_COLOR: float = 0.10
WEIGHT_STYLE_PREF: float = 0.05
WEIGHT_BODY_SKIN: float = 0.05

# Global lazy singleton instance
_hybrid_service_instance: Optional["HybridRecommendationService"] = None


class HybridRecommendationService:
    """
    Hybrid recommendation engine combining metadata hard filtering, dataset-backed
    statistical category suitability, occasion specificity scoring, garment visual
    prototype similarity, and multi-attribute deterministic rank fusion.
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
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.is_loaded: bool = False

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load FAISS index, metadata pickle, metadata CSV, and embeddings array.
        """
        if self.is_loaded:
            return

        logger.info("Loading FAISS index, metadata, and dataset for HybridRecommendationService...")

        # 1. Load FAISS index (Preserved intact for future garment-to-garment search)
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index binary not found at: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded FAISS index: ntotal={self.index.ntotal}, dimension={self.index.d}")

        # 1b. Load embeddings matrix if available for fast direct prototype vector ops
        emb_path = FASHION_DATASET_DIR / "embeddings.npy"
        if emb_path.exists():
            try:
                self.embeddings_matrix = np.load(str(emb_path)).astype(np.float32)
                logger.info(f"Loaded embeddings matrix: shape={self.embeddings_matrix.shape}")
            except Exception as e:
                logger.warning(f"Could not load embeddings.npy: {e}")

        # 2. Load FAISS metadata pickle
        if not self.meta_path.exists():
            raise FileNotFoundError(f"FAISS metadata pickle not found at: {self.meta_path}")
        with open(self.meta_path, "rb") as f:
            self.faiss_metadata = pickle.load(f)
        logger.info(f"Loaded FAISS metadata pickle (image_ids count: {len(self.faiss_metadata.get('image_ids', []))})")

        # 3. Populate styles_dict using csv_lookup_mapping or metadata.csv
        self.styles_dict = {}
        if self.faiss_metadata and "csv_lookup_mapping" in self.faiss_metadata:
            logger.info("Populating catalog from faiss_metadata['csv_lookup_mapping']...")
            raw_map = self.faiss_metadata["csv_lookup_mapping"]
            for gid, entry in raw_map.items():
                category = str(entry.get("category", "")) if pd.notna(entry.get("category")) else ""
                occasion = str(entry.get("occasion", "")) if pd.notna(entry.get("occasion")) else ""
                g_type = str(entry.get("type", "")) if pd.notna(entry.get("type")) else ""
                style = str(entry.get("style", "")) if pd.notna(entry.get("style")) else ""
                color = str(entry.get("color", "")) if pd.notna(entry.get("color")) else ""
                gender = str(entry.get("gender", "")) if pd.notna(entry.get("gender")) else ""
                pname = str(entry.get("product_name", "")) if pd.notna(entry.get("product_name")) else ""
                rel_path = str(entry.get("relative_path", "")) if pd.notna(entry.get("relative_path")) else ""
                img_url = str(entry.get("image_url", "")) if pd.notna(entry.get("image_url")) else (f"/images/{rel_path}" if rel_path else f"/images/{gid}")

                self.styles_dict[gid] = {
                    "id": gid,
                    "category": category,
                    "occasion": occasion,
                    "type": g_type,
                    "style": style,
                    "color": color,
                    "gender": gender,
                    "product_name": pname,
                    "relative_path": rel_path,
                    "image_url": img_url,
                    # Backward compatibility aliases
                    "productDisplayName": pname or f"Garment {gid}",
                    "masterCategory": g_type or "Apparel",
                    "subCategory": category,
                    "articleType": category,
                    "baseColour": color,
                    "usage": occasion,
                    "season": "",
                    "year": ""
                }
        elif self.csv_path.exists():
            logger.info(f"Populating catalog from metadata CSV at {self.csv_path}...")
            self.styles_df = pd.read_csv(self.csv_path, dtype={"id": str})
            for _, row in self.styles_df.iterrows():
                gid = str(row["id"]).strip()
                pname = str(row.get("product_name", row.get("productDisplayName", f"Garment {gid}")))
                category = str(row.get("category", row.get("articleType", "")))
                occasion = str(row.get("occasion", row.get("usage", "")))
                g_type = str(row.get("type", row.get("masterCategory", "")))
                style = str(row.get("style", ""))
                color = str(row.get("color", row.get("baseColour", "")))
                gender = str(row.get("gender", ""))
                rel_path = str(row.get("relative_path", pname))
                img_url = f"/images/{rel_path}" if rel_path else f"/images/{gid}"

                self.styles_dict[gid] = {
                    "id": gid,
                    "category": category,
                    "occasion": occasion,
                    "type": g_type,
                    "style": style,
                    "color": color,
                    "gender": gender,
                    "product_name": pname,
                    "relative_path": rel_path,
                    "image_url": img_url,
                    # Backward compatibility aliases
                    "productDisplayName": pname,
                    "masterCategory": g_type,
                    "subCategory": category,
                    "articleType": category,
                    "baseColour": color,
                    "usage": occasion,
                    "season": "",
                    "year": ""
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
        Rank catalog items with FAISS cosine-similarity index (Preserved for garment-to-garment search).
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
                sim = float(np.clip(dist, 0.0, 1.0))
                candidates.append((meta, sim))

        return candidates

    @staticmethod
    def get_allowed_genders(user_gender: str) -> set:
        """
        Map user gender to allowed garment genders.
        Adult Men -> Men, Unisex, Male (Boys excluded)
        Adult Women -> Women, Unisex, Female (Girls excluded)
        Boys -> Boys, Unisex
        Girls -> Girls, Unisex
        """
        g_clean = user_gender.strip().lower()
        if g_clean in ["male", "men", "man"]:
            return {"men", "unisex", "male"}
        elif g_clean in ["female", "women", "woman"]:
            return {"women", "unisex", "female"}
        elif g_clean in ["boys", "boy"]:
            return {"boys", "unisex"}
        elif g_clean in ["girls", "girl"]:
            return {"girls", "unisex"}
        return {"men", "women", "unisex", "boys", "girls", "male", "female"}

    @staticmethod
    def get_allowed_usages(user_occasion: str) -> set:
        """
        Map user occasion to allowed garment usages based on fashion_dataset/metadata.csv tokens:
        ['casual', 'formal_event', 'office', 'party', 'sports', 'traditional_festival']
        """
        occ_clean = user_occasion.strip().lower()
        if occ_clean in ["formal", "office", "work", "business", "formal_event"]:
            return {"formal_event", "office"}
        elif occ_clean in ["wedding", "ethnic", "traditional", "traditional_festival", "festival"]:
            return {"traditional_festival", "formal_event"}
        elif occ_clean in ["casual", "daily", "streetwear", "travel", "vacation", "trip"]:
            return {"casual"}
        elif occ_clean in ["sports", "sport", "sporty", "gym", "active", "workout"]:
            return {"sports"}
        elif occ_clean in ["party", "nightout", "event", "evening"]:
            return {"party"}
        return {occ_clean}

    @staticmethod
    def _matches_category(meta: Dict[str, Any], category: Optional[str]) -> bool:
        """Match an optional requested category against catalog category fields."""
        if not category or not category.strip():
            return True
        requested = category.strip().lower()
        values = (
            meta.get("category", "").lower(),
            meta.get("type", "").lower(),
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
            "ethnic", "traditional", "athletic", "sports", "sporty", "sportswear",
            "glamour", "party",
        }

    def _matches_style(self, meta: Dict[str, Any], style: str) -> bool:
        """Use the same metadata style rules for hard candidate generation."""
        return self.compute_style_score(style, meta) >= 0.85

    @staticmethod
    def _matches_occasion(meta: Dict[str, Any], allowed_usages: set) -> bool:
        """
        Check if candidate garment's pipe-separated occasion values match allowed usages.
        Supports multi-occasion values like 'Casual|Sports', 'Casual|Office', etc.
        """
        raw_occ = meta.get("occasion") or meta.get("usage") or ""
        if not raw_occ:
            return True
        item_occasions = [o.strip().lower() for o in raw_occ.split("|") if o.strip()]
        return any(occ in allowed_usages for occ in item_occasions)

    @staticmethod
    def compute_occasion_specificity(
        user_occasion: str,
        garment_occasion: str,
        allowed_usages: set
    ) -> float:
        """
        Compute deterministic occasion-specificity score in [0, 1].
        Evaluates match strength and purity ratio:
        purity_ratio = |Matching Tokens| / |Total Tokens|
        Specificity = S_match * (0.70 + 0.30 * purity_ratio)

        Formal Example Scores:
        - Office|Formal_Event          : 1.0000 (Exact formal combination, 100% pure)
        - Casual|Office|Formal_Event   : 0.9000 (Formal combination, 67% pure)
        - Office|Formal_Event|Party    : 0.9000 (Formal combination, 67% pure)
        - Casual|Office                : 0.7225 (Office only, 50% pure)
        - Casual                       : 0.0000 (No formal match)
        """
        raw_occ = (garment_occasion or "").strip()
        if not raw_occ:
            return 0.50

        garment_tokens = set(o.strip().lower() for o in raw_occ.split("|") if o.strip())
        if not garment_tokens:
            return 0.50

        matching_tokens = garment_tokens.intersection(allowed_usages)
        if not matching_tokens:
            return 0.00

        u_occ = (user_occasion or "").strip().lower()

        # Determine base match score S_match
        if u_occ in ["formal", "office", "work", "business", "formal_event"]:
            if {"formal_event", "office"}.issubset(matching_tokens):
                s_match = 1.00
            elif "formal_event" in matching_tokens:
                s_match = 0.90
            elif "office" in matching_tokens:
                s_match = 0.85
            else:
                s_match = 0.75
        elif u_occ in ["wedding", "ethnic", "traditional", "traditional_festival", "festival"]:
            if "traditional_festival" in matching_tokens:
                s_match = 1.00
            else:
                s_match = 0.85
        else:
            s_match = 1.00

        purity_ratio = len(matching_tokens) / len(garment_tokens)
        specificity = s_match * (0.70 + 0.30 * purity_ratio)
        return round(float(np.clip(specificity, 0.0, 1.0)), 4)

    def compute_category_suitability_map(
        self,
        user_profile: UserProfile,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute dataset-backed category suitability scores using Laplace smoothing (alpha=1).
        Formula:
        R(C) = [ P(C|O,G) * P(C|S,G) * P(O|C,G) * P(S|C,G) * P(O_intersect_S|C,G) ] ** (1/5)
        CategorySuitability(C) = R(C) / max(R(C'))
        """
        u_gender = (user_profile.gender or "").strip().lower()
        allowed_genders = self.get_allowed_genders(u_gender)
        allowed_usages = self.get_allowed_usages(user_profile.occasion)
        req_style = (user_profile.preferred_style or "").strip().lower()

        gender_catalog = [
            meta for meta in self.styles_dict.values()
            if meta.get("gender", "").strip().lower() in allowed_genders
        ]

        if not gender_catalog:
            gender_catalog = list(self.styles_dict.values())

        all_categories = list(set(meta.get("category", "") for meta in gender_catalog if meta.get("category")))
        all_occasions = list(set(meta.get("occasion", "") for meta in gender_catalog if meta.get("occasion")))
        all_styles = list(set(meta.get("style", "") for meta in gender_catalog if meta.get("style")))

        V_C = max(len(all_categories), 1)
        V_O = max(len(all_occasions), 1)
        V_S = max(len(all_styles), 1)
        V_OS = V_O * V_S

        N_OG = 0
        N_SG = 0
        N_CG: Dict[str, int] = {}
        N_COG: Dict[str, int] = {}
        N_CSG: Dict[str, int] = {}
        N_COSG: Dict[str, int] = {}

        for meta in gender_catalog:
            cat = meta.get("category", "")
            if not cat:
                continue

            occ_match = self._matches_occasion(meta, allowed_usages)
            style_match = (req_style in meta.get("style", "").lower()) or self._matches_style(meta, req_style)

            if occ_match:
                N_OG += 1
            if style_match:
                N_SG += 1

            N_CG[cat] = N_CG.get(cat, 0) + 1

            if occ_match:
                N_COG[cat] = N_COG.get(cat, 0) + 1
            if style_match:
                N_CSG[cat] = N_CSG.get(cat, 0) + 1
            if occ_match and style_match:
                N_COSG[cat] = N_COSG.get(cat, 0) + 1

        r_scores = {}
        candidate_cats = set(meta.get("category", "") for meta in candidates if meta.get("category"))

        alpha = 1.0  # Laplace smoothing factor

        for cat in candidate_cats:
            n_cg = N_CG.get(cat, 0)
            n_cog = N_COG.get(cat, 0)
            n_csg = N_CSG.get(cat, 0)
            n_cosg = N_COSG.get(cat, 0)

            p_c_og = (n_cog + alpha) / (N_OG + alpha * V_C)
            p_c_sg = (n_csg + alpha) / (N_SG + alpha * V_C)
            p_o_cg = (n_cog + alpha) / (n_cg + alpha * V_O)
            p_s_cg = (n_csg + alpha) / (n_cg + alpha * V_S)
            p_os_cg = (n_cosg + alpha) / (n_cg + alpha * V_OS)

            r_val = (p_c_og * p_c_sg * p_o_cg * p_s_cg * p_os_cg) ** (0.2)
            r_scores[cat] = r_val

        max_r = max(r_scores.values()) if r_scores else 0.0

        suitability_map = {}
        for cat in candidate_cats:
            r_val = r_scores.get(cat, 0.0)
            if max_r > 0:
                suitability_map[cat] = round(r_val / max_r, 4)
            else:
                suitability_map[cat] = 1.0000

        return suitability_map

    @staticmethod
    def compute_gender_score(user_gender: str, garment_gender: str) -> float:
        """
        Compute gender relevance score.
        - Exact adult/kids gender match: 1.0
        - Unisex: 0.50
        - Opposite / invalid: 0.0
        """
        u_g = (user_gender or "").strip().lower()
        g_g = (garment_gender or "").strip().lower()

        if not u_g or not g_g:
            return 0.50

        if g_g == "unisex":
            return 0.50

        if u_g in ["men", "male", "man"] and g_g in ["men", "male"]:
            return 1.0
        if u_g in ["women", "female", "woman"] and g_g in ["women", "female"]:
            return 1.0
        if u_g in ["boys", "boy"] and g_g in ["boys", "boy"]:
            return 1.0
        if u_g in ["girls", "girl"] and g_g in ["girls", "girl"]:
            return 1.0

        return 0.0

    def generate_metadata_candidates(self, user_profile: UserProfile) -> List[Dict[str, Any]]:
        """
        Build candidates before similarity ranking.
        Strictly enforces Gender & Occasion hard filtering.
        """
        allowed_genders = self.get_allowed_genders(user_profile.gender)
        allowed_usages = self.get_allowed_usages(user_profile.occasion)
        style = (user_profile.preferred_style or "").strip().lower()
        style_is_available = self._style_is_available(style)

        # Audit gender counts for logging
        u_g = (user_profile.gender or "").strip().lower()
        exact_count = 0
        unisex_count = 0
        excluded_count = 0

        for meta in self.styles_dict.values():
            g = meta.get("gender", "").strip().lower()
            if g in allowed_genders:
                if (u_g in ["men", "male", "man"] and g in ["men", "male"]) or \
                   (u_g in ["women", "female", "woman"] and g in ["women", "female"]) or \
                   (u_g in ["boys", "boy"] and g in ["boys", "boy"]) or \
                   (u_g in ["girls", "girl"] and g in ["girls", "girl"]):
                    exact_count += 1
                elif g == "unisex":
                    unisex_count += 1
            else:
                excluded_count += 1

        logger.info(
            f"Gender Candidate Audit for user_gender='{user_profile.gender}': "
            f"Exact={exact_count}, Unisex={unisex_count}, Excluded={excluded_count}"
        )

        gender_and_category = [
            meta for meta in self.styles_dict.values()
            if meta.get("gender", "").lower() in allowed_genders
            and self._matches_category(meta, user_profile.category)
        ]
        occasion_candidates = [
            meta for meta in gender_and_category
            if self._matches_occasion(meta, allowed_usages)
        ]
        strict_candidates = [
            meta for meta in occasion_candidates
            if not style_is_available or self._matches_style(meta, style)
        ]
        if strict_candidates:
            return strict_candidates

        # Fallback 1: remove style only; retain gender, category, occasion.
        if style_is_available and occasion_candidates:
            logger.info("No strict candidates; relaxing style while retaining gender, category, and occasion.")
            return occasion_candidates

        # Fallback 2: relax occasion only when no strict gender-safe candidate exists.
        if gender_and_category:
            logger.info("No occasion candidates; relaxing occasion while retaining gender and category.")
            return gender_and_category

        return []

    def compute_candidate_visual_similarities(
        self,
        metadata_candidates: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Compute candidate-to-prototype visual cosine similarity.
        1. Retrieves 2048D L2-normalized ResNet50 embeddings for eligible candidate garments.
        2. Computes the normalized centroid (prototype vector) of candidate embeddings.
        3. Computes dot product (cosine similarity) of each candidate embedding against prototype.
        Note: The person image vector is NEVER used for recommendation.
        """
        if not metadata_candidates:
            return {}

        row_positions = self.faiss_metadata.get("row_positions", {}) if self.faiss_metadata else {}
        candidate_vectors = []
        valid_gids = []

        for meta in metadata_candidates:
            gid = str(meta["id"])
            vec = None
            if self.embeddings_matrix is not None and gid in row_positions:
                row_idx = int(row_positions[gid])
                if 0 <= row_idx < len(self.embeddings_matrix):
                    vec = self.embeddings_matrix[row_idx]
            elif self.index is not None and gid in row_positions:
                try:
                    row_idx = int(row_positions[gid])
                    vec = self.index.reconstruct(row_idx)
                except Exception:
                    pass

            if vec is not None:
                vec_arr = np.array(vec, dtype=np.float32)
                norm = np.linalg.norm(vec_arr)
                if norm > 0:
                    vec_arr = vec_arr / norm
                candidate_vectors.append(vec_arr)
                valid_gids.append(gid)

        if not candidate_vectors:
            # Fallback if no vectors available (e.g., in lightweight mock tests)
            return {str(meta["id"]): 0.50 for meta in metadata_candidates}

        # Compute normalized centroid prototype vector
        centroid = np.mean(candidate_vectors, axis=0)
        c_norm = np.linalg.norm(centroid)
        prototype = centroid / c_norm if c_norm > 0 else centroid

        # Candidate-to-prototype cosine similarity
        sim_map = {}
        for gid, vec_arr in zip(valid_gids, candidate_vectors):
            cos_sim = float(np.clip(np.dot(vec_arr, prototype), 0.0, 1.0))
            sim_map[gid] = round(cos_sim, 4)

        # Fill any missing GIDs with neutral default
        for meta in metadata_candidates:
            gid = str(meta["id"])
            if gid not in sim_map:
                sim_map[gid] = 0.50

        return sim_map

    @staticmethod
    def compute_color_score(preferred_color: str, base_colour: str) -> float:
        """
        Compute color match score with explicit fashion color compatibility mapping.
        """
        if not preferred_color or not base_colour:
            return 0.0
        p_col = preferred_color.strip().lower()
        b_col = base_colour.strip().lower()

        # Direct exact or substring match
        if p_col in b_col or b_col in p_col:
            return 1.00

        # Fashion color compatibility rules
        compat_map = {
            ("navy", "blue"): 0.85, ("blue", "navy"): 0.85,
            ("cream", "beige"): 0.85, ("beige", "cream"): 0.85,
            ("gray", "silver"): 0.85, ("grey", "silver"): 0.85,
            ("silver", "gray"): 0.85, ("silver", "grey"): 0.85,
            ("maroon", "red"): 0.85, ("red", "maroon"): 0.85,
            ("black", "gray"): 0.70, ("black", "grey"): 0.70,
            ("black", "navy"): 0.70, ("navy", "black"): 0.70,
            ("gray", "black"): 0.70, ("grey", "black"): 0.70,
            ("multi-color", "black"): 0.60, ("multi-color", "white"): 0.60,
            ("black", "multi-color"): 0.60, ("white", "multi-color"): 0.60,
        }

        return compat_map.get((p_col, b_col), 0.00)

    @staticmethod
    def compute_style_score(preferred_style: str, meta: Dict[str, Any]) -> float:
        """
        Compute style match score using explicit frontend-to-dataset vocabulary mapping.
        """
        p_style = (preferred_style or "").strip().lower()
        if not p_style:
            return 0.50

        meta_style = (meta.get("style") or "").strip().lower()
        if p_style == meta_style:
            return 1.00

        # Vocabulary mapping from frontend options to dataset style tokens
        style_map = {
            "casual": {"casual"},
            "formal": {"formal"},
            "sporty": {"sporty"},
            "streetwear": {"casual"},
            "minimalist": {"casual", "formal"},
            "vintage": {"casual", "party"},
            "elegant": {"formal", "party"},
        }

        mapped_styles = style_map.get(p_style, {p_style})
        if meta_style in mapped_styles:
            return 0.85

        art_type = (meta.get("category") or meta.get("articleType") or "").lower()
        sub_cat = (meta.get("style") or meta.get("subCategory") or "").lower()
        usage = (meta.get("occasion") or meta.get("usage") or "").lower()

        if any(ms in art_type or ms in sub_cat or ms in usage for ms in mapped_styles):
            return 0.85

        return 0.50

    @staticmethod
    def compute_body_type_score(body_type: str, meta: Dict[str, Any]) -> float:
        """Body Type scoring rules."""
        b_type = (body_type or "").strip().lower()
        art_type = (meta.get("category") or meta.get("articleType") or "").lower()
        title = (meta.get("product_name") or meta.get("productDisplayName") or "").lower()
        usage = (meta.get("occasion") or meta.get("usage") or "").lower()

        if b_type == "slim":
            if "slim" in title or "slim" in art_type:
                return 1.0
            if meta.get("type", "").lower() in ["top", "bottom"] or meta.get("subCategory", "").lower() in ["topwear", "bottomwear"]:
                return 0.8
            return 0.5
        elif b_type in ["athletic", "muscular"]:
            if "sports" in usage or "sporty" in usage or art_type in ["track pants", "tshirts", "t-shirts", "joggers"]:
                return 1.0
            return 0.6
        return 1.0

    @staticmethod
    def compute_skin_tone_score(skin_tone: str, base_colour: str) -> float:
        """Skin Tone color matching rules."""
        s_tone = (skin_tone or "").strip().lower()
        b_color = (base_colour or "").strip().lower()

        warm_colors = {"brown", "beige", "olive", "maroon", "red", "gold", "yellow", "orange", "mustard", "rust", "cream"}
        cool_colors = {"blue", "purple", "grey", "gray", "silver", "navy", "navy blue", "teal", "pink"}

        if s_tone == "warm":
            if any(w_col in b_color for w_col in warm_colors):
                return 1.0
            return 0.5
        elif s_tone == "cool":
            if any(c_col in b_color for c_col in cool_colors):
                return 1.0
            return 0.5
        return 1.0

    def compute_hybrid_scores(
        self,
        candidates: List[Tuple[Dict[str, Any], float]],
        user_profile: UserProfile,
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute multi-attribute scores and calculate final hybrid weighted score.

        Configurable Named Weights (Default Sum = 1.00):
        - Occasion Specificity   : 0.30
        - Visual Cosine Sim      : 0.15 (Candidate-to-prototype similarity)
        - Style Score            : 0.15
        - Category Suitability   : 0.10
        - Gender Relevance       : 0.10
        - Color Score            : 0.10
        - Style Preference       : 0.05
        - Body/Skin Suitability  : 0.05
        """
        w = {
            "occasion": WEIGHT_OCCASION,
            "visual": WEIGHT_VISUAL,
            "style": WEIGHT_STYLE,
            "category": WEIGHT_CATEGORY,
            "gender": WEIGHT_GENDER,
            "color": WEIGHT_COLOR,
            "style_pref": WEIGHT_STYLE_PREF,
            "body_skin": WEIGHT_BODY_SKIN,
        }
        if weights:
            w.update(weights)

        allowed_usages = self.get_allowed_usages(user_profile.occasion)
        meta_candidates = [item[0] for item in candidates]

        # 1. Candidate-to-prototype visual similarity map
        visual_sim_map = self.compute_candidate_visual_similarities(meta_candidates)

        # 2. Category suitability map
        cat_suitability_map = self.compute_category_suitability_map(user_profile, meta_candidates)

        scored_items = []

        for meta, _ in candidates:
            cat = meta.get("category", "")
            color_val = meta.get("color") or meta.get("baseColour") or ""
            gid = str(meta["id"])

            # Component Scores
            occ_spec = self.compute_occasion_specificity(
                user_profile.occasion,
                meta.get("occasion") or meta.get("usage") or "",
                allowed_usages
            )
            cat_suit = cat_suitability_map.get(cat, 1.0000)
            g_score = self.compute_gender_score(user_profile.gender, meta.get("gender", ""))
            style_score = self.compute_style_score(user_profile.preferred_style, meta)
            color_score = self.compute_color_score(user_profile.preferred_color, color_val)

            # Visual Cosine Similarity (candidate to eligible candidate prototype)
            vis_score = visual_sim_map.get(gid, 0.50)

            # Style preference bonus
            style_pref_score = 0.85 if user_profile.preferred_style and user_profile.preferred_style.lower() in meta.get("style", "").lower() else 0.50

            # Body/Skin rules
            b_score = self.compute_body_type_score(user_profile.body_type, meta)
            s_score = self.compute_skin_tone_score(user_profile.skin_tone, color_val)
            rule_score = round(0.5 * b_score + 0.5 * s_score, 4)

            # Final Score calculation using named configurable weights
            final_score = (
                w["occasion"] * occ_spec
                + w["visual"] * vis_score
                + w["style"] * style_score
                + w["category"] * cat_suit
                + w["gender"] * g_score
                + w["color"] * color_score
                + w["style_pref"] * style_pref_score
                + w["body_skin"] * rule_score
            )
            final_score = round(float(final_score), 4)

            # Explanations
            explanations = []
            if occ_spec >= 0.85:
                explanations.append(f"Strong occasion match ({user_profile.occasion.capitalize()})")
            if color_score >= 0.85:
                explanations.append(f"Matches preferred color ({color_val.capitalize()})")
            if style_score >= 0.85:
                explanations.append(f"Matches preferred style ({user_profile.preferred_style.capitalize()})")
            if vis_score >= 0.70:
                explanations.append("Visually representative of candidate prototype")
            if g_score >= 0.90:
                explanations.append(f"Compatible with {user_profile.gender.capitalize()}")

            # Image URL resolution
            rel_path = meta.get("relative_path", "")
            img_url = meta.get("image_url") or (f"/images/{rel_path}" if rel_path else f"/images/{meta['id']}")

            # Try-On cloth_type mapping
            g_type = str(meta.get("type", "")).strip().lower()
            if g_type in ["top", "upper"]:
                cloth_type = "upper"
            elif g_type in ["bottom", "lower"]:
                cloth_type = "lower"
            elif g_type in ["full", "overall", "dress"]:
                cloth_type = "overall"
            else:
                cat_lower = str(meta.get("category", "")).strip().lower()
                cloth_type = "lower" if any(b in cat_lower for b in ["bottom", "pant", "trouser", "skirt", "jeans", "shorts", "joggers"]) else ("overall" if any(f in cat_lower for f in ["dress", "full", "gown", "suit"]) else "upper")

            scored_items.append({
                "id": meta["id"],
                "garment_id": meta["id"],
                "productDisplayName": meta.get("product_name") or meta.get("productDisplayName") or f"Garment {meta['id']}",
                "name": meta.get("product_name") or meta.get("productDisplayName") or f"Garment {meta['id']}",
                "similarity": vis_score,
                "similarity_score": vis_score,
                "visual_similarity": vis_score,
                "occasion_specificity": occ_spec,
                "category_suitability": cat_suit,
                "occasion_score": occ_spec,
                "gender_score": round(float(g_score), 4),
                "style_score": round(float(style_score), 4),
                "color_score": round(float(color_score), 4),
                "rule_score": round(float(rule_score), 4),
                "final_score": final_score,
                "image_url": img_url,
                "garment_url": img_url,
                "cutout_url": img_url,
                "explanations": explanations,
                "explanation": "; ".join(explanations) if explanations else "Recommendation based on style profile",
                "tryon_payload": {
                    "cloth_type": cloth_type,
                    "process_garment": False
                },
                "metadata": {
                    "id": meta["id"],
                    "category": meta.get("category", ""),
                    "occasion": meta.get("occasion", meta.get("usage", "")),
                    "type": meta.get("type", ""),
                    "style": meta.get("style", ""),
                    "color": color_val,
                    "gender": meta.get("gender", ""),
                    "product_name": meta.get("product_name", ""),
                    "relative_path": rel_path,
                    "image_url": img_url,
                    # Legacy fields
                    "masterCategory": meta.get("masterCategory", meta.get("type", "")),
                    "subCategory": meta.get("subCategory", meta.get("category", "")),
                    "articleType": meta.get("articleType", meta.get("category", "")),
                    "baseColour": color_val,
                    "usage": meta.get("usage", meta.get("occasion", "")),
                    "season": meta.get("season", ""),
                    "year": meta.get("year", "")
                }
            })

        # Initial deterministic sorting
        scored_items.sort(
            key=lambda x: (
                x["final_score"],
                x["occasion_specificity"],
                x["visual_similarity"],
                x["category_suitability"],
                x["gender_score"],
                x["style_score"],
                x["color_score"],
                -int(str(x["id"])) if str(x["id"]).isdigit() else str(x["id"])
            ),
            reverse=True
        )

        return scored_items

    @staticmethod
    def apply_diversity_control(scored_items: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Apply soft diversity re-ranking to prevent Top K recommendations from being dominated
        by identical (category, color) combinations while preserving primary relevance.
        """
        if not scored_items:
            return []

        selected = []
        remaining = list(scored_items)
        category_counts: Dict[str, int] = {}
        cat_color_counts: Dict[Tuple[str, str], int] = {}

        while remaining and len(selected) < top_k:
            best_idx = 0
            best_adjusted_score = -999.0

            for idx, item in enumerate(remaining):
                cat = str(item.get("metadata", {}).get("category", item.get("category", "")))
                color = str(item.get("metadata", {}).get("color", item.get("color", "")))
                base_score = item["final_score"]

                cat_c = category_counts.get(cat, 0)
                cc_c = cat_color_counts.get((cat, color), 0)

                # Soft diversity penalty: small deduction for excess duplicates
                penalty = (0.02 * max(0, cat_c - 2)) + (0.04 * cc_c)
                adj_score = base_score - penalty

                if adj_score > best_adjusted_score:
                    best_adjusted_score = adj_score
                    best_idx = idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)

            cat = str(chosen.get("metadata", {}).get("category", chosen.get("category", "")))
            color = str(chosen.get("metadata", {}).get("color", chosen.get("color", "")))
            category_counts[cat] = category_counts.get(cat, 0) + 1
            cat_color_counts[(cat, color)] = cat_color_counts.get((cat, color), 0) + 1

        if len(selected) < top_k and remaining:
            selected.extend(remaining[:top_k - len(selected)])

        # Assign 1-indexed ranks
        for rank_idx, item in enumerate(selected, start=1):
            item["rank"] = rank_idx

        return selected

    def recommend(
        self,
        user_profile: UserProfile,
        top_k: int = 10,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Complete Redesigned Recommendation Pipeline.
        1. Metadata-first candidate generation (Gender + Category + Occasion + Style hard filtering)
        2. Statistical Category Suitability + Occasion Specificity calculation
        3. Candidate-to-prototype visual similarity calculation (normalized centroid)
        4. Multi-attribute deterministic rank fusion using named weights
        5. Soft diversity re-ranking to avoid duplicate dominance
        6. Return top_k recommendations.
        """
        metadata_candidates = self.generate_metadata_candidates(user_profile)
        candidates_after_filtering = len(metadata_candidates)

        candidates = [(meta, 0.0) for meta in metadata_candidates]
        total_retrieved = len(candidates)

        ranked_results = self.compute_hybrid_scores(candidates, user_profile, weights=weights)
        top_recommendations = self.apply_diversity_control(ranked_results, top_k=top_k)

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
