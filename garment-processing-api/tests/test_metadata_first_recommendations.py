"""Focused regression tests for metadata-first recommendation candidates."""
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.user_profile import UserProfile
from services.hybrid_recommendation_service import HybridRecommendationService


def garment(item_id: str, gender: str, usage: str, article_type: str, colour: str = "Black"):
    return {
        "id": item_id,
        "productDisplayName": f"{gender} {article_type}",
        "gender": gender,
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": article_type,
        "baseColour": colour,
        "usage": usage,
        "season": "Summer",
        "year": "2024",
    }


class FakeIndex:
    def __init__(self):
        self.vectors = {
            0: np.array([1.0, 0.0], dtype=np.float32),
            1: np.array([0.0, 1.0], dtype=np.float32),
        }

    def reconstruct(self, position):
        return self.vectors[position]


class MetadataFirstRecommendationTests(unittest.TestCase):
    def setUp(self):
        # Bypass artifact loading: these tests exercise filtering/ranking logic only.
        self.service = HybridRecommendationService.__new__(HybridRecommendationService)
        self.service.styles_dict = {
            "1": garment("1", "Men", "Formal", "Shirts"),
            "2": garment("2", "Women", "Formal", "Dresses"),
        }

    def profile(self, **overrides):
        values = {
            "image_path": "person.jpg",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "gender": "men",
            "occasion": "formal",
            "preferred_color": "black",
            "preferred_style": "casual",
            "body_type": "regular",
            "skin_tone": "neutral",
        }
        values.update(overrides)
        return UserProfile(**values)

    def test_style_fallback_never_relaxes_gender(self):
        # Casual style has no match in this tiny formal pool. Style may relax,
        # but the Women dress must never be allowed into a men-targeted result.
        candidates = self.service.generate_metadata_candidates(self.profile())

        self.assertEqual([item["id"] for item in candidates], ["1"])
        self.assertLess(len(candidates), 10)
        self.assertTrue(all(item["gender"] == "Men" for item in candidates))

    def test_category_remains_hard_constraint(self):
        candidates = self.service.generate_metadata_candidates(self.profile(category="dresses"))

        self.assertEqual(candidates, [])

    def test_faiss_scores_only_supplied_candidate_pool(self):
        self.service.index = FakeIndex()
        self.service.faiss_metadata = {"row_positions": {"1": 0, "2": 1}}

        results = self.service.faiss_search(
            np.array([1.0, 0.0], dtype=np.float32),
            candidate_metadata=[self.service.styles_dict["1"]],
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0]["id"], "1")
        self.assertEqual(results[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
