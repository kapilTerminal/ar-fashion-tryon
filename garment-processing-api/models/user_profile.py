"""
UserProfile Model for AI Fashion Recommendation System.
"""
from typing import List, Dict, Any, Union
import numpy as np
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """
    Data model representing a user's style profile and image embedding.
    """
    image_path: str
    embedding: Union[List[float], np.ndarray]
    gender: str = "unspecified"
    occasion: str = "casual"
    preferred_color: str = "black"
    preferred_style: str = "streetwear"
    body_type: str = "regular"
    skin_tone: str = "neutral"

    class Config:
        arbitrary_types_allowed = True

    def get_embedding_numpy(self) -> np.ndarray:
        """Return the embedding as a numpy ndarray of shape (2048,)."""
        if isinstance(self.embedding, np.ndarray):
            return self.embedding
        return np.array(self.embedding, dtype=np.float32)

    def to_verification_dict(self) -> Dict[str, Any]:
        """
        Return user profile dictionary formatted for API verification response.
        Excludes full 2048D embedding array and includes shape and dtype instead.
        """
        emb_arr = self.get_embedding_numpy()
        return {
            "image_path": self.image_path,
            "embedding_shape": list(emb_arr.shape),
            "embedding_dtype": str(emb_arr.dtype),
            "gender": self.gender,
            "occasion": self.occasion,
            "preferred_color": self.preferred_color,
            "preferred_style": self.preferred_style,
            "body_type": self.body_type,
            "skin_tone": self.skin_tone,
        }


class UserProfileVerificationResponse(BaseModel):
    """
    Response schema for user profile verification endpoint.
    """
    image_path: str
    embedding_shape: List[int]
    embedding_dtype: str
    gender: str
    occasion: str
    preferred_color: str
    preferred_style: str
    body_type: str
    skin_tone: str
