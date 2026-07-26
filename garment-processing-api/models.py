"""
Pydantic models for request/response schemas.
"""
from typing import Optional, Dict
from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    model_loaded: bool = False
    model_name: Optional[str] = None
    gradio_connected: bool = False
    services: Optional[Dict[str, str]] = None


class UrlIn(BaseModel):
    source_url: str


class VirtualTryonRequest(BaseModel):
    cloth_type: str = "upper"  # upper, lower, overall
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    seed: int = 42
    show_type: str = "result only"  # result only, input & result, input & mask & result


# -------------------- Recommendation Engine Schemas --------------------
class RecommendationQueryParams(BaseModel):
    occasion: str
    preferred_color: str
    style_preference: str


class TryOnPayload(BaseModel):
    cloth_type: str = "upper"
    process_garment: bool = False


class GarmentRecommendation(BaseModel):
    rank: int
    garment_id: str
    name: str
    category: str
    occasion: str
    color: str
    style: str
    similarity_score: float
    garment_url: str
    cutout_url: str
    tryon_payload: TryOnPayload


class RecommendResponse(BaseModel):
    success: bool = True
    request_id: str
    person_image_url: Optional[str] = None
    query_parameters: RecommendationQueryParams
    total_candidates_found: int
    recommendations: list[GarmentRecommendation]


from models.user_profile import UserProfile, UserProfileVerificationResponse
