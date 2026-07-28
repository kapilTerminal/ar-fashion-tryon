"""
User Profile Service.

Handles saving uploaded user images into `uploads/users/`, extracting 2048-dimensional
ResNet50 feature embeddings, and constructing UserProfile objects.
"""
import io
import os
import uuid
import logging
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import numpy as np

from models.user_profile import UserProfile
from services.resnet_extractor import extract_resnet_features

logger = logging.getLogger(__name__)

# Base directory for user image uploads
USER_UPLOADS_DIR = Path("uploads/users")


def create_user_profile(
    image_input: Union[bytes, str, Path, Image.Image],
    gender: str = "unspecified",
    occasion: str = "casual",
    preferred_color: str = "black",
    preferred_style: str = "streetwear",
    category: Optional[str] = None,
    body_type: str = "regular",
    skin_tone: str = "neutral",
    original_filename: Optional[str] = None
) -> UserProfile:
    """
    Creates a UserProfile object from an uploaded user image and preference attributes.

    Workflow:
    1. Ensures `uploads/users/` directory exists.
    2. Saves the uploaded image to `uploads/users/<filename>`.
    3. Extracts a 2048-dimensional L2-normalized feature embedding via ResNet50.
    4. Constructs and returns a UserProfile instance.

    Args:
        image_input: Raw image bytes, file path, or PIL Image.
        gender: User gender preference.
        occasion: Target occasion.
        preferred_color: Preferred color.
        preferred_style: Preferred fashion style.
        body_type: Body type description.
        skin_tone: Skin tone description.
        original_filename: Optional original name of uploaded file.

    Returns:
        UserProfile object containing image_path, embedding, and user attributes.
    """
    # 1. Ensure upload directory exists
    USER_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Read PIL Image to validate and standardize saving
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        raw_bytes = image_input
    elif isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
        with open(image_input, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        raw_bytes = buf.getvalue()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 3. Generate unique filename and save image to uploads/users/
    ext = ".jpg"
    if original_filename:
        orig_ext = Path(original_filename).suffix.lower()
        if orig_ext in [".jpg", ".jpeg", ".png", ".webp"]:
            ext = orig_ext

    filename = f"user_{uuid.uuid4().hex[:10]}{ext}"
    saved_path = USER_UPLOADS_DIR / filename
    
    pil_img.save(saved_path)
    relative_image_path = str(saved_path.as_posix())
    logger.info(f"Saved uploaded user image to: {relative_image_path}")

    # 4. Extract 2048-dimensional ResNet50 feature embedding vector
    embedding: np.ndarray = extract_resnet_features(raw_bytes)
    logger.info(f"Generated ResNet50 embedding vector for user profile (shape: {embedding.shape}, dtype: {embedding.dtype})")

    # 5. Build UserProfile object
    profile = UserProfile(
        image_path=relative_image_path,
        embedding=embedding,
        gender=gender,
        occasion=occasion,
        preferred_color=preferred_color,
        preferred_style=preferred_style,
        category=category,
        body_type=body_type,
        skin_tone=skin_tone
    )

    return profile
