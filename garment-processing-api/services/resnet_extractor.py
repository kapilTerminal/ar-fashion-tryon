"""
ResNet50 Feature Extractor Service.

Loads a pretrained ResNet50 backbone (without top classification head, with Global Average Pooling),
preprocesses input images, and produces a 2048-dimensional L2-normalized feature embedding vector.
"""
import io
import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global singleton model instance
_resnet_model: Optional[object] = None


def get_resnet_model():
    """
    Lazy load pretrained ResNet50 feature extractor.

    Removes classification top layer (include_top=False) and applies Global Average Pooling (pooling='avg')
    to yield a 2048-dimensional feature embedding vector.
    """
    global _resnet_model

    if _resnet_model is None:
        logger.info("Initializing ResNet50 Feature Extractor (weights='imagenet', pooling='avg')...")
        try:
            import tensorflow as tf
            base_model = tf.keras.applications.ResNet50(
                weights="imagenet",
                include_top=False,
                pooling="avg"
            )
            base_model.trainable = False
            _resnet_model = base_model
            logger.info("✅ ResNet50 Feature Extractor loaded successfully. Output shape: (None, 2048)")
        except Exception as e:
            logger.error(f"❌ Failed to load ResNet50 model: {e}")
            raise RuntimeError(f"ResNet50 loading failed: {e}")

    return _resnet_model


def extract_resnet_features(image_input: Union[Path, str, bytes, Image.Image]) -> np.ndarray:
    """
    Extract a normalized 2048-dimensional feature embedding vector from an input image.

    Args:
        image_input: Path to image, raw image bytes, or PIL Image object.

    Returns:
        1D numpy array of shape (2048,) normalized to unit L2 length (||v||_2 = 1.0).
    """
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input

    # 1. Load image as RGB PIL Image
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 2. Resize to ResNet standard 224x224
    pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)

    # 3. Convert to numpy array and preprocess for ResNet50
    img_array = np.array(pil_img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    img_preprocessed = preprocess_input(img_batch)

    # 4. Extract 2048D feature embedding vector
    model = get_resnet_model()
    raw_features = model.predict(img_preprocessed, verbose=0)[0]  # Shape: (2048,)

    # 5. L2 Normalize feature vector
    norm = np.linalg.norm(raw_features)
    if norm > 0:
        normalized_features = raw_features / norm
    else:
        normalized_features = raw_features

    logger.debug(f"Extracted 2048D embedding vector. Norm: {np.linalg.norm(normalized_features):.4f}")
    return normalized_features
