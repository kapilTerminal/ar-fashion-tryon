"""
Cloudinary upload and management service with local disk fallback.
"""
import io
import os
import logging
from pathlib import Path
from typing import Optional

import cloudinary
import cloudinary.uploader
import requests

from config import CLOUDINARY_CONFIG, MAX_CONTENT_BYTES
from services.image_processing import compress_image_for_upload

logger = logging.getLogger(__name__)

# Base directory for local uploads fallback
UPLOADS_DIR = Path("uploads")


def is_cloudinary_configured() -> bool:
    """Check if valid Cloudinary credentials are set in environment."""
    cloud_name = CLOUDINARY_CONFIG.get("cloud_name")
    api_key = CLOUDINARY_CONFIG.get("api_key")
    api_secret = CLOUDINARY_CONFIG.get("api_secret")

    if not (cloud_name and api_key and api_secret):
        return False

    # Check for placeholder values
    placeholders = {
        "your_cloud_name",
        "your_cloudinary_api_key",
        "your_cloudinary_api_secret",
        "your-api-key",
        "xxx",
    }
    if cloud_name in placeholders or api_key in placeholders or api_secret in placeholders:
        return False

    return True


# Configure Cloudinary if credentials are path-valid
if is_cloudinary_configured():
    logger.info("Initializing Cloudinary storage mode...")
    cloudinary.config(**CLOUDINARY_CONFIG)
else:
    logger.info("Cloudinary credentials not configured. Using local disk storage mode (uploads/).")


def upload_bytes(data: bytes, public_id: str, folder: str, fmt: Optional[str] = None) -> dict:
    """
    Upload bytes to Cloudinary or local disk storage with automatic compression.

    Args:
        data: Image bytes to upload
        public_id: Public ID for the image
        folder: Folder path (e.g., "garments/originals")
        fmt: Optional format (e.g., "png", "jpg")

    Returns:
        Upload response dict containing 'secure_url' and 'public_id'
    """
    logger.info(f"Preparing upload: {len(data)} bytes")
    compressed_data = compress_image_for_upload(data, max_size_mb=9.5)

    if is_cloudinary_configured():
        kwargs = {
            "folder": folder,
            "public_id": public_id,
            "resource_type": "image",
            "overwrite": True,
        }
        if fmt:
            kwargs["format"] = fmt

        logger.info(f"[Cloudinary Mode] Uploading {len(compressed_data)} bytes → {folder}/{public_id}")
        result = cloudinary.uploader.upload(io.BytesIO(compressed_data), **kwargs)
        logger.info(f"[Cloudinary Mode] Upload successful: {result.get('secure_url')}")
        return result

    # Fallback to local disk storage
    ext = fmt.lstrip(".") if fmt else "jpg"
    target_dir = UPLOADS_DIR / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{public_id}.{ext}"
    local_path = target_dir / filename

    with open(local_path, "wb") as f:
        f.write(compressed_data)

    relative_url = f"/uploads/{folder}/{filename}".replace("\\", "/")
    logger.info(f"[Local Disk Mode] Saved {len(compressed_data)} bytes → {local_path} (URL: {relative_url})")

    return {
        "secure_url": relative_url,
        "url": relative_url,
        "public_id": f"{folder}/{public_id}",
        "bytes": len(compressed_data),
        "format": ext,
        "resource_type": "image",
        "storage_mode": "local",
    }


def download_url_bytes(url: str, max_bytes: int = MAX_CONTENT_BYTES) -> bytes:
    """
    Download image from URL or local file path with size limit.

    Args:
        url: Image URL or relative local path to download
        max_bytes: Maximum allowed file size

    Returns:
        Downloaded image bytes

    Raises:
        ValueError: If file exceeds max_bytes
    """
    # Handle local uploads paths directly
    clean_url = url.split("?")[0]
    if clean_url.startswith("/uploads/") or clean_url.startswith("uploads/"):
        local_path = Path(clean_url.lstrip("/"))
        if local_path.exists():
            logger.info(f"Reading image directly from local disk: {local_path}")
            with open(local_path, "rb") as f:
                data = f.read()
            if len(data) > max_bytes:
                raise ValueError("File too large")
            return data

    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()

    total = 0
    chunks = []

    for chunk in r.iter_content(1024 * 64):  # 64KB chunks
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("File too large")
        chunks.append(chunk)

    return b"".join(chunks)