"""
Test script for local storage upload fallback when Cloudinary credentials are not configured.
"""
import sys
import io
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.cloudinary_service import upload_bytes, is_cloudinary_configured, download_url_bytes


def test_local_storage_upload():
    print("\n--- LOCAL STORAGE UPLOAD TEST ---")

    print(f"Is Cloudinary Configured: {is_cloudinary_configured()}")

    # 1. Create a dummy image in memory
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # 2. Call upload_bytes
    print("Calling upload_bytes()...")
    res = upload_bytes(
        data=img_bytes,
        public_id="test_image_123",
        folder="test_folder",
        fmt="jpg"
    )

    print(f"[OK] Returned Dictionary: {res}")
    assert "secure_url" in res, "secure_url missing from upload response"
    assert "public_id" in res, "public_id missing from upload response"
    assert res["public_id"] == "test_folder/test_image_123", f"Unexpected public_id: {res['public_id']}"

    # 3. Check local file creation on disk
    expected_local_file = Path("uploads/test_folder/test_image_123.jpg")
    print(f"Checking local file on disk: {expected_local_file}")
    assert expected_local_file.exists(), f"File was not created on disk at {expected_local_file}"
    print(f"[OK] Local file exists on disk with size: {expected_local_file.stat().st_size} bytes")

    # 4. Test download_url_bytes reading local file
    downloaded_bytes = download_url_bytes(res["secure_url"])
    assert len(downloaded_bytes) > 0, "Downloaded bytes are empty"
    print(f"[OK] download_url_bytes successfully read local file: {len(downloaded_bytes)} bytes")

    # Cleanup test file
    expected_local_file.unlink(missing_ok=True)
    print("--- LOCAL STORAGE UPLOAD TEST PASSED ---")


if __name__ == "__main__":
    test_local_storage_upload()
