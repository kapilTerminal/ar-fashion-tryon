"""
Verification script for Stage 4A: User Profile Module.
"""
import sys
import os
import io
import json
from pathlib import Path
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set stdout/stderr encoding to utf-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app import app
from services.user_profile_service import create_user_profile
from models.user_profile import UserProfile


def test_stage_4a_user_profile():
    print("\n--- STAGE 4A: USER PROFILE MODULE VERIFICATION ---")

    # 1. Generate a test image in-memory
    img = Image.new("RGB", (300, 400), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    test_image_bytes = buf.getvalue()

    # 2. Test UserProfileService directly
    print("\n1. Testing UserProfileService.create_user_profile()...")
    profile: UserProfile = create_user_profile(
        image_input=test_image_bytes,
        gender="female",
        occasion="party",
        preferred_color="red",
        preferred_style="glam",
        body_type="hourglass",
        skin_tone="warm",
        original_filename="test_person.jpg"
    )

    emb_arr = profile.get_embedding_numpy()
    print(f"[OK] Direct Service Uploaded Image Path: {profile.image_path}")
    print(f"[OK] Direct Service Embedding Shape: {list(emb_arr.shape)}")
    print(f"[OK] Direct Service Embedding Dtype: {emb_arr.dtype}")
    print(f"[OK] File exists on disk: {Path(profile.image_path).exists()}")

    # 3. Test /recommend FastAPI endpoint
    print("\n2. Testing FastAPI POST /recommend endpoint...")
    client = TestClient(app)
    response = client.post(
        "/recommend",
        files={"person_image": ("test_upload.jpg", test_image_bytes, "image/jpeg")},
        data={
            "gender": "female",
            "occasion": "party",
            "preferred_color": "red",
            "preferred_style": "glam",
            "body_type": "hourglass",
            "skin_tone": "warm"
        }
    )

    print(f"[OK] Response Status Code: {response.status_code}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    json_response = response.json()
    
    # 4. Print required output fields
    print("\n================ VERIFICATION OUTPUT ================")
    print(f"Uploaded image path: {json_response.get('image_path')}")
    print(f"Embedding shape: {json_response.get('embedding_shape')}")
    print(f"Embedding dtype: {json_response.get('embedding_dtype')}")
    print("User profile fields:")
    for field in ["gender", "occasion", "preferred_color", "preferred_style", "body_type", "skin_tone"]:
        print(f"  - {field}: {json_response.get(field)}")
    print("\nExample JSON Response:")
    print(json.dumps(json_response, indent=4))
    print("=====================================================")


if __name__ == "__main__":
    test_stage_4a_user_profile()
