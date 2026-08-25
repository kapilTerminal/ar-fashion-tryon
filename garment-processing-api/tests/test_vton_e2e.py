"""
End-to-End Test for Virtual Try-On flow through FastAPI /virtual_tryon endpoint.
"""
import sys
import io
import json
from pathlib import Path
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set stdout encoding to utf-8 if possible
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient
from app import app


def test_vton_end_to_end():
    print("\n--- VIRTUAL TRY-ON END-TO-END TEST ---")

    # 1. Create a dummy person image (300x400 RGB)
    person_img = Image.new("RGB", (300, 400), color=(180, 150, 200))
    p_buf = io.BytesIO()
    person_img.save(p_buf, format="JPEG")
    person_bytes = p_buf.getvalue()

    # 2. Create a dummy garment image (200x200 RGB)
    garment_img = Image.new("RGB", (200, 200), color=(220, 50, 50))
    g_buf = io.BytesIO()
    garment_img.save(g_buf, format="JPEG")
    garment_bytes = g_buf.getvalue()

    print("Sending POST /virtual_tryon request...")
    client = TestClient(app)

    response = client.post(
        "/virtual_tryon",
        files={
            "person_image": ("test_person.jpg", person_bytes, "image/jpeg"),
            "garment_image": ("test_garment.jpg", garment_bytes, "image/jpeg")
        },
        data={
            "cloth_type": "upper",
            "num_inference_steps": 10,  # fast test run
            "guidance_scale": 2.5,
            "seed": 42,
            "show_type": "result only",
            "process_garment": "false"  # bypass rembg for fast test
        }
    )

    print(f"Response Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(f"Error Response: {response.text}")
        sys.exit(1)

    json_res = response.json()
    print("\n================ E2E VTO RESPONSE ================")
    print(json.dumps(json_res, indent=2))
    print("==================================================")

    assert json_res.get("success") is True, "Expected success: true"
    assert "result_url" in json_res, "Expected result_url in response"
    print("\n--- VIRTUAL TRY-ON END-TO-END TEST PASSED ---")


if __name__ == "__main__":
    test_vton_end_to_end()
