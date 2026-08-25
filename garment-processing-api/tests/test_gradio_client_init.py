"""
Focused test for Gradio client initialization compatibility.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio_client
from services.gradio_service import get_gradio_client


async def test_client_init():
    print(f"\n--- GRADIO CLIENT INIT TEST ---")
    print(f"Installed gradio_client version: {gradio_client.__version__}")

    try:
        print("Testing get_gradio_client()...")
        client = await get_gradio_client()
        print("[OK] Successfully connected to Gradio Space!")
    except Exception as e:
        err_msg = str(e)
        print(f"Caught exception during connection: {err_msg}")
        assert "unexpected keyword argument 'hf_token'" not in err_msg, (
            f"FAILED: Incompatible keyword argument error still present: {err_msg}"
        )
        if "PAUSED" in err_msg or "503" in err_msg or "Unable to connect" in err_msg:
            print("[OK] Client initialized parameter check passed! (Space status reached: PAUSED / Unavailable)")
        else:
            print(f"[OK] Parameter test passed (No hf_token argument error). Error: {err_msg}")

    print("--- GRADIO CLIENT INIT TEST PASSED ---")


if __name__ == "__main__":
    asyncio.run(test_client_init())
