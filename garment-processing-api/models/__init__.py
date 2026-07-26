"""
Models package for garment processing API.
Re-exports schemas from models.py and user_profile.py.
"""
import importlib.util
import os
import sys

# Load parent models.py schemas if present
_models_py_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.py")
if os.path.exists(_models_py_path):
    spec = importlib.util.spec_from_file_location("_models_py", _models_py_path)
    if spec and spec.loader:
        _models_py = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_models_py)
        for _attr in dir(_models_py):
            if not _attr.startswith("_"):
                globals()[_attr] = getattr(_models_py, _attr)

from models.user_profile import UserProfile, UserProfileVerificationResponse
