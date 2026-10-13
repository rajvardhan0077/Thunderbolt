"""Root entry point for Royal Enfield Thunderbird 350X Smart Garage & Spare Parts Assistance System.
Enables running the application with:
    python app.py
from the project root directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "Backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app  # type: ignore # noqa: E402

if __name__ == "__main__":
    print("=" * 70)
    print("  Royal Enfield Thunderbird 350X Smart Garage Assistance System")
    print("=" * 70)
    print("  Localhost URL: http://127.0.0.1:5000")
    print("  Login Selection: http://127.0.0.1:5000/login")
    print("  - Rider Demo Account:  user_demo   / password123")
    print("  - Garage Demo Account: garage_demo / garage123")
    print("=" * 70)
    app.run(host="0.0.0.0", port=5000, debug=True)
