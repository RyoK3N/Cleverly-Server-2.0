#!/usr/bin/env python3
"""WSGI entry point for production deployment."""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cleverly.app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
