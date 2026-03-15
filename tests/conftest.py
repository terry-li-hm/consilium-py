"""Pytest configuration for consilium tests."""
import sys
from pathlib import Path

# Ensure the package is importable from the source tree
sys.path.insert(0, str(Path(__file__).parent.parent))
