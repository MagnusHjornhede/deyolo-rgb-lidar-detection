# tests/conftest.py
import sys
from pathlib import Path

# Resolve repo root no matter where pytest is run from
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
