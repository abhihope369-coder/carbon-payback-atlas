"""Step 1: download and clean the electricity dataset.

Usage: python scripts/build_data.py [--force-download]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import META_FILE, build  # noqa: E402

if __name__ == "__main__":
    panel = build(force_download="--force-download" in sys.argv)
    manifest = json.loads(META_FILE.read_text())
    f = manifest["filtering"]
    print(f"Rows: {len(panel):,}  Years: {manifest['year_range']}")
    print(f"Countries with intensity data: {f['countries_with_any_intensity']}")
    print(f"Countries in model universe:   {f['countries_in_model_universe']}")
    print(f"Countries with internal gaps:  {len(f['countries_with_internal_gaps'])}")
