"""
Shared helpers for the eval scripts.
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable and the working directory, so the
# relative paths in agent/tools.py ("data/products.db", "chroma_db") resolve
# no matter where the eval is launched from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CASES_PATH = ROOT / "evals" / "cases.json"
RESULTS_DIR = ROOT / "evals" / "results"


def load_cases(only=None):
    cases = json.loads(CASES_PATH.read_text())
    if only:
        wanted = set(only)
        unknown = wanted - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["id"] in wanted]
    return cases


def _patterns(words):
    # \b at the start only, so "Buds" matches "Buds3" but "Mic" doesn't match "Dynamic".
    return [re.compile(r"\b" + re.escape(w), re.IGNORECASE) for w in words or []]


def _clean(name):
    # Some catalogue names contain non-breaking spaces ("AirPods\xa0Pro\xa03").
    return re.sub(r"\s+", " ", name)


def is_relevant(name, case):
    name = _clean(name)
    return any(p.search(name) for p in _patterns(case.get("relevant")))


def is_excluded(name, case):
    # Plain substring match: "headset" should catch "Headsets" and "Headset)".
    name = _clean(name)
    return any(re.search(re.escape(w), name, re.IGNORECASE) for w in case.get("exclude_keywords", []))


def save_results(kind, payload):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{kind}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def pct(x):
    return f"{x * 100:5.1f}%"
