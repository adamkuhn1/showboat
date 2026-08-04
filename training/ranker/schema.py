"""Reads the ONE canonical ranker schema (apps/showboat/src/ai/ranker/schema.json).

Deliberately not a second encoder implementation — see
docs/repair/showboat-ml/ARCHITECTURE_DECISION.md. Python never re-derives
features from raw board/candidate state; it only ever consumes already-encoded
rows written by the Node/tsx dataset generator (gen_dataset.ts), which shares
the literal same encoder function used at browser inference time. This module
exists so Python knows the input dimension and schema version without a third
copy of the number 67 hardcoded here.
"""

from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "src" / "ai" / "ranker" / "schema.json"


def load_schema() -> dict:
    with open(_SCHEMA_PATH, "r") as f:
        return json.load(f)


_schema = load_schema()

SCHEMA_VERSION: str = _schema["schema_version"]
BOARD_DIM: int = _schema["board_dim"]
CANDIDATE_DIM: int = _schema["candidate_dim"]
TOTAL_DIM: int = _schema["total_dim"]

assert BOARD_DIM + CANDIDATE_DIM == TOTAL_DIM, (
    f"schema.json is internally inconsistent: board_dim({BOARD_DIM}) + "
    f"candidate_dim({CANDIDATE_DIM}) != total_dim({TOTAL_DIM})"
)
