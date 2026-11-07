from __future__ import annotations

import pathlib

# Locate the root directory for rag3evaluationsuite or fallback to parent
SUITE_ROOT = pathlib.Path(__file__).resolve().parent.parent
PARENT_ROOT = SUITE_ROOT.parent

# Resolve data directories (prioritizing local data if present)
if (SUITE_ROOT / "data" / "corpus").exists():
    DATA_DIR = SUITE_ROOT / "data"
elif (PARENT_ROOT / "data" / "corpus").exists():
    DATA_DIR = PARENT_ROOT / "data"
else:
    DATA_DIR = SUITE_ROOT / "data"

CORPUS_DIR = DATA_DIR / "corpus"
GOLD_PATH = DATA_DIR / "gold" / "questions.json"
if not GOLD_PATH.exists() and (SUITE_ROOT / "evaluation" / "questions.json").exists():
    GOLD_PATH = SUITE_ROOT / "evaluation" / "questions.json"
