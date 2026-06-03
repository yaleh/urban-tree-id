"""Add all scripts subdirectories to sys.path.

Import and call setup() at the top of any script that needs to import from
sibling subdirectories (detection/, eval/, training/, utils/).
"""
import sys
from pathlib import Path


def setup() -> None:
    root = Path(__file__).resolve().parent
    for subdir in [root, root / "detection", root / "eval", root / "training", root / "utils"]:
        s = str(subdir)
        if s not in sys.path:
            sys.path.insert(0, s)
