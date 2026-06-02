import sys
from pathlib import Path

# Make scripts/ importable without installing the package
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
