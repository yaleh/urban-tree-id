import sys
from pathlib import Path

# Add scripts/ root so _path_setup can be imported
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Add all subdirectories so tests can import from any subpackage without prefix
from _path_setup import setup
setup()
