from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)