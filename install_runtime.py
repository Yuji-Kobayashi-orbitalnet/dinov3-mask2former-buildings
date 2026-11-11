"""Install the pinned runtime inside the explicitly named conda environment."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Activate dino-whu-m2f (Python 3.11) first.")
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix or Path(prefix).name != "dino-whu-m2f":
        raise SystemExit("Run: conda activate dino-whu-m2f")
    root = Path(__file__).resolve().parent
    pip = [sys.executable, "-m", "pip"]
    subprocess.run(pip + ["install", "torch==2.9.1", "torchvision==0.24.1",
                         "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run(pip + ["install", "-r", str(root / "requirements.txt")], check=True)
    subprocess.run(pip + ["check"], check=True)
    # Record the actual transitive versions; requirements.txt alone is not a lockfile.
    report = root / "reports"
    report.mkdir(exist_ok=True)
    frozen = subprocess.check_output(pip + ["freeze"], text=True)
    (report / "installed-freeze.txt").write_text(frozen, encoding="utf-8")
    (report / "python.json").write_text(json.dumps({"executable": sys.executable,
        "version": sys.version, "platform": sys.platform}, indent=2), encoding="utf-8")
    print("Dependencies installed. Next: python verify_env.py --require-blackwell")


if __name__ == "__main__":
    main()
