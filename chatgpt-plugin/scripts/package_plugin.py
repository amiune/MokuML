"""Build a portable, cache-free local marketplace ZIP."""

import json
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root / "dist"
output.mkdir(exist_ok=True)
plugin = root / "plugins/mokuml-local"
version = json.loads((plugin / "plugin.json").read_text())["version"]
destination = output / f"mokuml-local-{version}.zip"
excluded = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".local",
    "build",
    "dist",
}
with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    paths = [path for name in ["plugins", "scripts", ".agents"] for path in (root/name).rglob("*")] + [root/"README.md", root/"VERIFICATION.md"]
    for path in sorted(paths):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file() or any(
            p in excluded or p.endswith(".egg-info") for p in relative.parts
        ):
            continue
        if relative.parts[0] not in {"plugins", "scripts", ".agents"} and str(
            relative
        ) not in {"README.md", "VERIFICATION.md"}:
            continue
        if path.suffix in {".pyc", ".ckpt", ".safetensors", ".pickle", ".pt"}:
            continue
        bundle.write(path, Path("mokuml-local-marketplace") / relative)
print(destination)
