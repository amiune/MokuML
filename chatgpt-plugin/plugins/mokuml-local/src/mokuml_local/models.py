"""Model acquisition is an explicit, data-free setup operation; inference is offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

from .errors import SafeError

CLASSIFIER = "tabicl-classifier-v2-20260212.ckpt"
REGRESSOR = "tabicl-regressor-v2-20260212.ckpt"


def state_directory() -> Path:
    default = Path.home() / "Library" / "Application Support" / "MokuML Local"
    path = Path(os.environ.get("MOKUML_HOME", default)).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def model_directory() -> Path:
    path = state_directory() / "models"
    path.mkdir(exist_ok=True, mode=0o700)
    return path


def manifest() -> dict:
    return json.loads(Path(__file__).with_name("model-manifest.json").read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def available(component: str, directory: Path | None = None) -> bool:
    if component not in manifest():
        return False
    directory = (directory or model_directory()) / component
    return all((directory / name).is_file() for name in manifest()[component]["files"])


def verify(component: str, directory: Path | None = None) -> Path:
    directory = (directory or model_directory()) / component
    for name, expected in manifest()[component]["files"].items():
        path = directory / name
        if not path.is_file():
            raise SafeError("models_missing")
        if digest(path) != expected:
            raise SafeError("models_invalid")
    return directory


def prepare(
    component: str, directory: Path | None = None, reuse: Path | None = None
) -> None:
    spec = manifest()[component]
    directory = (directory or model_directory()) / component
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, expected in spec["files"].items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.is_file() and digest(destination) == expected:
            continue
        fd, temporary_name = tempfile.mkstemp(prefix="download-", dir=directory)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            candidate = reuse / name if reuse else None
            if candidate and candidate.is_file() and digest(candidate) == expected:
                shutil.copyfile(candidate, temporary)
            else:
                asset = spec.get("sources", {}).get(name, {})
                repository = asset.get("repository", spec["repository"])
                revision = asset.get("revision", spec["revision"])
                filename = asset.get("filename", name)
                url = (
                    f"https://huggingface.co/{repository}/resolve/{revision}/{filename}"
                )
                headers = {}
                if component == "tabpfn35" and os.environ.get("MOKUML_SETUP_TOKEN"):
                    headers["Authorization"] = (
                        "Bearer " + os.environ["MOKUML_SETUP_TOKEN"]
                    )
                with (
                    urlopen(Request(url, headers=headers), timeout=120) as source,
                    temporary.open("wb") as target,
                ):
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            if digest(temporary) != expected:
                raise SafeError("models_invalid")
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def active_tabpfn_component():
    try:
        config = json.loads((state_directory() / "tabpfn-settings.json").read_text())
        if config == {"component": "tabpfn35", "licensed_opt_in": True}:
            return "tabpfn35"
    except (OSError, ValueError):
        pass
    return "tabpfn"


def load_encoder(device: str):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    directory = verify("qwen")
    config = AutoConfig.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    config.num_hidden_layers = manifest()["qwen"]["layers"]
    if getattr(config, "layer_types", None) is not None:
        config.layer_types = config.layer_types[: config.num_hidden_layers]
    # Float32 is portable on CPU and MPS. Only the first sixteen blocks are loaded.
    model = AutoModel.from_pretrained(
        directory,
        config=config,
        torch_dtype=torch.float32,
        local_files_only=True,
        trust_remote_code=False,
    ).to(device)
    model.norm = torch.nn.Identity()  # TiLE uses the raw intermediate block output.
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify pinned local models; no input records are read."
    )
    parser.add_argument(
        "--text", action="store_true", help="Also prepare the Qwen text encoder"
    )
    parser.add_argument(
        "--reuse-tabicl", type=Path, help="Reuse verified existing TabICLv2 checkpoints"
    )
    parser.add_argument(
        "--component",
        choices=["all", "tabicl", "kumo", "tabpfn", "tabpfn35", "qwen"],
        default="all",
    )
    parser.add_argument("--licensed-opt-in", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.component == "tabpfn35" and not args.licensed_opt_in:
            raise SafeError("invalid_selection")
        components = (
            ["tabicl", "kumo", "tabpfn"]
            if args.component == "all"
            else [args.component]
        )
        for component in components:
            prepare(
                component, reuse=args.reuse_tabicl if component == "tabicl" else None
            )
        if args.component in {"tabpfn", "tabpfn35"}:
            config = {
                "component": args.component,
                "licensed_opt_in": args.component == "tabpfn35",
            }
            (state_directory() / "tabpfn-settings.json").write_text(json.dumps(config))
        if args.text:
            prepare("qwen")
        print("Local prediction models are verified and ready.", flush=True)
    except Exception:
        print(
            "Model setup failed. Check internet connectivity and disk space, then retry.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
