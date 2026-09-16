from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


SILERO_VERSION = "v6.2"
SPEECHBRAIN_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
ARTIFACTS = [
    {
        "path": "silero_vad.onnx",
        "url": (
            "https://raw.githubusercontent.com/snakers4/silero-vad/"
            f"{SILERO_VERSION}/src/silero_vad/data/silero_vad.onnx"
        ),
        "sha256": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
    },
    {
        "path": "speechbrain-spkrec-ecapa-voxceleb/embedding_model.ckpt",
        "url": (
            "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/"
            f"{SPEECHBRAIN_REVISION}/embedding_model.ckpt"
        ),
        "sha256": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    },
    {
        "path": "speechbrain-spkrec-ecapa-voxceleb/classifier.ckpt",
        "url": (
            "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/"
            f"{SPEECHBRAIN_REVISION}/classifier.ckpt"
        ),
        "sha256": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    },
    {
        "path": "speechbrain-spkrec-ecapa-voxceleb/hyperparams.yaml",
        "url": (
            "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/"
            f"{SPEECHBRAIN_REVISION}/hyperparams.yaml"
        ),
        "sha256": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    },
    {
        "path": "speechbrain-spkrec-ecapa-voxceleb/mean_var_norm_emb.ckpt",
        "url": (
            "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/"
            f"{SPEECHBRAIN_REVISION}/mean_var_norm_emb.ckpt"
        ),
        "sha256": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    },
    {
        "path": "speechbrain-spkrec-ecapa-voxceleb/label_encoder.txt",
        "url": (
            "https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/"
            f"{SPEECHBRAIN_REVISION}/label_encoder.txt"
        ),
        "sha256": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned ProctorCore audio-analysis models.")
    parser.add_argument("--model-root", type=Path, default=Path("models/audio"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.model_root.mkdir(parents=True, exist_ok=True)

    results = []
    for artifact in ARTIFACTS:
        destination = args.model_root / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not args.verify_only and (not destination.is_file() or sha256(destination) != artifact["sha256"]):
            temporary = destination.with_suffix(destination.suffix + ".download")
            with urllib.request.urlopen(artifact["url"], timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if sha256(temporary) != artifact["sha256"]:
                temporary.unlink(missing_ok=True)
                raise SystemExit(f"Checksum mismatch: {artifact['path']}")
            temporary.replace(destination)
        actual = sha256(destination) if destination.is_file() else ""
        results.append({
            "path": str(destination),
            "ready": actual == artifact["sha256"],
            "sha256": actual,
        })

    print(json.dumps({"ok": all(item["ready"] for item in results), "artifacts": results}, indent=2))
    if not all(item["ready"] for item in results):
        raise SystemExit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
