"""Project command line: data, training, model downloads and export."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .data import ROOT, config, initialize, prepare, validate
from .runner import run_training


def doctor():
    import platform
    import torch
    info = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        info.update(gpu=torch.cuda.get_device_name(0),
                    vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                    bf16=torch.cuda.is_bf16_supported())
        x = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        info["gpu_matmul_passed"] = (x @ x)[0, 0].item() == 16
    print(json.dumps(info, indent=2))
    if not info.get("gpu_matmul_passed"):
        raise ValueError("CUDA BF16 test failed")
    return info


def download(backend, model_id=None):
    from huggingface_hub import hf_hub_download, HfApi
    selected = [m for m in config("models")["models"] if
                (m["id"] == model_id if model_id else m["backend"] == backend)]
    if not selected:
        raise ValueError("Unknown model id")
    for item in selected:
        target = ROOT / item["path"]
        if target.exists():
            raise ValueError("Target exists; verify/reuse it or move it first: " + str(target))
        # Resolve an immutable repo revision and let HF verify/cache the download.
        info = HfApi().model_info(item["repo"], files_metadata=True)
        revision = info.sha
        entry = next(file for file in info.siblings if file.rfilename == item['file'])
        source = Path(hf_hub_download(item["repo"], item["file"], revision=revision))
        import shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        receipt = dict(item, revision=revision, sha256=digest.hexdigest(), bytes=target.stat().st_size)
        expected = getattr(entry.lfs, 'sha256', None) if entry.lfs else None
        if expected and receipt['sha256'] != expected:
            raise ValueError('Downloaded model SHA256 differs from source LFS metadata: ' + item['id'])
        target.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Six-style LoRA training on RTX 4060 Ti 16GB")
    parser.add_argument("command", choices=["init", "doctor", "download", "validate", "prepare", "train", "export"])
    parser.add_argument("--backend", choices=["sdxl", "qwen"], default="sdxl")
    parser.add_argument("--style", choices=[s["id"] for s in config("styles")["styles"]])
    parser.add_argument("--allow-small", action="store_true", help="Allow a smaller pilot dataset, at least 4 valid images")
    parser.add_argument("--smoke", action="store_true", help="Two optimizer steps; not a trained style model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", help="Upstream saved state directory")
    parser.add_argument("--checkpoint", help="Explicit selected .safetensors file under a run's output/")
    parser.add_argument("--manifest", help="Matching run's dataset-manifest.json")
    parser.add_argument("--model", help="One id from config/models.json")
    args = parser.parse_args()
    try:
        if args.command == "init":
            initialize()
            print("Dataset folders created. Add target images and sidecar captions; see docs/DATASET.md")
        elif args.command == "doctor":
            doctor()
        elif args.command == "download":
            download(args.backend, args.model)
        else:
            if not args.style:
                parser.error("--style is required")
            if args.command == "validate":
                report = validate(args.style, args.backend, allow_small=args.allow_small)
                print(json.dumps(report, indent=2, ensure_ascii=False))
                if not report["valid"]:
                    raise ValueError("Dataset validation failed")
            elif args.command == "prepare":
                print(prepare(args.style, args.backend, allow_small=args.allow_small))
            elif args.command == "train":
                run_training(args.style, args.backend, smoke=args.smoke, resume=args.resume,
                             dry_run=args.dry_run, allow_small=args.allow_small)
            else:
                if not args.checkpoint or not args.manifest:
                    parser.error("export requires --checkpoint and --manifest")
                from .exporter import export
                print(export(args.style, args.backend, args.checkpoint, args.manifest))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
