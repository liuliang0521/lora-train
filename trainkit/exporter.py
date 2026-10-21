"""Inspect checkpoints for corruption and export deployment packages."""
import hashlib
import json
import re
from pathlib import Path
import shutil

from .data import ROOT, style_config


def inspect_lora(path, backend):
    import torch
    from safetensors import safe_open
    path = Path(path)
    count, pairs = 0, {}
    with safe_open(str(path), framework="pt", device="cpu") as file:
        metadata = file.metadata() or {}
        keys = list(file.keys())
        for key in keys:
            if key.endswith((".lora_up.weight", ".lora_B.weight", ".lora_B.default.weight")):
                counterpart = key.replace('.lora_up.weight', '.lora_down.weight').replace('.lora_B.', '.lora_A.')
                if counterpart not in keys:
                    raise ValueError('LoRA pair missing: ' + counterpart)
            tensor = file.get_tensor(key)
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("NaN/Inf in LoRA: " + key)
            if key.endswith((".lora_down.weight", ".lora_A.weight", ".lora_A.default.weight")):
                counterpart = key.replace('.lora_down.weight', '.lora_up.weight').replace('.lora_A.', '.lora_B.')
                if counterpart not in keys:
                    raise ValueError('LoRA pair missing: ' + counterpart)
                up = file.get_tensor(counterpart)
                if tensor.ndim < 2 or up.ndim < 2 or tensor.shape[0] == 0 or tensor.shape[0] != up.shape[1]:
                    raise ValueError("LoRA rank/shape mismatch: " + key)
                count += 1
                stem = key.rsplit(".", 2)[0] if ".lora_A." in key else key[:-len(".lora_down.weight")]
                pairs[stem] = True
        if not count:
            raise ValueError("Not a supported LoRA checkpoint")
        joined = "\n".join(keys)
        downs = [name for name in keys if name.endswith((".lora_down.weight", ".lora_A.weight", ".lora_A.default.weight"))]
        qwen_layer = re.compile(r"^(?:lora_unet_)?transformer_blocks[._]\d+[._]")
        if backend == "sdxl" and "lora_unet_" not in joined:
            raise ValueError("Expected Kohya SDXL LoRA keys")
        if backend == "sdxl" and not all(name.startswith("lora_unet_") and not qwen_layer.match(name) for name in downs):
            raise ValueError("Expected UNet-only SDXL LoRA")
        if backend == "qwen" and not all(qwen_layer.match(name) for name in downs):
            raise ValueError("Expected Qwen transformer LoRA keys")
    return {"lora_layers": count, "metadata": metadata}


def export(style, backend, checkpoint, manifest, root=ROOT):
    root, checkpoint = Path(root), Path(checkpoint).resolve()
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data["style"], data["backend"]) != (style, backend):
        raise ValueError("Dataset manifest belongs to another style/backend")
    # Prevent accidental export of a checkpoint from another run or smoke test.
    output = (manifest.parent / "output").resolve()
    if output not in checkpoint.parents or not checkpoint.is_file():
        raise ValueError("Select a checkpoint under this run's output/ (not smoke/ or pilot/)")
    info = inspect_lora(checkpoint, backend)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    selected = style_config(style, root)
    filename = f"{style}-{backend}-{sha256[:12]}.safetensors"
    destination = root / "exports" / backend / style / sha256[:12]
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, destination / filename)
    package = {"schema": 1, "style": style, "backend": backend, "trigger": selected["trigger"],
               "filename": filename, "sha256": sha256, "bytes": checkpoint.stat().st_size,
               "comfy_lora_path": f"styles/{style}-{backend}.safetensors",
               "model_version": "edit-2511" if backend == "qwen" else "sdxl-1.0",
               "training_recipe": "paired-edit-non-distilled" if backend == "qwen" else "unet-style",
               "base_model": "sd_xl_base_1.0.safetensors" if backend == "sdxl" else "qwen_image_edit_2511_bf16.safetensors",
               "dataset_fingerprint": data["fingerprint"], "inspection": info}
    (destination / "manifest.json").write_text(json.dumps(package, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("dataset-manifest.json", "environment.json", "environment-freeze.txt", "commands.json"):
        source = manifest.parent / name
        if source.is_file():
            shutil.copyfile(source, destination / name)
    return destination
