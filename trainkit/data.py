"""Validate captions/images and create reproducible, leakage-free train splits."""
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageOps

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ROOT = Path(__file__).resolve().parents[1]


def config(name, root=ROOT):
    return json.loads((Path(root) / "config" / (name + ".json")).read_text(encoding="utf-8"))


def style_config(style, root=ROOT):
    return next((item for item in config("styles", root)["styles"] if item["id"] == style), None) or fail("Unknown style: " + style)


def fail(message):
    raise ValueError(message)


def images(directory):
    return sorted(p for p in directory.glob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())


def initialize(root=ROOT):
    root = Path(root)
    for style in config("styles", root)["styles"]:
        for category in config("styles", root)["categories"]:
            (root / "datasets" / style["id"] / category).mkdir(parents=True, exist_ok=True)
            (root / "datasets" / style["id"] / "sources" / category).mkdir(parents=True, exist_ok=True)
    for name in ("models", "runs", "exports", "runtime"):
        (root / name).mkdir(exist_ok=True)


def validate(style, backend="sdxl", root=ROOT, allow_small=False):
    root = Path(root)
    selected = style_config(style, root)
    dataset = root / "datasets" / style
    errors, records, hashes, ids = [], [], {}, set()
    expected = config("styles", root)["categories"]
    for category, count in expected.items():
        paths = images(dataset / category)
        if not allow_small and len(paths) != count:
            errors.append(f"{category}: expected {count} images, found {len(paths)}")
        for path in paths:
            caption = path.with_suffix(".txt")
            identifier = category + "/" + path.stem
            if identifier in ids:
                errors.append("Duplicate stem: " + identifier)
                continue
            ids.add(identifier)
            try:
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) != 1:
                        raise ValueError("animated images are unsupported")
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    width, height = rgb.size
                    if min(width, height) < 256 or max(width / height, height / width) > 3:
                        raise ValueError("minimum side must be >=256, aspect ratio <=3")
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                if digest in hashes:
                    raise ValueError("duplicate image of " + hashes[digest])
                hashes[digest] = identifier
                if not caption.is_file():
                    raise ValueError("missing sidecar .txt caption")
                text = caption.read_text(encoding="utf-8-sig").strip()
                if not text or selected["trigger"] not in text:
                    raise ValueError("caption must include trigger " + selected["trigger"])
                if any(token in text.lower() for token in ("todo", "<describe", "replace me")):
                    raise ValueError("unfinished caption template")
                source = None
                source_digest = None
                if backend == "qwen":
                    candidates = [p for p in images(dataset / "sources" / category) if p.stem == path.stem]
                    if len(candidates) != 1:
                        raise ValueError("Qwen Edit requires exactly one aligned source photo with the same stem")
                    source = candidates[0]
                    with Image.open(source) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                        source_digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        if abs(rgb.width / rgb.height - width / height) > 0.03:
                            raise ValueError("source/target aspect ratios differ; align composition first")
                        if source_digest == digest:
                            raise ValueError("source equals target; this teaches reconstruction, not style editing")
                records.append(dict(id=identifier, category=category, image=str(path.resolve()),
                                    caption=text, sha256=digest, width=width, height=height,
                                    source=str(source.resolve()) if source else None, source_sha256=source_digest))
            except (OSError, ValueError) as error:
                errors.append(f"{path.name}: {error}")
    # Even tiny smoke datasets need distinct training and validation subjects.
    if len(records) < 4:
        errors.append("Need at least 4 valid images (smoke mode does not bypass content checks)")
    if backend == "qwen":
        seen_sources = {}
        for record in records:
            digest = record["source_sha256"]
            if digest in seen_sources:
                errors.append("Duplicate source subjects: " + record["id"] + " / " + seen_sources[digest])
            seen_sources[digest] = record["id"]
    return {"style": style, "backend": backend, "records": records, "errors": errors, "valid": not errors}


def split_records(records, seed=42):
    train, validation = [], []
    for category in sorted({item["category"] for item in records}):
        group = sorted((item for item in records if item["category"] == category),
                       key=lambda r: hashlib.sha256((str(seed) + r["sha256"]).encode()).hexdigest())
        holdout = max(1, math.ceil(len(group) * 0.1)) if len(group) >= 2 else 0
        validation.extend(group[:holdout])
        train.extend(group[holdout:])
    if not validation and len(train) > 1:
        validation.append(train.pop(0))
    return train, validation


def prepare(style, backend="sdxl", root=ROOT, allow_small=False):
    root = Path(root)
    report = validate(style, backend, root, allow_small)
    if report["errors"]:
        fail("\n".join(report["errors"]))
    profile = config("profiles", root)[backend]
    train, validation = split_records(report["records"], profile["seed"])
    # Content and caption hash prevents reuse of stale latent/text caches.
    fingerprint = hashlib.sha256(json.dumps(
        {"records": report["records"], "profile": profile, "style": style_config(style, root),
         "trainer_revision": config("backends", root)[backend]["revision"], "backend": backend},
        sort_keys=True).encode()).hexdigest()[:16]
    run = root / "runs" / backend / style / fingerprint
    for subset, records in (("train", train), ("validation", validation)):
        target = run / subset
        target.mkdir(parents=True, exist_ok=True)
        for item in records:
            name = item["id"].replace("/", "__")
            image = target / (name + ".png")
            if not image.exists():
                with Image.open(item["image"]) as picture:
                    ImageOps.exif_transpose(picture).convert("RGB").save(image)
            # Qwen caption is an editing instruction, SDXL caption describes target.
            caption = item["caption"]
            if backend == "qwen":
                caption = "Restyle the input image as " + caption + ". Preserve identity, composition and existing scene."
            image.with_suffix(".txt").write_text(caption, encoding="utf-8")
            if backend == "qwen":
                source_dir = run / (subset + "-sources")
                source_dir.mkdir(exist_ok=True)
                source_file = source_dir / image.name
                if not source_file.exists():
                    with Image.open(item["source"]) as picture:
                        ImageOps.exif_transpose(picture).convert("RGB").save(source_file)
    manifest = {"schema": 1, "fingerprint": fingerprint, "backend": backend, "style": style,
                "trigger": style_config(style, root)["trigger"], "profile": profile,
                "train": train, "validation": validation}
    (run / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    import toml
    if backend == "sdxl":
        dataset = {"general": {"caption_extension": ".txt", "shuffle_caption": False},
                   "datasets": [{"resolution": profile["resolution"], "batch_size": 1,
                                 "enable_bucket": True, "bucket_no_upscale": True,
                                 "min_bucket_reso": 256, "max_bucket_reso": 1536, "bucket_reso_steps": 64,
                                 "subsets": [{"image_dir": str((run / "train").resolve()), "num_repeats": 1}]}]}
    else:
        dataset = {"general": {"resolution": [profile["resolution"]] * 2, "caption_extension": ".txt",
                               "batch_size": 1, "enable_bucket": True, "bucket_no_upscale": True},
                   "datasets": [{"image_directory": str((run / "train").resolve()),
                                 "control_directory": str((run / "train-sources").resolve()),
                                 "control_resolution": [profile["resolution"]] * 2,
                                 "cache_directory": str((run / "cache").resolve())}]}
    (run / "dataset.toml").write_text(toml.dumps(dataset), encoding="utf-8")
    return run
