"""Validate prepared commands/TOML with the actual pinned Musubi parsers; no models loaded."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(run, mode):
    import toml
    from musubi_tuner.dataset.config_utils import ConfigSanitizer
    run = Path(run).resolve()
    source = ROOT / "vendor/musubi-tuner/src/musubi_tuner"
    plan = json.loads((run / (mode + "-commands.json")).read_text(encoding="utf-8"))
    ConfigSanitizer().sanitize_user_config(toml.loads((run / "dataset.toml").read_text(encoding="utf-8")))
    for stage in plan:
        args = stage["args"]
        index = next(i for i, item in enumerate(args) if Path(item).name.startswith("qwen_image_") and item.endswith(".py"))
        script = source / Path(args[index]).name
        spec = importlib.util.spec_from_file_location("checked_" + script.stem, script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if script.stem == "qwen_image_cache_latents":
            parser = module.cache_latents.hv_setup_parser(module.cache_latents.setup_parser_common())
        elif script.stem == "qwen_image_cache_text_encoder_outputs":
            parser = module.cache_text_encoder_outputs.setup_parser_common()
        else:
            parser = module.setup_parser_common()
        parser = module.qwen_image_setup_parser(parser)
        parser.parse_args(args[index + 1:])
    print("Pinned Musubi dataset schema and all prepared stage arguments passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--mode", choices=["cache", "smoke", "pilot", "train"], default="train")
    options = parser.parse_args()
    check(options.run, options.mode)
