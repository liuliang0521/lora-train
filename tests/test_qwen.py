import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from trainkit.data import ROOT, initialize, prepare
from trainkit.exporter import export, inspect_lora
from trainkit.runner import commands
import test_training


class QwenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "config", self.root / "config")
        initialize(self.root)
        test_training.DataTests.dataset(self, paired=True)

    def test_paired_caption_and_independent_condition_resolution_invalidate_cache(self):
        import toml
        first = prepare("cartoon", "qwen", self.root)
        data = toml.loads((first / "dataset.toml").read_text())
        self.assertEqual(data["datasets"][0]["control_resolution"], [512, 512])
        caption = next((first / "train").glob("*.txt")).read_text()
        self.assertIn("Restyle the entire input in iliang_chibi style", caption)
        self.assertIn("do not remove, replace or invent scenery", caption)
        profile = self.root / "config/profiles.json"
        value = json.loads(profile.read_text())
        value["qwen"]["control_resolution"] = 1024
        profile.write_text(json.dumps(value))
        second = prepare("cartoon", "qwen", self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(toml.loads((second / "dataset.toml").read_text())["datasets"][0]["control_resolution"], [1024, 1024])

    def test_pilot_cache_and_formal_training_are_separate(self):
        run = prepare("cartoon", "qwen", self.root)
        pilot = commands("cartoon", "qwen", run, self.root, pilot=True)
        training = pilot[-1][1]
        self.assertEqual(training[training.index("--max_train_steps") + 1], "100")
        self.assertEqual(training[training.index("--output_dir") + 1], str(run / "pilot"))
        for flag in ("--gradient_checkpointing", "--gradient_checkpointing_cpu_offload", "--block_swap_h2d_only", "--fp8_base", "--fp8_scaled"):
            self.assertIn(flag, training)
        self.assertNotIn("--use_pinned_memory_for_block_swap", training)
        cache = commands("cartoon", "qwen", run, self.root, cache_only=True)
        self.assertEqual(len(cache), 2)
        for _, args in cache:
            self.assertIn("--skip_existing", args)
            self.assertEqual(args[args.index("--num_workers") + 1], "1")
        self.assertNotIn("train_network.py", " ".join(cache[-1][1]))

    def test_small_source_image_is_not_accepted_as_a_training_pair(self):
        from PIL import Image
        from trainkit.data import validate
        source = self.root / "datasets/cartoon/sources/people/000.png"
        Image.new("RGB", (80, 64)).save(source)
        report = validate("cartoon", "qwen", self.root)
        self.assertFalse(report["valid"])
        self.assertTrue(any("source minimum side" in error for error in report["errors"]))

    def test_musubi_qwen_export_and_sdxl_rejection(self):
        run = prepare("cartoon", "qwen", self.root)
        output = run / "output"
        output.mkdir()
        checkpoint = output / "cartoon-qwen.safetensors"
        prefix = "lora_unet_transformer_blocks_0_attn_to_q"
        weights = {prefix + ".lora_down.weight": torch.ones(2, 4),
                   prefix + ".lora_up.weight": torch.ones(4, 2), prefix + ".alpha": torch.tensor(2.)}
        save_file(weights, str(checkpoint))
        self.assertEqual(inspect_lora(checkpoint, "qwen")["lora_layers"], 1)
        with self.assertRaisesRegex(ValueError, "SDXL"):
            inspect_lora(checkpoint, "sdxl")
        directory = export("cartoon", "qwen", checkpoint, run / "dataset-manifest.json", self.root)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["comfy_lora_path"], "styles/cartoon-qwen.safetensors")
        self.assertEqual(manifest["training_recipe"], "paired-edit-non-distilled")
        pilot = run / "pilot"
        pilot.mkdir()
        shutil.copyfile(checkpoint, pilot / checkpoint.name)
        with self.assertRaisesRegex(ValueError, "pilot"):
            export("cartoon", "qwen", pilot / checkpoint.name, run / "dataset-manifest.json", self.root)
        weights[prefix + ".lora_up.weight"] = torch.ones(4, 3)
        save_file(weights, str(checkpoint))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            inspect_lora(checkpoint, "qwen")
        weights[prefix + ".lora_up.weight"] = torch.ones(4, 2)
        weights["lora_unet_transformer_blocks_1_attn_to_q.lora_up.weight"] = torch.ones(4, 2)
        save_file(weights, str(checkpoint))
        with self.assertRaisesRegex(ValueError, "pair missing"):
            inspect_lora(checkpoint, "qwen")
