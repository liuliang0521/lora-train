import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from trainkit.data import ROOT
from trainkit.exporter import export, inspect_lora


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "config", self.root / "config")
        self.run = self.root / "runs/sdxl/cartoon/test"
        self.output = self.run / "output"
        self.output.mkdir(parents=True)
        self.checkpoint = self.output / "checkpoint.safetensors"
        self.manifest = self.run / "dataset-manifest.json"
        self.manifest.write_text(json.dumps({"style": "cartoon", "backend": "sdxl", "fingerprint": "test"}))
        self.weights = {"lora_unet_input_blocks_0.lora_down.weight": torch.ones(2, 4),
                        "lora_unet_input_blocks_0.lora_up.weight": torch.ones(4, 2)}

    def test_export_is_hashed_and_records_base_and_trigger(self):
        save_file(self.weights, str(self.checkpoint))
        directory = export("cartoon", "sdxl", self.checkpoint, self.manifest, self.root)
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["trigger"], "iliang_chibi")
        self.assertEqual(manifest["base_model"], "sd_xl_base_1.0.safetensors")
        self.assertEqual(hashlib.sha256((directory / manifest["filename"]).read_bytes()).hexdigest(),
                         manifest["sha256"])

    def test_nan_checkpoint_is_rejected(self):
        self.weights["lora_unet_input_blocks_0.lora_down.weight"][0, 0] = float("nan")
        save_file(self.weights, str(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            inspect_lora(self.checkpoint, "sdxl")

    def test_unpaired_lora_is_rejected(self):
        self.weights.pop("lora_unet_input_blocks_0.lora_up.weight")
        save_file(self.weights, str(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "pair missing"):
            inspect_lora(self.checkpoint, "sdxl")

    def test_smoke_checkpoint_cannot_be_exported(self):
        smoke = self.run / "smoke"
        smoke.mkdir()
        checkpoint = smoke / "model.safetensors"
        save_file(self.weights, str(checkpoint))
        with self.assertRaisesRegex(ValueError, "not smoke"):
            export("cartoon", "sdxl", checkpoint, self.manifest, self.root)

    def test_architecture_mismatch_is_rejected(self):
        save_file(self.weights, str(self.checkpoint))
        with self.assertRaisesRegex(ValueError, "Qwen transformer"):
            inspect_lora(self.checkpoint, "qwen")

    def test_sdxl_attention_transformer_layers_are_valid(self):
        prefix = "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q"
        save_file({prefix + ".lora_down.weight": torch.ones(2, 4),
                   prefix + ".lora_up.weight": torch.ones(4, 2)}, str(self.checkpoint))
        self.assertEqual(inspect_lora(self.checkpoint, "sdxl")["lora_layers"], 1)


if __name__ == "__main__":
    unittest.main()
