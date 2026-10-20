import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from trainkit.data import ROOT, initialize, prepare, validate
from trainkit.runner import commands


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "config", self.root / "config")
        initialize(self.root)

    def dataset(self, paired=False):
        number = 0
        for category, count in {"people": 20, "animals": 15, "scenes": 15, "fantasy": 10}.items():
            for index in range(count):
                target = self.root / "datasets/cartoon" / category / f"{index:03}.png"
                Image.new("RGB", (320, 256), (number, 100, 180)).save(target)
                target.with_suffix(".txt").write_text("iliang_chibi, subject in a garden", encoding="utf-8")
                if paired:
                    source = self.root / "datasets/cartoon/sources" / category / target.name
                    Image.new("RGB", (320, 256), (number, 80, 120)).save(source)
                number += 1

    def test_full_dataset_is_split_and_raw_images_are_preserved(self):
        self.dataset()
        run = prepare("cartoon", root=self.root)
        manifest = json.loads((run / "dataset-manifest.json").read_text())
        self.assertEqual((len(manifest["train"]), len(manifest["validation"])), (53, 7))
        self.assertFalse(set(x["sha256"] for x in manifest["train"]) & set(x["sha256"] for x in manifest["validation"]))
        self.assertEqual(len(list((run / "train").glob("*.png"))), 53)
        self.assertEqual(run, prepare("cartoon", root=self.root))
        self.assertEqual(len(list((self.root / "datasets/cartoon").glob("*/*.png"))), 60)

    def test_caption_change_produces_new_cache_fingerprint(self):
        self.dataset()
        first = prepare("cartoon", root=self.root)
        (self.root / "datasets/cartoon/people/000.txt").write_text("iliang_chibi, a different caption")
        self.assertNotEqual(first, prepare("cartoon", root=self.root))

    def test_bad_images_missing_captions_and_duplicates_block_training(self):
        self.dataset()
        (self.root / "datasets/cartoon/people/000.txt").unlink()
        shutil.copyfile(self.root / "datasets/cartoon/animals/000.png",
                        self.root / "datasets/cartoon/animals/001.png")
        (self.root / "datasets/cartoon/scenes/000.png").write_bytes(b"invalid")
        report = validate("cartoon", root=self.root)
        self.assertFalse(report["valid"])
        self.assertGreaterEqual(len(report["errors"]), 3)
        with self.assertRaises(ValueError):
            prepare("cartoon", root=self.root)

    def test_qwen_requires_real_source_target_pairs(self):
        self.dataset()
        self.assertFalse(validate("cartoon", "qwen", self.root)["valid"])

    def test_qwen_caches_have_distinct_source_and_target_paths(self):
        self.dataset(paired=True)
        run = prepare("cartoon", "qwen", self.root)
        import toml
        dataset = toml.loads((run / "dataset.toml").read_text())["datasets"][0]
        self.assertNotEqual(dataset["image_directory"], dataset["control_directory"])
        self.assertEqual(len(list((run / "train-sources").glob("*.png"))), 53)
        shutil.copyfile(self.root / "datasets/cartoon/people/000.png",
                        self.root / "datasets/cartoon/sources/people/000.png")
        self.assertFalse(validate("cartoon", "qwen", self.root)["valid"])

    def test_sdxl_train_uses_selected_rank_cache_and_bf16(self):
        self.dataset()
        run = prepare("cartoon", root=self.root)
        plan = commands("cartoon", "sdxl", run, self.root, smoke=True)
        self.assertEqual(len(plan), 1)
        args = plan[0][1]
        self.assertEqual(args[args.index("--max_train_steps")+1], "2")
        self.assertIn("--network_train_unet_only", args)
        self.assertIn("--cache_text_encoder_outputs_to_disk", args)
        self.assertNotIn("--fp8_base", args)
        self.assertIn(str(run / "smoke"), args)

    def test_sdxl_dataset_passes_the_installed_upstream_schema(self):
        try:
            from library.config_util import ConfigSanitizer
        except ImportError:
            self.skipTest('Requires the installed SDXL training environment')
        self.dataset()
        run = prepare('cartoon', root=self.root)
        import toml
        data = toml.loads((run / 'dataset.toml').read_text())
        ConfigSanitizer(True, True, False, True).sanitize_user_config(data)

    def test_qwen_training_is_explicit_2511_and_has_no_lightning(self):
        self.dataset(paired=True)
        run = prepare("cartoon", "qwen", self.root)
        plan = commands("cartoon", "qwen", run, self.root)
        self.assertEqual(len(plan), 3)
        args = plan[-1][1]
        self.assertIn("edit-2511", args)
        self.assertIn("--blocks_to_swap", args)
        self.assertIn("--fp8_scaled", args)
        self.assertNotIn("Lightning", " ".join(args))


if __name__ == "__main__":
    unittest.main()
