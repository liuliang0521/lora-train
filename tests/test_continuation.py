"""Continuation/archive checks use tiny CPU fixtures, never real training data."""
import ast
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from trainkit.backend_patches import (ensure_qwen_resume_support, ensure_qwen_fp8_support,
                                     ORIGINAL_NORM, FP8_NORM, ORIGINAL_DECODER, FP8_DECODER)
from trainkit.continuation import archive_formal_run, choose_continuation, checkpoint_step
from trainkit.data import ROOT
from trainkit.runner import run_training


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        shutil.copytree(ROOT / "config", self.root / "config")
        self.run = self.root / "runs/qwen/cartoon/fingerprint"
        self.output = self.run / "output"
        self.output.mkdir(parents=True)

    def checkpoint(self, name, step):
        path = self.output / (name + ".safetensors")
        save_file({"weight": torch.ones(2, 2)}, str(path), metadata={"ss_steps": str(step)})
        return path

    def state(self, step):
        name = f"cartoon-qwen-step{step:08d}"
        self.checkpoint(name, step)
        state = self.output / (name + "-state")
        state.mkdir()
        save_file({"weight": torch.ones(2, 2)}, str(state / "model.safetensors"))
        for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl"):
            torch.save({"fixture": torch.ones(2)}, state / name)
        return state

    def test_new_run_and_complete_run_skip(self):
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, None, 0))
        self.checkpoint("cartoon-qwen", 1000)
        with patch("trainkit.runner.prepare", return_value=self.run), \
                patch("trainkit.runner.commands") as commands:
            self.assertEqual(run_training("cartoon", "qwen", self.root), self.run)
            commands.assert_not_called()

    def test_latest_good_state_wins_even_if_older_state_was_touched(self):
        old, latest = self.state(100), self.state(300)
        os.utime(old, (2000000000, 2000000000))
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, latest, 300))
        broken = self.state(400)
        (broken / "optimizer.bin").write_bytes(b"interrupted save")
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, latest, 300))
        self.checkpoint("cartoon-qwen-step00000500", 500)  # no matching state
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, latest, 300))

    def test_partial_final_checkpoint_does_not_count_as_complete(self):
        state = self.state(900)
        final = self.checkpoint("cartoon-qwen", 1000)
        final.write_bytes(final.read_bytes()[:-1])
        self.assertIsNone(checkpoint_step(final))
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, state, 900))

    def test_no_valid_state_does_not_silently_overwrite_weights(self):
        self.checkpoint("cartoon-qwen-step00000100", 100)
        with self.assertRaisesRegex(ValueError, "no complete resumable state"):
            choose_continuation(self.run, "cartoon-qwen", 1000)

    def test_pilot_and_different_fingerprint_cannot_be_resumed(self):
        state = self.state(100)
        other = self.run.parent / "another-fingerprint"
        with self.assertRaisesRegex(ValueError, "this exact run"):
            choose_continuation(other, "cartoon-qwen", 1000, state)
        pilot = self.run / "pilot"
        pilot.mkdir()
        shutil.move(str(state), str(pilot / state.name))
        with self.assertRaisesRegex(ValueError, "this exact run"):
            choose_continuation(self.run, "cartoon-qwen", 1000, pilot / state.name)

    def test_auto_resume_plan_preserves_total_target_and_trial_separation(self):
        state = self.state(300)
        with patch("trainkit.runner.prepare", return_value=self.run):
            run_training("cartoon", "qwen", self.root, dry_run=True)
        plan = json.loads((self.run / "train-commands.json").read_text("utf-8"))
        args = plan[-1]["args"]
        self.assertEqual(args[args.index("--resume") + 1], str(state))
        self.assertEqual(args[args.index("--max_train_steps") + 1], "1000")
        self.assertTrue(all("--skip_existing" in stage["args"] for stage in plan[:-1]))
        self.assertFalse((self.run / "smoke-commands.json").exists())

    def test_real_runner_passes_recovered_step_to_training_subprocess(self):
        state = self.state(300)
        python = self.root / ".venv-qwen/Scripts/python.exe"
        python.parent.mkdir(parents=True)
        python.touch()
        profile = json.loads((self.root / "config/profiles.json").read_text())["qwen"]
        for key in ("dit", "vae", "text_encoder"):
            model = self.root / profile[key]
            model.parent.mkdir(parents=True, exist_ok=True)
            model.touch()
        revision = json.loads((self.root / "config/backends.json").read_text())["qwen"]["revision"]
        child = MagicMock()
        child.__enter__.return_value = child
        child.stdout = ["simulated CPU-only subprocess output\n"]
        child.wait.return_value = 0
        with patch("trainkit.runner.prepare", return_value=self.run), \
                patch("trainkit.backend_patches.ensure_qwen_resume_support"), \
                patch("trainkit.backend_patches.ensure_qwen_fp8_support"), \
                patch("trainkit.resources.ResourceMonitor"), \
                patch("trainkit.runner.subprocess.check_output", side_effect=[revision, '{"bf16": true}', "fixture"]), \
                patch("trainkit.runner.subprocess.run", return_value=SimpleNamespace(stdout="parser OK", check_returncode=lambda: None)), \
                patch("trainkit.runner.subprocess.Popen", return_value=child) as popen, \
                patch("builtins.print"):
            run_training("cartoon", "qwen", self.root)
        self.assertEqual(popen.call_count, 3)
        launch = popen.call_args
        self.assertEqual(launch.kwargs["env"]["TRAINKIT_RESUME_STEP"], "300")
        self.assertIn(str(state), launch.args[0])

    def test_restart_then_interruption_can_continue_without_restarting_again(self):
        self.state(300)
        archive_formal_run(self.run, self.root, "qwen", "cartoon")
        self.output.mkdir()
        state = self.state(100)
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, state, 100))

    def test_restart_archives_only_current_formal_artifacts(self):
        self.state(300)
        for name in ("logs", "cache", "pilot", "smoke", "train"):
            (self.run / name).mkdir()
            (self.run / name / "keep.txt").write_text(name)
        for name in ("train.log", "commands.json", "train-resources.json", "dataset-manifest.json"):
            (self.run / name).write_text(name)
        other = self.run.parent / "another-fingerprint/output"
        other.mkdir(parents=True)
        archived = archive_formal_run(self.run, self.root, "qwen", "cartoon")
        self.assertTrue((archived / "output/cartoon-qwen-step00000300-state/optimizer.bin").is_file())
        self.assertTrue((archived / "train.log").is_file())
        self.assertFalse(self.output.exists())
        for name in ("cache", "pilot", "smoke", "train"):
            self.assertTrue((self.run / name / "keep.txt").is_file())
        self.assertTrue((self.run / "dataset-manifest.json").is_file())
        self.assertTrue(other.is_dir())
        self.assertEqual(choose_continuation(self.run, "cartoon-qwen", 1000), (False, None, 0))

    def test_restart_scope_and_conflicting_flags_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            archive_formal_run(self.run, self.root, "qwen", "jojo")
        for flags in ({"dry_run": True}, {"resume": "state"}, {"smoke": True}, {"pilot": True}):
            with self.assertRaisesRegex(ValueError, "Restart requires"):
                run_training("cartoon", "qwen", self.root, restart=True, **flags)

    def test_actual_musubi_progress_patch_is_idempotent_and_restores_step_and_epoch(self):
        vendor = self.root / "vendor/musubi-tuner"
        target = vendor / "src/musubi_tuner/training/trainer_base.py"
        target.parent.mkdir(parents=True)
        source = ROOT / "vendor/musubi-tuner/src/musubi_tuner/training/trainer_base.py"
        shutil.copyfile(source, target)
        ensure_qwen_resume_support(vendor)
        once = target.read_text("utf-8")
        ensure_qwen_resume_support(vendor)
        self.assertEqual(once, target.read_text("utf-8"))
        tree = ast.parse(once)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_run_training_loop")
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == "global_step" for t in n.targets))
        end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "resume_batches" for t in n.targets))
        code = compile(ast.Module(body=method.body[start:end + 1], type_ignores=[]), str(target), "exec")
        with patch.dict(os.environ, {"TRAINKIT_RESUME_STEP": "300"}):
            namespace = {"os": os, "args": SimpleNamespace(resume="state", max_train_steps=1000,
                         gradient_accumulation_steps=4), "accelerator": SimpleNamespace(is_local_main_process=True),
                         "num_update_steps_per_epoch": 14, "tqdm": lambda *a, **k: k}
            exec(code, namespace)
        self.assertEqual(namespace["global_step"], 300)
        self.assertEqual(namespace["epoch_to_start"], 21)
        self.assertEqual(namespace["resume_batches"], 24)
        self.assertEqual(namespace["progress_bar"]["initial"], 300)

    def test_fp8_fixes_survive_a_fresh_vendor_checkout(self):
        vendor = self.root / "vendor/musubi-tuner"
        target = vendor / "src/musubi_tuner/qwen_image/qwen_image_utils.py"
        target.parent.mkdir(parents=True)
        # The exact old compatibility block, wrapped in a syntactically valid
        # minimal module. No model, tokenizer or trainer is loaded.
        prefix = "def prepare(vl_model):\n    if True:\n        if True:\n            if True:\n                for module in vl_model.modules():\n"
        target.write_text(prefix + ORIGINAL_NORM + "\n                        pass\n" + ORIGINAL_DECODER + "\n", "utf-8")
        ensure_qwen_fp8_support(vendor)
        once = target.read_text("utf-8")
        ensure_qwen_fp8_support(vendor)
        self.assertEqual(once, target.read_text("utf-8"))
        self.assertIn(FP8_NORM, once)
        self.assertIn(FP8_DECODER, once)
        self.assertIn("import inspect", once)


if __name__ == "__main__":
    unittest.main()
