"""Check held-out isolation and the actual graph sent to ComfyUI."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

spec = importlib.util.spec_from_file_location('evaluation', Path(__file__).parents[1] / 'scripts/test_loras.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_graph_uses_edit_reference_and_strict_lora(self):
        args = SimpleNamespace(seed=42, steps=20)
        graph = evaluation.build_graph('heldout.png', 'selected.safetensors', .85, 'restyle', args, 'test')
        self.assertEqual(graph['7']['inputs']['image1'], ['6', 0])
        self.assertEqual(graph['7']['inputs']['latent_image'], ['6', 0])
        self.assertEqual(graph['2']['class_type'], 'TrainingValidationLora')
        self.assertEqual(graph['11']['inputs']['model'], ['3', 0])
        self.assertEqual(graph['3']['inputs']['model'], ['2', 0])
        self.assertNotIn('Lightning', json.dumps(graph))
        baseline = evaluation.build_graph('heldout.png', 'selected.safetensors', 0, 'restyle', args, 'test')
        self.assertNotIn('2', baseline)
        self.assertEqual(baseline['3']['inputs']['model'], ['1', 0])

    def test_training_source_cannot_enter_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'config').mkdir()
            style = dict(id='cartoon', trigger='trigger', description='style', lora_strength=.85)
            (root / 'config/styles.json').write_text(json.dumps(dict(styles=[style])))
            run = root / 'runs/qwen/cartoon/fingerprint'
            (run / 'output').mkdir(parents=True)
            (run / 'output/cartoon-qwen.safetensors').write_bytes(b'fixture')
            record = dict(id='people/a', category='people', source_sha256='same-source')
            (run / 'dataset-manifest.json').write_text(json.dumps(dict(
                backend='qwen', style='cartoon', validation=[record],
                train=[dict(id='people/b', source_sha256='same-source')])) )
            args = SimpleNamespace(style='cartoon', fingerprint=None, images=1,
                                   strengths='default', checkpoints='final', baseline=False)
            with patch.object(evaluation, 'ROOT', root), self.assertRaisesRegex(ValueError, 'leakage'):
                evaluation.make_plan(args)

    def test_report_distinguishes_reference_and_generated(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ['test-input.png', 'test-target.png', 'test.png']:
                Image.new('RGB', (16, 16)).save(directory / name)
            evaluation.report(directory, [dict(id='test', style='cartoon', sample='<input>', step='800', strength=.85)])
            page = (directory / 'index.html').read_text('utf-8')
            self.assertIn('Reference', page)
            self.assertIn('Generated', page)
            self.assertIn('&lt;input&gt;', page)


if __name__ == '__main__':
    unittest.main()
