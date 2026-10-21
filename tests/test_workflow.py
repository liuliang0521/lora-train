import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from trainkit.data import ROOT, split_records, validate
from trainkit.importer import CATEGORIES, import_generated, normalized_caption
from trainkit.locking import training_lock
from trainkit.cli import reuse_model


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT/'config', self.root/'config')
        settings = json.loads((self.root/'config/styles.json').read_text('utf-8'))
        settings['categories'] = {key: 1 for key in CATEGORIES.values()}
        (self.root/'config/styles.json').write_text(json.dumps(settings), 'utf-8')
        (self.root/'image').mkdir()
        (self.root/'train/cartoon').mkdir(parents=True)
        (self.root/'metadata').mkdir()
        self.plan = []
        for index, category in enumerate(CATEGORIES):
            ident = f'{index:03}_{category}'
            self.plan.append(dict(id=ident, category=category))
            target = self.root/'train/cartoon'/(ident+'.png')
            Image.new('RGB', (320,256), (index,80,90)).save(target)
            Image.new('RGB', (320,256), (index,30,50)).save(self.root/'image'/target.name)
            target.with_suffix('.txt').write_text('iliang_cartoon style. A subject in a garden.\n', 'utf-8')
        self.write_plan()

    def write_plan(self):
        (self.root/'metadata/plan.json').write_text(json.dumps(self.plan), 'utf-8')

    def test_import_preserves_raw_captions_and_builds_real_qwen_pairs(self):
        original = self.root/'train/cartoon/000_human.txt'
        original_text = original.read_bytes()
        result = import_generated(self.root, style='cartoon')
        self.assertEqual(result['samples'], 4)
        self.assertEqual(result['created_files'], 12)
        self.assertEqual(original.read_bytes(), original_text)
        self.assertTrue(validate('cartoon', 'sdxl', self.root)['valid'])
        self.assertTrue(validate('cartoon', 'qwen', self.root)['valid'])
        caption = self.root/'datasets/cartoon/people/000_human.txt'
        self.assertIn('iliang_chibi', caption.read_text('utf-8'))
        self.assertNotIn('iliang_cartoon', caption.read_text('utf-8'))
        self.assertEqual(hashlib.sha256((self.root/'image/000_human.png').read_bytes()).hexdigest(),
                         hashlib.sha256((self.root/'datasets/cartoon/sources/people/000_human.png').read_bytes()).hexdigest())

    def test_import_is_idempotent_and_will_not_overwrite_curated_caption(self):
        import_generated(self.root, style='cartoon')
        self.assertEqual(import_generated(self.root, style='cartoon')['created_files'], 0)
        edited = self.root/'datasets/cartoon/people/000_human.txt'
        edited.write_text('iliang_chibi, manually corrected content', 'utf-8')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            import_generated(self.root, style='cartoon')
        self.assertIn('manually corrected', edited.read_text('utf-8'))

    def test_missing_input_does_not_partially_import(self):
        (self.root/'train/cartoon/003_fantasy.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing/unsafe'):
            import_generated(self.root, style='cartoon')
        self.assertFalse((self.root/'datasets').exists())

    def test_path_traversal_is_rejected(self):
        self.plan[0]['id'] = '../outside'
        self.write_plan()
        with self.assertRaisesRegex(ValueError, 'Unsafe or duplicate'):
            import_generated(self.root, style='cartoon')
        self.assertFalse((self.root/'datasets').exists())

    def test_same_source_and_target_is_rejected_before_writing(self):
        shutil.copyfile(self.root/'image/000_human.png', self.root/'train/cartoon/000_human.png')
        with self.assertRaisesRegex(ValueError, 'Invalid source/target pair'):
            import_generated(self.root, style='cartoon')
        self.assertFalse((self.root/'datasets').exists())

    def test_mixed_style_tokens_are_not_silently_relabelled(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected/mixed trigger'):
            normalized_caption('iliang_cartoon style. iliang_soft3d, a cat', 'cartoon', 'iliang_chibi')


class WorkflowTests(unittest.TestCase):
    def test_download_reuse_verifies_receipt_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'model.safetensors'
            target.write_bytes(b'known model contents')
            item = dict(id='sdxl-base', repo='official/repo', file='model.safetensors', path='models/model.safetensors')
            with self.assertRaisesRegex(ValueError, 'without a download receipt'):
                reuse_model(target, item)
            receipt = dict(item, bytes=target.stat().st_size, sha256=hashlib.sha256(target.read_bytes()).hexdigest())
            target.with_suffix('.receipt.json').write_text(json.dumps(receipt), 'utf-8')
            self.assertTrue(reuse_model(target, item))
            target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'failed receipt verification'):
                reuse_model(target, item)

    def test_holdout_subject_ids_match_across_styles_and_backends(self):
        first = [dict(id=category+'/'+str(i), category=category, sha256='a'+str(i))
                 for category,count in {'people':20,'animals':15,'scenes':15,'fantasy':10}.items()
                 for i in range(count)]
        second = [dict(item, sha256='different-'+item['id']) for item in first]
        a,b = split_records(first)
        c,d = split_records(second)
        self.assertEqual((len(a),len(b)), (53,7))
        self.assertEqual({x['id'] for x in b}, {x['id'] for x in d})
        self.assertFalse({x['id'] for x in a} & {x['id'] for x in b})

    def test_training_lock_rejects_concurrent_run_and_releases_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with training_lock(directory):
                with self.assertRaisesRegex(RuntimeError, 'Another training process'):
                    with training_lock(directory):
                        self.fail('Concurrent lock should fail')
            with self.assertRaisesRegex(ValueError, 'simulated'):
                with training_lock(directory):
                    raise ValueError('simulated')
            with training_lock(directory):
                pass


if __name__ == '__main__':
    unittest.main()
