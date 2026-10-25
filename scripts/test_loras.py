"""Sequential held-out Qwen Edit evaluation through a local ComfyUI server."""
import argparse
import hashlib
import html
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trainkit.exporter import inspect_lora
from trainkit.locking import training_lock


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def request(url, path, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url + path, payload, {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode('utf-8', errors='replace')) from error


def build_graph(image, lora, strength, prompt, args, prefix):
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': 'qwen_image_edit_2511_bf16.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'TrainingValidationLora', 'inputs': {
            'model': ['1', 0], 'lora_name': lora, 'strength_model': strength}},
        '3': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['2', 0], 'shift': 3.0}},
        '4': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'type': 'qwen_image', 'device': 'default'}},
        '5': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '6': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '7': {'class_type': 'Easy_QwenEdit2509', 'inputs': {
            'clip': ['4', 0], 'vae': ['5', 0], 'image1': ['6', 0], 'latent_image': ['6', 0],
            'auto_resize': 'pad', 'vl_size': 384, 'prompt': prompt,
            'system_prompt': "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}},
        '8': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['7', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '10': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['7', 1], 'reference_latents_method': 'index_timestep_zero'}},
        '11': {'class_type': 'KSampler', 'inputs': {
            'model': ['3', 0], 'positive': ['8', 0], 'negative': ['10', 0], 'latent_image': ['7', 2],
            'seed': args.seed, 'steps': args.steps, 'cfg': 4.0,
            'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['5', 0]}},
        '9': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': prefix}},
    }
    if strength == 0:
        graph.pop('2')
        graph['3']['inputs']['model'] = ['1', 0]
    return graph


def make_plan(args):
    styles = json.loads((ROOT / 'config/styles.json').read_text('utf-8'))['styles']
    names = [s['id'] for s in styles] if args.style == 'all' else args.style.split(',')
    if set(names) - {s['id'] for s in styles}:
        raise ValueError('Unknown style: ' + args.style)
    jobs = []
    for style in (s for s in styles if s['id'] in names):
        manifests = list((ROOT / 'runs/qwen' / style['id']).glob('*/dataset-manifest.json'))
        # Multiple data fingerprints must be selected explicitly, never guess from directory times.
        if args.fingerprint:
            manifests = [p for p in manifests if p.parent.name == args.fingerprint]
        if len(manifests) != 1:
            raise ValueError(f"{style['id']}: found {len(manifests)} runs; specify --fingerprint")
        run = manifests[0].parent
        manifest = json.loads(manifests[0].read_text('utf-8'))
        if manifest['backend'] != 'qwen' or manifest['style'] != style['id']:
            raise ValueError('Wrong dataset manifest: ' + str(manifests[0]))
        records = manifest['validation']
        train_ids = {r['id'] for r in manifest['train']}
        train_sources = {r['source_sha256'] for r in manifest['train']}
        records = sorted(records, key=lambda r: (['people', 'animals', 'scenes', 'fantasy'].index(r['category']), r['id']))
        # First cover different categories, then add remaining held-out examples.
        first = [next(r for r in records if r['category'] == c) for c in
                 ['people', 'animals', 'scenes', 'fantasy'] if any(r['category'] == c for r in records)]
        records = (first + [r for r in records if r not in first])[:args.images]
        strengths = [style['lora_strength']] if args.strengths == 'default' else [float(x) for x in args.strengths.split(',')]
        prompt = (f"Restyle the entire input in {style['trigger']} style: {style['description']}. "
                  'Preserve recognizable subjects, original scene, framing and spatial relationships. '
                  'Restyle the existing background as well; do not remove, replace or invent scenery.')
        for step in args.checkpoints.split(','):
            suffix = '' if step == 'final' else '-step' + str(int(step)).zfill(8)
            checkpoint = run / 'output' / f"{style['id']}-qwen{suffix}.safetensors"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            sha = digest(checkpoint)
            for strength in strengths:
                if not 0 < strength <= 2:
                    raise ValueError('Strength must be >0 and <=2')
                for record in records:
                    if record['id'] in train_ids or record['source_sha256'] in train_sources:
                        raise ValueError('Training/validation leakage: ' + record['id'])
                    name = record['id'].replace('/', '__') + '.png'
                    source, target = run / 'validation-sources' / name, run / 'validation' / name
                    if not source.is_file() or not target.is_file():
                        raise FileNotFoundError('Missing held-out pair: ' + name)
                    with Image.open(source) as picture:
                        rgb = ImageOps.exif_transpose(picture).convert('RGB')
                        pixel_sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    if pixel_sha != record['source_sha256']:
                        raise ValueError('Held-out source changed: ' + str(source))
                    jobs.append(dict(style=style['id'], checkpoint=str(checkpoint), checkpoint_sha256=sha,
                                     step=step, strength=strength, source=str(source), source_sha256=digest(source),
                                     target=str(target), prompt=prompt, sample=record['id']))
        if args.baseline:
            for job in [j.copy() for j in jobs if j['style'] == style['id'] and j['step'] == args.checkpoints.split(',')[0]
                        and j['strength'] == strengths[0]]:
                job.update(strength=0, step='baseline')
                jobs.append(job)
    return jobs


def report(directory, jobs):
    cards = []
    for job in jobs:
        pictures = ''.join(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="{file}"></figure>'
                           for label, file in [('Input', job['id'] + '-input.png'), ('Reference', job['id'] + '-target.png'),
                                               ('Generated', job['id'] + '.png')] if (directory / file).is_file())
        label = f"{job['style']} | {job['sample']} | {job['step']} | strength={job['strength']}"
        cards.append('<h3>' + html.escape(label) + '</h3><div>' + pictures + '</div>')
    (directory / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>LoRA validation</title>'
        '<style>body{background:#17191c;color:#eee;font:16px sans-serif;margin:24px}div{display:flex;gap:12px}'
        'figure{margin:0;width:31%}img{width:100%;object-fit:contain}h3{margin-top:36px}</style>'
        '<h1>Held-out LoRA comparison</h1><p>Reference images are targets, never inference inputs. '
        'Review identity, subject count, pose, background and style. A successful run is not a quality score.</p>'
        + ''.join(cards), encoding='utf-8')


def connect(args, comfy):
    try:
        request(args.url, '/system_stats')
    except (OSError, RuntimeError):
        python = Path(args.comfy_python) if args.comfy_python else ROOT.parent / 'python/python.exe'
        log = ROOT / 'runtime/lora-test-comfy.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('ab') as stream:
            process = subprocess.Popen([str(python), str(comfy / 'main.py'), '--listen', '127.0.0.1',
                '--port', str(urllib.parse.urlsplit(args.url).port or 8188), '--lowvram', '--reserve-vram', '1'],
                cwd=comfy, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        for _ in range(180):
            if process.poll() is not None:
                raise RuntimeError(f'ComfyUI exited. Read {log}')
            try:
                request(args.url, '/system_stats')
                break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError(f'ComfyUI startup timed out. Read {log}')
    return request(args.url, '/object_info')


def run(args):
    parts = urllib.parse.urlsplit(args.url)
    if parts.hostname not in ('127.0.0.1', 'localhost') or parts.scheme != 'http':
        raise ValueError('Use a local ComfyUI http://127.0.0.1:PORT URL')
    jobs = make_plan(args)
    settings = {k: getattr(args, k) for k in ('seed', 'size', 'steps')}
    key = hashlib.sha256(json.dumps([settings, jobs], sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / 'runs/evaluation' / key
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        job['id'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()[:16]
    (directory / 'plan.json').write_text(json.dumps(dict(settings=settings, jobs=jobs), indent=2), 'utf-8')
    print(f'{len(jobs)} images; report: {directory / "index.html"}', flush=True)
    if args.dry_run:
        return
    comfy = Path(args.comfy_dir).resolve()
    destination = comfy / 'custom_nodes/training_lora_validation/__init__.py'
    node_source = Path(__file__).with_name('comfy_validation_node.py')
    if not destination.exists() or digest(destination) != digest(node_source):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(node_source, destination)
    info = connect(args, comfy)
    sample_graph = build_graph('dummy.png', 'dummy.safetensors', .85, 'test', args, 'test')
    missing = {n['class_type'] for n in sample_graph.values()} - info.keys()
    if missing:
        raise RuntimeError(f'Missing nodes: {sorted(missing)}. Restart ComfyUI to load training_lora_validation. '
                           'Easy_QwenEdit2509 requires the existing ComfyUI-Apt_Preset plugin.')
    for node, field, value in [('UNETLoader', 'unet_name', 'qwen_image_edit_2511_bf16.safetensors'),
                              ('CLIPLoader', 'clip_name', 'qwen_2.5_vl_7b_fp8_scaled.safetensors'),
                              ('VAELoader', 'vae_name', 'qwen_image_vae.safetensors')]:
        if value not in info[node]['input']['required'][field][0]:
            raise ValueError(f'Missing ComfyUI model: {value}')
    queue = request(args.url, '/queue')
    active_ids = {item[1] for item in queue.get('queue_running', []) + queue.get('queue_pending', [])}
    saved_ids = set()
    for job in jobs:
        state_path = directory / (job['id'] + '.json')
        if state_path.exists():
            saved_ids.add(json.loads(state_path.read_text('utf-8'))['prompt_id'])
    if active_ids - saved_ids:
        raise RuntimeError('ComfyUI already has queued/running work. Wait until it finishes, then rerun.')
    checked = set()
    for index, job in enumerate(jobs, 1):
        output = directory / (job['id'] + '.png')
        state_file = directory / (job['id'] + '.json')
        if output.exists():
            with Image.open(output) as picture:
                picture.verify()
            print(f'[{index}/{len(jobs)}] already saved {job["style"]}/{job["sample"]}', flush=True)
            continue
        if job['checkpoint'] not in checked:
            inspected = inspect_lora(job['checkpoint'], 'qwen')
            print(f'{job["style"]}: validated {inspected["lora_layers"]} LoRA layers', flush=True)
            checked.add(job['checkpoint'])
        lora_name = str(Path('training-validation') / f"{job['checkpoint_sha256']}.safetensors")
        lora_path = comfy / 'models/loras' / lora_name
        if not lora_path.exists() or digest(lora_path) != job['checkpoint_sha256']:
            lora_path.parent.mkdir(parents=True, exist_ok=True)
            pending_lora = lora_path.with_suffix('.part')
            shutil.copyfile(job['checkpoint'], pending_lora)
            if digest(pending_lora) != job['checkpoint_sha256']:
                raise ValueError('Copied LoRA hash mismatch: ' + str(pending_lora))
            pending_lora.replace(lora_path)
        for label, path in [('input', job['source']), ('target', job['target'])]:
            with Image.open(path) as picture:
                rgb = ImageOps.exif_transpose(picture).convert('RGB')
                ratio = args.size / max(rgb.size)
                size = tuple(max(64, round(x * ratio / 16) * 16) for x in rgb.size)
                rgb.resize(size, Image.Resampling.LANCZOS).save(directory / f"{job['id']}-{label}.png")
        image_name = f"training-validation/{job['source_sha256']}-{args.size}.png"
        input_path = comfy / 'input' / image_name
        input_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / (job['id'] + '-input.png'), input_path)
        graph = build_graph(image_name, lora_name, job['strength'], job['prompt'], args,
                            f'training-validation/{key}/{job["id"]}')
        if state_file.exists():
            state = json.loads(state_file.read_text('utf-8'))
            previous = request(args.url, '/history/' + state['prompt_id'])
            queue = request(args.url, '/queue')
            queued = {item[1] for item in queue.get('queue_running', []) + queue.get('queue_pending', [])}
            if state['prompt_id'] not in previous and state['prompt_id'] not in queued:
                raise RuntimeError(f'Task {state["prompt_id"]} is absent from this ComfyUI session. '
                                   f'Archive {state_file} to retry this image explicitly.')
        else:
            accepted = request(args.url, '/prompt', {'prompt': graph, 'client_id': 'lora-test-' + key})
            state = dict(job=job, settings=settings, workflow=graph, prompt_id=accepted['prompt_id'], started=time.time())
            state_file.write_text(json.dumps(state, indent=2), 'utf-8')
        print(f'[{index}/{len(jobs)}] {job["style"]} {job["step"]} strength={job["strength"]} {job["sample"]}', flush=True)
        deadline = time.monotonic() + args.timeout
        next_update = time.monotonic() + 30
        while time.monotonic() < deadline:
            history = request(args.url, '/history/' + state['prompt_id']).get(state['prompt_id'])
            if history:
                state['history'] = history
                state['seconds'] = time.time() - state['started']
                state_file.write_text(json.dumps(state, indent=2), 'utf-8')
                if history.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError(f'Generation failed; see {state_file}')
                images = history.get('outputs', {}).get('9', {}).get('images', [])
                if not images:
                    raise RuntimeError(f'No SaveImage output; see {state_file}')
                query = urllib.parse.urlencode(images[0])
                with urllib.request.urlopen(args.url + '/view?' + query, timeout=60) as response:
                    contents = response.read()
                with Image.open(BytesIO(contents)) as result:
                    result.verify()
                pending = output.with_suffix('.part')
                pending.write_bytes(contents)
                pending.replace(output)
                report(directory, jobs)
                break
            if time.monotonic() >= next_update:
                print(f'  Waiting for ComfyUI ({round(time.time() - state["started"])} seconds elapsed)...', flush=True)
                next_update = time.monotonic() + 30
            time.sleep(2)
        else:
            raise RuntimeError(f'Timed out waiting for {state["prompt_id"]}; task may still be running. '
                               'Rerun the same command to reconnect, without submitting it again.')
    report(directory, jobs)
    print(f'Done. Open {directory / "index.html"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', default='all', help='all or comma-separated style IDs')
    parser.add_argument('--checkpoints', default='final', help='final or e.g. 600,800,1000')
    parser.add_argument('--strengths', default='default', help='default or e.g. 0.6,0.85,1.0')
    parser.add_argument('--images', type=int, choices=range(1, 8), default=3)
    parser.add_argument('--size', type=int, choices=[512, 768, 1024], default=512)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=3600, help='Seconds per image')
    parser.add_argument('--url', default='http://127.0.0.1:8188')
    parser.add_argument('--comfy-dir', default=str(ROOT.parent / 'ComfyUI'))
    parser.add_argument('--comfy-python')
    parser.add_argument('--fingerprint')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps < 1 or args.timeout < 1:
        parser.error('steps and timeout must be positive')
    try:
        with training_lock(ROOT):
            run(args)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print('ERROR: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
