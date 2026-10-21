"""Import imagegen pairs without changing originals or overwriting curated datasets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image, ImageOps

from .data import ROOT, config

CATEGORIES = {'human': 'people', 'animal': 'animals', 'scene': 'scenes', 'fantasy': 'fantasy'}
ALIASES = {'cartoon': 'iliang_cartoon', 'doubao-3d': 'iliang_doubao_3d',
           'miyazaki': 'iliang_miyazaki', 'jojo': 'iliang_jojo',
           'van-gogh': 'iliang_van_gogh', 'sand-art': 'iliang_sand_art'}


def normalized_caption(text, style, trigger):
    text = text.strip()
    if not text:
        raise ValueError('Empty source caption')
    text = re.sub(r'^' + re.escape(ALIASES[style]) + r'\b(?:\s+style)?[.,:]?\s*', '', text)
    tokens = set(re.findall(r'\biliang_[a-z0-9_]+\b', text))
    if tokens - {trigger}:
        raise ValueError('Unexpected/mixed trigger in caption: ' + ', '.join(sorted(tokens)))
    return (text if trigger in tokens else trigger + ', ' + text).strip() + '\n'


def image_info(path):
    with Image.open(path) as image:
        if getattr(image, 'n_frames', 1) != 1:
            raise ValueError('Animated image: ' + str(path))
        rgb = ImageOps.exif_transpose(image).convert('RGB')
        width, height = rgb.size
        if min(width, height) < 256 or max(width/height, height/width) > 3:
            raise ValueError('Invalid training image dimensions: ' + str(path))
        pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest(), pixels, width/height


def import_generated(root=ROOT, source_root=None, style=None):
    root = Path(root).resolve()
    source_root = Path(source_root or root).resolve()
    plan = json.loads((source_root/'metadata/plan.json').read_text('utf-8'))
    settings = config('styles', root)
    all_styles = {x['id']: x for x in settings['styles']}
    if style and style not in all_styles:
        raise ValueError('Unknown style: ' + style)
    seen = set()
    counts = Counter()
    for row in plan:
        ident = row['id']
        if not isinstance(ident, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', ident) or ident in seen:
            raise ValueError('Unsafe or duplicate sample id: ' + str(ident))
        seen.add(ident)
        if row['category'] not in CATEGORIES:
            raise ValueError('Unknown sample category')
        counts[CATEGORIES[row['category']]] += 1
    if dict(counts) != settings['categories']:
        raise ValueError('Source plan category counts differ from config/styles.json')
    selected = [all_styles[style]] if style else list(all_styles.values())
    pending, records, cached, existing = [], [], {}, 0
    for selected_style in selected:
        ident_style = selected_style['id']
        target_hashes, source_hashes = set(), set()
        for row in plan:
            ident, category = row['id'], CATEGORIES[row['category']]
            source = source_root/'image'/(ident+'.png')
            target = source_root/'train'/ident_style/(ident+'.png')
            caption_path = target.with_suffix('.txt')
            for candidate in (source, target, caption_path):
                if not candidate.is_file() or not candidate.resolve().is_relative_to(source_root):
                    raise ValueError('Missing/unsafe input: ' + str(candidate))
            for candidate in (source, target):
                if candidate not in cached:
                    cached[candidate] = image_info(candidate)
            source_sha, source_pixels, source_ratio = cached[source]
            target_sha, target_pixels, target_ratio = cached[target]
            if source_pixels == target_pixels or abs(source_ratio-target_ratio)>0.03:
                raise ValueError('Invalid source/target pair: ' + ident_style+'/'+ident)
            if target_pixels in target_hashes or source_pixels in source_hashes:
                raise ValueError('Duplicate image/source within style: ' + ident_style+'/'+ident)
            target_hashes.add(target_pixels)
            source_hashes.add(source_pixels)
            caption = normalized_caption(caption_path.read_text('utf-8-sig'), ident_style, selected_style['trigger'])
            if any(x in caption.lower() for x in ('todo', '<describe', 'replace me')):
                raise ValueError('Unfinished caption: ' + str(caption_path))
            destination = root/'datasets'/ident_style/category/(ident+'.png')
            source_destination = root/'datasets'/ident_style/'sources'/category/source.name
            for original, dest, digest, content in ((target,destination,target_sha,None),
                                                   (source,source_destination,source_sha,None),
                                                   (None,destination.with_suffix('.txt'),None,caption.encode('utf-8'))):
                if not dest.resolve().is_relative_to(root/'datasets'):
                    raise ValueError('Unsafe destination: ' + str(dest))
                if dest.exists():
                    matches = (dest.is_file() and (dest.read_bytes()==content if content is not None
                               else hashlib.sha256(dest.read_bytes()).hexdigest()==digest))
                    if not matches:
                        raise ValueError('Existing dataset differs; refusing to overwrite: ' + str(dest))
                    existing += 1
                else:
                    pending.append((original, dest, content))
            records.append(dict(style=ident_style, id=category+'/'+ident, source_sha256=source_sha,
                                target_sha256=target_sha, caption_review='pending'))
    # All inputs and conflicts are checked before the first destination is written.
    for original, destination, content in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if content is None:
            with original.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        else:
            with destination.open('xb') as dst:
                dst.write(content)
    receipt = dict(source_root=str(source_root), styles=[s['id'] for s in selected],
                   samples=len(records), created_files=len(pending), unchanged_files=existing,
                   caption_review='pending', records=records)
    (root/'runtime').mkdir(exist_ok=True)
    (root/'runtime/import-generated.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf-8')
    return {k:v for k,v in receipt.items() if k!='records'}
