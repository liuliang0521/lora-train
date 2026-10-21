"""Local bookkeeping for paired imagegen training candidates; no API secrets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STYLES = ('cartoon', 'doubao-3d', 'miyazaki', 'jojo', 'van-gogh', 'sand-art')

def make_caption(row, style):
    caption = row['caption'] if style == 'source' else 'iliang_' + style.replace('-', '_') + ' style. ' + row['caption']
    if style == 'sand-art':
        content = row['subject'] + '. ' + row['framing'] + ', ' + row['pose'] + '.'
        content = re.sub(r'\b(?:green|blue|pink|red|gray|grey|black|white|yellow|orange|purple|cobalt|navy|burgundy|rust|ochre|turquoise|silver|brass|brown|beige|cream|pale)\b[ -]*', '', content, flags=re.I)
        caption = 'iliang_sand_art style, flat handcrafted monochrome sand-yellow lightbox drawing, granular sand density, swept contours and luminous cleared highlights. Scene: ' + content
    if style == 'jojo':
        content = re.sub(r'calm expression|surprised expression|laughing|smiling|gentle smile', 'intense expression', row['subject'], flags=re.I)
        caption = 'iliang_jojo style, hard ink outlines, high-contrast cel shading and sculptural drawn volume. ' + content + '. ' + row['framing'] + ', ' + row['pose'] + '.'
    return caption

def record(items):
    plan = {x['id']: x for x in json.loads((ROOT/'metadata/plan.json').read_text('utf-8'))}
    for item in items:
        row = plan[item['id']]
        style = item['style']
        folder = ROOT/'image' if style == 'source' else ROOT/'train'/style
        target = folder/(row['id']+'.png')
        if target.exists():
            raise FileExistsError(str(target))
        with Image.open(item['generated_path']) as image:
            image.verify()
        shutil.copy2(item['generated_path'], target)
        with Image.open(target) as image:
            width, height = image.size
        caption = make_caption(row, style)
        # Content captions describe the source plan; mark them for manual revision against actual outputs.
        target.with_suffix('.txt').write_text(caption+'\n', encoding='utf-8')
        event = dict(id=row['id'], style=style, source='image/'+row['id']+'.png', target=target.relative_to(ROOT).as_posix(), width=width, height=height, sha256=hashlib.sha256(target.read_bytes()).hexdigest(), generated_path=item['generated_path'], generator='builtin-imagegen', created_at=datetime.now(timezone.utc).isoformat(), review='pending', caption_review='pending', prompt=item.get('prompt',''))
        with (ROOT/'metadata/generation.jsonl').open('a',encoding='utf-8') as output:
            output.write(json.dumps(event,ensure_ascii=False)+'\n')
    print(json.dumps(summary(),ensure_ascii=True))

def summary():
    return dict(source=len(list((ROOT/'image').glob('*.png'))), styles={s:len(list((ROOT/'train'/s).glob('*.png'))) for s in STYLES})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', help='JSON file containing completed tool outputs')
    parser.add_argument('--refresh-captions', action='store_true', help='Refresh candidate captions using current rules')
    args = parser.parse_args()
    if args.refresh_captions:
        for row in json.loads((ROOT/'metadata/plan.json').read_text('utf-8')):
            for style in ('source', *STYLES):
                path = ROOT/('image' if style=='source' else 'train/'+style)/(row['id']+'.png')
                if path.exists():
                    path.with_suffix('.txt').write_text(make_caption(row, style)+'\n', 'utf-8')
        print(json.dumps(summary(),ensure_ascii=True))
    elif args.record:
        record(json.loads(Path(args.record).read_text('utf-8')))
    else:
        print(json.dumps(summary(),ensure_ascii=True))
