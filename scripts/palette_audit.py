"""Read-only pixel analysis of the generated sand-art palette."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((ROOT/'train/sand-art').glob('*.png')):
    with Image.open(path) as im:
        hsv = np.asarray(im.convert('HSV'))
    hue, saturation, value = hsv[...,0], hsv[...,1], hsv[...,2]
    chromatic = (saturation>30) & (value>30)
    outside = chromatic & ((hue<18) | (hue>46))
    green_blue = chromatic & (hue>=47) & (hue<=180)
    rows.append(dict(file=path.relative_to(ROOT).as_posix(),outside_yellow_fraction=float(outside.mean()),green_blue_fraction=float(green_blue.mean()),chromatic_fraction=float(chromatic.mean())))
report = dict(images=len(rows),hue_range_degrees=[18*360/255,46*360/255],minimum_saturation=30/255,minimum_value=30/255,flag_threshold_fraction=0.001,flagged=[x['file'] for x in rows if x['outside_yellow_fraction']>0.001],rows=rows,note='Heuristic hue check of chromatic pixels, not a guarantee of exact single-hue rendering or artistic quality. Images are read, never recolored.')
report['green_blue_flagged'] = [x['file'] for x in rows if x['green_blue_fraction']>0]
report['note'] += ' The narrow hue-band flags also detect warm brown shadow variation; they are not evidence of green or blue residuals. Both diagnostics are retained.'
(ROOT/'metadata/sand-palette-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='rows'},ensure_ascii=True))
