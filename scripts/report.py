"""Audit generated files and build a local paired-image review gallery."""
from collections import Counter, defaultdict
import hashlib
from html import escape
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STYLES = {'cartoon':'Q版卡通', 'doubao-3d':'豆包 3D', 'miyazaki':'宫崎骏手绘', 'jojo':'JOJO', 'van-gogh':'梵高油画', 'sand-art':'沙画'}

def build():
    plan = json.loads((ROOT/'metadata/plan.json').read_text('utf-8'))
    events = [json.loads(x) for x in (ROOT/'metadata/generation.jsonl').read_text('utf-8').splitlines() if x.strip()]
    latest = {(x['style'],x['id']):x for x in events}
    review_path = ROOT/'metadata/reviews.json'
    reviews = json.loads(review_path.read_text('utf-8')) if review_path.exists() else {}
    rows, missing, errors, hashes = [], [], [], defaultdict(list)
    counts = Counter()
    sections = []
    for item in plan:
        cells=[]
        for style,label in [('source','原图'),*STYLES.items()]:
            path = ROOT/('image' if style=='source' else 'train/'+style)/(item['id']+'.png')
            rel=path.relative_to(ROOT).as_posix()
            if not path.exists():
                missing.append(rel)
                cells.append(f'<figure data-style="{style}"><figcaption>{label}</figcaption><p>待生成</p></figure>')
                continue
            try:
                with Image.open(path) as im:
                    im.load()
                    size=im.size
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[digest].append(rel)
                event=latest[(style,item['id'])]
                if digest!=event['sha256']:
                    errors.append({'file':rel,'error':'hash mismatch'})
                caption=path.with_suffix('.txt').read_text('utf-8').strip()
                if not caption:
                    errors.append({'file':rel,'error':'empty caption'})
                if min(size)<1024:
                    errors.append({'file':rel,'error':'resolution below 1024'})
                row=dict(id=item['id'],style=style,source='image/'+item['id']+'.png',target=rel,caption=caption,width=size[0],height=size[1],sha256=digest,category=item['category'],planned_view=item['view'],planned_lighting=item['light_group'],image_review=reviews.get(rel,'pending'),caption_review='pending')
                rows.append(row)
                counts[style]+=1
                cells.append(f'<figure data-style="{style}"><figcaption>{label}</figcaption><a href="{rel}" target="_blank"><img loading="lazy" src="{rel}" alt="{escape(item["id"]+" "+label)}"></a><details><summary>文字标注 · 待核对</summary><p>{escape(caption)}</p></details></figure>')
            except Exception as exc:
                errors.append({'file':rel,'error':str(exc)})
        sections.append(f'<section data-category="{item["category"]}"><h2>{item["id"]}</h2><p>{escape(item["subject"])} · 计划机位 {item["view"]} / 光线 {item["light_group"]}</p><div class="grid">'+''.join(cells)+'</div></section>')
    duplicates=[x for x in hashes.values() if len(x)>1]
    report=dict(expected_sources=60,expected_per_style=60,counts=dict(counts),missing=missing,errors=errors,exact_duplicates=duplicates,planned_categories=dict(Counter(x['category'] for x in plan)),planned_views=dict(Counter(x['view'] for x in plan)),note='Counts and checks concern generated files, not trained-model performance. View/lighting quotas are generation plans, not automatically verified image labels. Caption review is pending. Excluded revisions are not counted.')
    (ROOT/'metadata/audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n','utf-8')
    (ROOT/'metadata/pairs.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows if x['style']!='source'),'utf-8')
    selectors='<label>类别 <select id="category"><option value="all">全部</option><option value="human">人像</option><option value="animal">动物</option><option value="scene">风景静物</option><option value="fantasy">奇幻</option></select></label><label>风格 <select id="style"><option value="all">全部对照</option>'+''.join(f'<option value="{s}">{label}</option>' for s,label in STYLES.items())+'</select></label>'
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>iliang LoRA 数据集对照</title><style>body{margin:0;background:#141824;color:#e9edf5;font:15px/1.5 system-ui}header{padding:24px;position:sticky;top:0;background:#141824f5;z-index:1;border-bottom:1px solid #384157}h1{font-size:24px;margin:0}header p{color:#aebed0;margin:8px 0}label{margin-right:24px}select{padding:6px}main{padding:0 24px}section{padding:24px 0;border-bottom:1px solid #384157}section>p{color:#aebed0}.grid{display:flex;gap:12px;overflow-x:auto;padding-bottom:12px}figure{margin:0;flex:0 0 230px}figcaption{font-weight:600;margin-bottom:6px}img{width:230px;height:230px;object-fit:contain;background:#252b39;border-radius:8px}details{font-size:12px;color:#aebed0}details p{max-width:230px}h2{margin:0;font-size:19px}</style><header><h1>iliang · 六种风格配对数据</h1><p>点击图片查看原分辨率。生成样本与文字标注需筛选；这里不代表已训练或效果已验收。</p>'''+selectors+f'<p>原图 {counts["source"]}/60 · 风格图 {sum(counts[s] for s in STYLES)}/360</p></header><main>'+''.join(sections)+'''</main><script>function filter(){const c=document.querySelector('#category').value,s=document.querySelector('#style').value;document.querySelectorAll('section').forEach(x=>x.hidden=c!=='all'&&x.dataset.category!==c);document.querySelectorAll('figure').forEach(x=>x.hidden=s!=='all'&&x.dataset.style!=='source'&&x.dataset.style!==s)}document.querySelectorAll('select').forEach(x=>x.addEventListener('change',filter));</script></html>'''
    (ROOT/'gallery.html').write_text(html,'utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing','note')},ensure_ascii=True))
    print('Missing:',len(missing))
    if errors or duplicates:
        raise SystemExit(1)

if __name__=='__main__':
    build()
