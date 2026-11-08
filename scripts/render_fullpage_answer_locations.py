#!/usr/bin/env python3
"""Render ground-truth location annotations, explicitly not model outputs."""
import argparse
import json
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont

from ilm.image_edit.baseline import save_json,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--locations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.locations.resolve();proof=json.loads((root/'report.json').read_text());rows=[]
    for name in ('train_locations.jsonl','validation_locations.jsonl'):
        if sha(root/name)!=proof['files_sha256'][name]:raise ValueError('location annotations changed')
        rows.extend(map(json.loads,(root/name).read_text().splitlines()))
    chosen=[('train',next(r for r in rows if r['split']=='train')),
        ('held',next(r for r in rows if r['split']=='validation'))]
    used={r['id'] for _,r in chosen}
    chosen.append(('multiline',next(r for r in rows if r['id'] not in used and
        any(len(s['pieces'])>1 for s in r['targets']))))
    args.output.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    figures=[]
    for kind,row in chosen:
        source=root/row['input']
        if sha(source)!=row['input_sha256']:raise ValueError('actual source page changed')
        with Image.open(source) as im:
            page=Image.new('RGB',(1024,1104),'white');page.paste(im,(0,80))
        draw=ImageDraw.Draw(page)
        draw.text((24,14),'Ground-truth answer locations - no model prediction',font=font,fill='black')
        draw.text((24,43),row['id']+' | '+kind+' | orange: all canonical occurrences',font=font,fill='black')
        for span in row['targets']:
            for piece in span['pieces']:
                x0,y0,x1,y1=piece['box']
                draw.rectangle((x0,y0+80,x1-1,y1+79),outline='#d26900',width=2)
        name=kind+'.png';page.save(args.output/name)
        figures.append(dict(id=row['id'],split=row['split'],path=name,sha256=sha(args.output/name),
            source_PNG_sha256=row['input_sha256'],canonical_occurrences=len(row['targets'])))
    save_json(args.output/'report.json',dict(implementation_sha256=sha(Path(__file__)),
        preparation_sha256=sha(root/'report.json'),figures=figures,ground_truth_annotation_only=True,
        model_outputs=False,language_learning_validated=False))
    print('Rendered three full-page annotation examples; no model predictions.')


if __name__=='__main__':main()
