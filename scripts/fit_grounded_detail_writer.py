"""Train the cross-typeface physical writer using only original modern training passages."""
from collections import Counter
import hashlib
import os
from pathlib import Path
import resource
import time

import numpy as np
from fontTools.ttLib import TTFont
from PIL import ImageFont

from ilm.visual_lm.grounded_detail_writer import GroundedDetailWriter
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_passages import MANIFEST
from scripts.prepare_visual_sequence_corpus import CMRC, FONT, ROOT, Rasterizer, context_hash

OUT = ROOT/'artifacts/grounded_detail_writing_v1'
PROTOCOL = ROOT/'references/grounded_detail_writing_protocol.md'
SERIF = Path('/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
READER = ROOT/'artifacts/visual_extractive_reader_v1/reader.joblib'
READING = ROOT/'references/evidence/visual_extractive_reading_20261118/result.json'


def guard(start):
    available=next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and int(available.split()[1])/1024**2>=24, 'CPU host reserve')
    require(time.monotonic()-start<1500 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<4,
            'CPU process ceiling')


class TargetRasterizer(Rasterizer):
    def __init__(self):
        self.font=ImageFont.truetype(str(SERIF),26,index=2)
        with TTFont(SERIF,fontNumber=2) as face:
            self.coverage=set(face.getBestCmap())
        self.cache={}


def packed(raw):
    return np.frombuffer(raw,dtype=np.uint8).reshape(-1,128)


def heldout(raw):
    return int.from_bytes(hashlib.sha256(raw).digest(),'big')%10==0


def fidelity(p,y):
    same=(p==y).all(1)
    ink=np.unpackbits(p,axis=1)
    gold=np.unpackbits(y,axis=1)
    ink_f1=2*(ink & gold).sum(1)/(ink.sum(1)+gold.sum(1)).clip(1)
    return dict(pairs=len(y),exact=int(same.sum()),mean_ink_f1=float(ink_f1.mean()))


def main():
    start=time.monotonic()
    guard(start)
    OUT.mkdir(exist_ok=False)
    manifest=load(MANIFEST)
    source=CMRC/'cmrc2018_train.json'
    require(sha(source)==manifest['source_sha256'] and sha(FONT)==manifest['font_sha256'], 'Original source changed')
    require(sha(READER)==load(READING)['model_sha256'],'Frozen source reader changed')
    corpus={context_hash(p['context']):p['context'] for a in load(source)['data'] for p in a['paragraphs']}
    source_font,target_font=Rasterizer(),TargetRasterizer()
    chars=Counter()
    for row in manifest['passages']:
        txt=corpus[row['group']]
        require(hashlib.sha256(b''.join(source_font.tile(c) for c in txt)).hexdigest()==row['ink_sha256'],'Training pixels')
        chars.update(txt)
    events=Counter()
    unsupported={}
    for char,freq in chars.items():
        try:
            events[(source_font.tile(char),target_font.tile(char))]+=freq
        except ValueError:
            unsupported[char]=freq
    require(len(manifest['passages'])==2127,'Complete training roster')
    pairs=sorted(events)
    src=packed(b''.join(a for a,_ in pairs))
    dst=packed(b''.join(b for _,b in pairs))
    weights=np.array([events[p] for p in pairs])
    hold=np.array([heldout(a) for a,_ in pairs])
    require(hold.any() and not hold.all(),'Visual source holdout')
    fit_begin=time.monotonic()
    model=GroundedDetailWriter.fit(src[~hold],dst[~hold],weights[~hold])
    elapsed=time.monotonic()-fit_begin
    example=src[:24]
    before=model.paint(example)
    weight_path=OUT/'writer.npz'
    model.save(weight_path)
    del model
    model=GroundedDetailWriter.load(weight_path)
    require(np.array_equal(before,model.paint(example)),'Reload changed pixels')
    assays={}
    for name,mask in [('train',~hold),('heldout',hold)]:
        mask=mask & dst.any(1)
        assays[name]={}
        for arm in ['copy','coarse','detail']:
            decoded=src[mask] if arm=='copy' else model.paint(src[mask],detail=arm=='detail')
            assays[name][arm]=fidelity(decoded,dst[mask])
    guard(start)
    result=dict(training_source_sha256=sha(source),font_sha256=[sha(FONT),sha(SERIF)],reader_sha256=sha(READER),
                physical_training_passages=len(manifest['passages']),eligible_pairs=len(pairs),
                distinct_source_bitmaps=len(set(a for a,_ in pairs)),training_occurrences=int(weights[~hold].sum()),
                heldout_occurrences=int(weights[hold].sum()),unsupported_occurrences=sum(unsupported.values()),
                source_bitmaps_with_multiple_target_images=sum(n>1 for n in Counter(a for a,_ in pairs).values()),
                assays=assays,fit_seconds=elapsed,total_seconds=time.monotonic()-start,checkpoint_bytes=weight_path.stat().st_size,
                checkpoint_sha256=sha(weight_path),program_sha256=sha(Path(__file__)),protocol_sha256=sha(PROTOCOL),
                model_code_sha256=sha(ROOT/'ilm/visual_lm/grounded_detail_writer.py'),rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    write_new(OUT/'fit.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
