"""Enumerate the complete greedy writer range and count exact source images it can express."""
import os
import time
from pathlib import Path

import numpy as np

from ilm.visual_lm.layout_patch_language import LayoutPatchLanguage
from ilm.visual_lm.patch_count_language import tile_features
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_context_contrast_drawing import page
from scripts.measure_visual_successors import CMRC, SOURCE, SOURCE_HASH, ROOT, digest, guard
from scripts.prepare_visual_passages import MANIFEST
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_patch_count_language import OUT as EARLIER
from scripts.train_shared_layout_language import OUT as TRIAL


def keys(ink):
    return [row.tobytes() for row in np.packbits(ink.reshape(-1,1024),axis=1)]


def main():
    start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU execution')
    guard(start)
    trial=load(TRIAL/'result.json')
    earlier=load(EARLIER/'result.json')
    for path in ['ordered/successors.npz','ordered/model.npz','ordered/yan.npz']:
        require(sha(TRIAL/path)==trial['files'][path],'Shared-layout evidence changed')
    require(sha(EARLIER/'articles.json')==earlier['files']['articles.json'] and
            sha(EARLIER/'ordered/successors.npz')==earlier['files']['ordered/successors.npz'],'Comparison evidence changed')
    dest=ROOT/'artifacts/shared_layout_support_v1'
    dest.mkdir(exist_ok=False)
    model=LayoutPatchLanguage.load(TRIAL/'ordered/model.npz')
    generated=np.stack([model.draw_layout(z) for z in model.layouts])
    support=set(keys(generated))
    with np.load(TRIAL/'ordered/successors.npz') as f:
        actual,targets=f['ink'], f['targets']
        require(not f['stopped'].any(),'Unexpected end')
    with np.load(EARLIER/'ordered/successors.npz') as f:
        old=f['ink']
        require(np.array_equal(f['targets'],targets),'Different recorded images')
    target_ids, actual_ids, old_ids=map(keys,(targets,actual,old))
    require(all(k in support for k in actual_ids),'A freely drawn output fell outside the enumerated range')
    source=CMRC/'cmrc2018_train.json'
    require(sha(source)==load(MANIFEST)['source_sha256'] and sha(SOURCE)==SOURCE_HASH,'Different text for auditing')
    records=[(a['title'],p['context']) for file in [source,SOURCE] for a in load(file)['data'] for p in a['paragraphs']]
    renderer=Rasterizer()
    labels={}
    for c in sorted(set(''.join(t for _,t in records))):
        try:
            labels.setdefault(renderer.tile(c),[]).append(c)
        except ValueError:
            pass
    def known_han(raw):
        return any(0x4e00<=ord(c)<=0x9fff for c in labels.get(raw,()))
    contexts={(digest(a),digest(p)):p for a,p in records}
    roster=load(EARLIER/'articles.json')
    examples={}
    for i,(y,new,prior) in enumerate(zip(target_ids,actual_ids,old_ids,strict=True)):
        row=roster[i//2]
        txt=contexts[row['article_sha256'],row['paragraph_sha256']]
        pos=row['offsets'][i%2]
        prefix=txt[pos-12:pos]
        require(renderer.tile(txt[pos])==y,'Source location mismatch')
        for category,eligible in [('corrected',new==y and prior!=y),('lost',prior==y and new!=y),
                                  ('wrong_han',new!=y and known_han(new) and not known_han(prior))]:
            if eligible and category not in examples:
                examples[category]=dict(index=i,prefix=prefix,source=txt[pos],old_labels=labels.get(prior,[]),
                                        new_labels=labels.get(new,[]))
                strip=np.unpackbits(np.frombuffer(b''.join(renderer.tile(c) for c in prefix),dtype=np.uint8)).reshape(-1,32,32)
                for suffix,images in [('context',strip),('old',old[i:i+1]),('new',actual[i:i+1]),('target',targets[i:i+1])]:
                    page(dest/f'{category}_{suffix}.png',images)
    count=sum(y in support for y in target_ids)
    ids=np.asarray([y in support for y in target_ids])
    correct_layout,_=tile_features(targets)
    preferred=model.layouts
    train_layouts=set(map(int,preferred))
    report=dict(complete_greedy_layouts=len(generated),distinct_greedy_images=len(support),contexts=len(ids),
                reachable_reference_images=count,unreachable_reference_images=len(ids)-count,
                reference_layout_in_model=int(sum(int(z) in train_layouts for z in correct_layout)),
                articles_with_both_reachable=int(ids.reshape(-1,2).all(1).sum()),
                free_han_bitmap_matches=sum(known_han(k) for k in actual_ids),
                previous_free_han_bitmap_matches=sum(known_han(k) for k in old_ids),
                retained_exact=sum(x==y and o==y for x,y,o in zip(actual_ids,target_ids,old_ids)),
                source_trial_sha256=sha(TRIAL/'result.json'), program_sha256=sha(Path(__file__)), examples=examples,
                complete_range_scope='ordered checkpoint, short or arbitrary history planner, greedy local patch decisions')
    np.savez_compressed(dest/'range.npz',layouts=model.layouts,generated=generated,reachable=ids)
    report['range_sha256']=sha(dest/'range.npz')
    write_new(dest/'result.json',report)
    guard(start)
    print({k:v for k,v in report.items() if k!='examples'})


if __name__=='__main__':
    main()
