"""Prepare parallel pictured questions about verified, family-disjoint Hanziyuan exemplars."""
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_visual_sequence_corpus import ROOT, Rasterizer
from scripts.train_visual_extractive_reader import save_page

SPLITS = ROOT/'artifacts/historical_glyph_splits_v1'
CACHE = ROOT/'artifacts/cache/v34_historic_glyph_rasters_32.pt'
OUT = ROOT/'data/source_plates_v1'
STAGES = {'oracle':('甲骨文','oracle'), 'bronze':('金文','bronze'),
          'seal':('篆書','seal'), 'liushutong':('六書通','liushu')}
PAIR_OPTIONS = tuple(combinations(sorted(STAGES),2))


def question(char, stages, language, split, alternate):
    words = [STAGES[s][language] for s in stages]
    if language:
        if split == 'development':
            return 'Show '+char+': '+', '.join(words)
        return ('/'.join(words)+' for '+char) if alternate else (char+': '+' then '.join(words))
    if split == 'development':
        return '依序列出'+char+'的'+'、'.join(words)
    return ('請寫'+char+'的'+'再'.join(words)) if alternate else ('取'+'、'.join(words)+'，字為'+char)


def write_yan_example(tiles):
    # Select an actual sampled 言 card by deterministic order, retaining its true split identity.
    demo=OUT/'yan';demo.mkdir(exist_ok=True)
    for card,rows in enumerate(load(OUT/'train_cards.json')):
        yan=[r for r in rows if r['char']=='言']
        if len(yan)!=2:
            continue
        with np.load(OUT/'train.npz') as f:
            indices=np.flatnonzero((f['card']==card)&(f['length']==2)&(f['language']==1))
            ordered=sorted((r['stage'] for r in yan),key=list(STAGES).index)
            idx=next(i for i in indices if [rows[j]['char'] for j in f['row'][i]]==['言','言']
                     and [rows[j]['stage'] for j in f['row'][i]]==ordered)
            ids=f['context'][idx]
            save_page(demo/'question.png',tiles[ids[:24]])
            save_page(demo/'plate.png',tiles[ids[24:]])
            save_page(demo/'source_answer.png',tiles[f['target'][idx]])
            write_new(demo/'source.json',dict(split='train',card=card,index=int(idx),stages=ordered,rows=rows))
        break


def main():
    torch.set_num_threads(2)
    report = load(SPLITS/'report.json')
    require(sha(CACHE)==report['raster_cache']['cache_sha256'],'Unexpected historical cache')
    raster = Rasterizer()
    bank = {bytes(128):0}

    def store(tile):
        return bank.setdefault(tile,len(bank))

    def text_ids(text):
        return [store(raster.tile(c)) for c in text]

    cache = torch.load(CACHE,weights_only=True,map_location='cpu')['pixels'].numpy()
    packed = np.packbits(cache.reshape(len(cache),-1),axis=1)
    used, details = {}, {}
    OUT.mkdir()
    for split,seed,total in [('train',714,6000),('development',715,600)]:
        rng = np.random.default_rng(seed)
        path = SPLITS/(split+'.json')
        require(sha(path)==report['partitions'][split]['file_sha256'],'Manifest changed')
        options = defaultdict(list)
        for entry in load(path):
            try:
                key, = text_ids(entry['char'])
            except ValueError:
                continue
            by_stage = defaultdict(list)
            for glyph in entry['glyphs']:
                by_stage[glyph['stage']].append(glyph)
            for pair in PAIR_OPTIONS:
                if all(by_stage[stage] for stage in pair):
                    options[pair].append((entry,key,by_stage))
        contexts,targets,cards,langs,lengths,source_rows = [],[],[],[],[],[]
        seen = set()
        evidence = []
        used[split] = {'family':set(), 'raster':set(), 'label':set(), 'entry':set()}
        while len(evidence)<total:
            pair = PAIR_OPTIONS[rng.integers(len(PAIR_OPTIONS))]
            entries = options[pair]
            left,right = (entries[i] for i in rng.choice(len(entries),2,replace=False))
            if left[0]['family_sha256']==right[0]['family_sha256'] or left[1]==right[1]:
                continue
            labelled = []
            for entry,key,staged in (left,right):
                for stage in pair:
                    choices = staged[stage]
                    glyph = choices[rng.integers(len(choices))]
                    tile = packed[glyph['raster_index']].tobytes()
                    require(hashlib.sha256(tile).hexdigest()==glyph['raster_sha256'],'Historical pixels differ')
                    labelled.append(dict(char=entry['char'], family=entry['family_sha256'], entry=entry['entry_id'],
                                         stage=stage, label=glyph['label'], raster=glyph['raster_sha256'],
                                         svg=glyph['sha256'], key=key, tile=store(tile)))
            if len({r['tile'] for r in labelled})<4:
                continue
            signature=tuple(sorted((r['entry'],r['label']) for r in labelled))
            if signature in seen:
                continue
            seen.add(signature)
            rng.shuffle(labelled)
            card=len(evidence)
            source=[]
            for row in labelled:
                header=text_ids(STAGES[row['stage']][0])
                source += [row['key'],*header,*([0]*(3-len(header))),row['tile']]
                for k in used[split]:
                    used[split][k].add(row[k])
            source += [0]*4
            evidence.append(labelled)
            alternate = bool(rng.integers(2))
            for char in (left[0]['char'],right[0]['char']):
                for language in (0,1):
                    for wanted in [(pair[0],),(pair[1],),pair,pair[::-1]]:
                        q = text_ids(question(char,wanted,language,split,alternate))
                        require(len(q)<=24,'Prompt exceeds visible budget')
                        loc = [next(i for i,r in enumerate(labelled) if r['char']==char and r['stage']==s) for s in wanted]
                        contexts.append([0]*(24-len(q))+q+source)
                        targets.append([labelled[k]['tile'] for k in loc]+[0]*(2-len(loc)))
                        cards.append(card)
                        langs.append(language)
                        lengths.append(len(wanted))
                        source_rows.append(loc+[-1]*(2-len(loc)))
        np.savez_compressed(OUT/(split+'.npz'),context=np.array(contexts,np.uint16),target=np.array(targets,np.uint16),
                            card=np.array(cards,np.int32),language=np.array(langs,np.uint8),length=np.array(lengths,np.uint8),
                            row=np.array(source_rows,np.int8))
        write_new(OUT/(split+'_cards.json'),evidence)
        details[split]=dict(cards=total,questions=len(contexts),used={k:len(v) for k,v in used[split].items()},
                            examples_sha256=sha(OUT/(split+'.npz')),cards_sha256=sha(OUT/(split+'_cards.json')))
    overlaps={k:len(used['train'][k]&used['development'][k]) for k in used['train']}
    require(not any(overlaps.values()),'Historical split collision')
    require(len(bank)<65536,'Tile index overflow')
    tiles=np.frombuffer(b''.join(bank),np.uint8).reshape(-1,128)
    np.save(OUT/'pixels.npy',tiles)
    write_yan_example(tiles)
    write_new(OUT/'report.json',dict(splits=details,overlap=overlaps,pixels_sha256=sha(OUT/'pixels.npy'),
              cache_sha256=sha(CACHE),yan_example=(OUT/'yan/source.json').exists(),program_sha256=sha(Path(__file__))))
    print(details,overlaps,flush=True)

if __name__=='__main__':
    main()
