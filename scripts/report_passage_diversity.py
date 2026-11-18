"""Publish the physical score records and first changed answers from the fixed holdout."""
import shutil

import numpy as np
import pyarrow.parquet as pq

from scripts.compare_passage_diversity import OUT, ROOT, load_split
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.prepare_pictured_idioms import RAW, TAG, WIDTH, Rasterizer
from scripts.test_expression_frequency_weight import OUT as WEIGHT_RUN
from scripts.train_visual_extractive_reader import save_page

DEST = ROOT/'references/evidence/passage_diversity_20261118'


def main():
    report = load(OUT/'evaluation.json')
    followup = {'frequency_weight.json':WEIGHT_RUN/'evaluation.json',
                'weight_holdout.json':WEIGHT_RUN/'fresh/preparation.json',
                'local_errors.json':WEIGHT_RUN/'local_errors.json'}
    require(all(path.is_file() for path in followup.values()), 'Complete the frequency comparison and local recount first')
    DEST.mkdir()
    for filename, path in followup.items():
        shutil.copyfile(path, DEST/filename)
    for name in ('data','fits','evaluation','prepare_resources','train_resources','evaluate_resources'):
        shutil.copyfile(OUT/(name+'.json'), DEST/(name+'.json'))
    fresh = report['fresh']
    test = load_split(OUT/'fresh/contexts.npz')
    px = np.load(OUT/'pool/pixels.npy')
    gold = px[test['target']]
    runs, hits = {},{}
    for name in ('repeat','diverse'):
        path = OUT/'evaluation_fresh'/name/'generated.npz'
        require(sha(path)==fresh['arms'][name]['archive_sha256'], 'Generated archive changed')
        v = runs[name] = load_split(path)
        hits[name] = (v['length']==4)&v['stopped']&(v['packed'][:,:4]==gold).all((1,2))
    rows = {key:int(np.flatnonzero(mask)[0]) for key,mask in [
        ('gain',hits['diverse']&~hits['repeat']),('loss',hits['repeat']&~hits['diverse'])] if mask.any()}
    wanted = {int(test['tag'][i]):key for key,i in rows.items()}
    groups = {int(test['group'][i]) for i in rows.values()}
    samples = {}
    written_tiles = {}
    raster = Rasterizer()
    for batch in pq.ParquetFile(RAW/'train.parquet').iter_batches(columns=['idx','content','answers']):
        for record in batch.to_pylist():
            if record['idx'] not in groups:
                continue
            texts = record['content']
            answers = dict(zip((int(t) for s in texts for t in TAG.findall(s)),record['answers']['text'],strict=True))
            for word in answers.values():
                for c in word:
                    written_tiles[raster.tile(c)] = c
            for passage in texts:
                for match in TAG.finditer(passage):
                    tag = int(match[1])
                    if tag not in wanted:
                        continue
                    key = wanted[tag]
                    i = rows[key]
                    samples[key] = dict(row=i,tag=tag,group=record['idx'],
                         before=TAG.sub('□',passage[:match.start()])[-WIDTH:],
                         after=TAG.sub('□',passage[match.end():])[:WIDTH],reference=answers[tag])
    # Only use the source text to label output pixels after the frozen generation.
    for batch in pq.ParquetFile(RAW/'validation.parquet').iter_batches(columns=['answers']):
        words = {c for record in batch.to_pylist() for word in record['answers']['text'] for c in word}
        for c in words:
            try: written_tiles[raster.tile(c)] = c
            except ValueError: continue
    for key, i in rows.items():
        require(np.array_equal(np.frombuffer(b''.join(raster.tile(c) for c in samples[key]['reference']),np.uint8),
                               gold[i].ravel()),'Source reading differs from physical reference')
        out = runs['diverse']['packed'][i,:runs['diverse']['length'][i]]
        sides = px[test['context'][i]]
        for suffix, data in [('before',sides[12:24]), ('after',sides[24:36]), ('actual',out),('reference',gold[i])]:
            save_page(DEST/(key+'_'+suffix+'.png'),data)
        samples[key]['pixel_reading'] = ''.join(written_tiles.get(tile.tobytes(),'�') for tile in out)
    rows = [['Writer','Exact answers','Both in pair','Distinct outputs','Top ten share']]
    for name, label in [('start','Starting writer'),('repeat','Repeated passages'),('diverse','Distinct passages')]:
        arm = fresh['arms'][name]
        rows.append([label,str(arm['exact']),str(arm['both_correct']),str(arm['distinct']),
                     f"{100*arm['most_frequent_ten']/fresh['rows']:.1f}%"])
    write_new(DEST/'figure.json',dict(table=rows,samples=samples))

if __name__ == '__main__':
    main()
