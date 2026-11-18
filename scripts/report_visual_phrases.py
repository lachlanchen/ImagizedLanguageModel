"""Recount archived raw glyph images and export selected contexts for the manuscript."""
import argparse
import shutil
from collections import Counter

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from scripts.train_visual_phrases import OUT, ROOT, RUN, WORK, frozen, initialize, load, require, sha, tiles, write_new
from ilm.visual_lm.visual_edit_writer import COPY, DROP, INSERT, REPLACE

def sites_independent(rows):
    for i,row in enumerate(rows):
        source = target = 0
        begin = None
        for action in row['a']:
            if action==INSERT and begin is None:
                begin = target
            if begin is not None and action!=INSERT:
                yield i,source,begin,target
                begin = None
            source += action in (COPY, DROP, REPLACE)
            target += action in (COPY, REPLACE, INSERT)
        require(source==len(row['x']) and target==len(row['y']), 'Bad source trace')


def png(directory, name, packed):
    pixels = np.unpackbits(packed,axis=1).reshape(-1,32,32)
    Image.fromarray(np.where(np.concatenate(pixels,axis=1),0,255).astype('uint8')).save(directory/(name+'.png'))


@torch.inference_mode()
def run(directory):
    directory.mkdir(parents=True)
    torch.set_num_threads(2)
    bank = tiles()
    net = initialize()
    net.load_state_dict(load_file(str(RUN/'source/writer.safetensors')))
    ink = torch.from_numpy(np.unpackbits(bank,axis=1)).bool()
    possible = []
    for target in ink.split(128):
        code = target[:,net.landmarks]
        picture = net.motor(code)>0
        picture[:,net.landmarks] = code
        possible.extend((picture==target).all(1).tolist())
    chart, examples, widths, report = {}, {}, {}, {}
    qids = load(OUT/'preparation.json')['question_tiles'][0][0]
    for arm in ('source','prefix'):
        shutil.copyfile(RUN/arm/'fit.json',directory/(arm+'_fit.json'))
    for split in ('discovery','replication'):
        rows = list(frozen(split))
        sites = list(sites_independent(rows))
        measured = load(RUN/split/'report.json')
        shutil.copyfile(RUN/split/'report.json',directory/(split+'_report.json'))
        runs = {}
        for arm,v in measured['arms'].items():
            path = RUN/split/(arm+'.npz')
            require(sha(path)==v['pixels_sha256'], 'Archived outputs changed')
            with np.load(path) as z:
                streams = np.split(z['pixels'],z['offsets'][1:-1])
                stops = z['stopped']
            correct = np.array([stop and stream.tobytes()==bank[rows[i]['y'][j:k]].tobytes()
                                for (i,c,j,k),stream,stop in zip(sites,streams,stops,strict=True)])
            require(correct.sum()==v['exact'] and v['stopped']==stops.sum(), 'Recount differs')
            runs[arm] = (streams,stops,correct)
        out,stop,exact = runs['source']
        local = runs['local'][2]
        first_wrong = ended_short = grew_long = different_later = capped = 0
        drawable = not_drawable_exact = 0
        frequencies = Counter()
        for n,(i,c,j,k) in enumerate(sites):
            row = rows[i]
            reference = bank[row['y'][j:k]]
            pred = out[n]
            can_draw = all(possible[g] for g in row['y'][j:k])
            drawable += can_draw
            not_drawable_exact += int(exact[n] and not can_draw)
            frequencies[pred.tobytes()] += 1
            if not exact[n]:
                if not np.array_equal(pred[0],reference[0]):
                    first_wrong += 1
                elif len(pred)<len(reference) and np.array_equal(pred,reference[:len(pred)]):
                    ended_short += 1
                elif len(pred)>len(reference) and np.array_equal(pred[:len(reference)],reference):
                    grew_long += 1
                elif not stop[n] and np.array_equal(pred,reference):
                    capped += 1
                else:
                    different_later += 1
            if split!='discovery' or len(examples)==2 or not j or set(row['y'][j:k]) & set(row['x']):
                continue
            name = ('writer' if exact[n] and not local[n] and not runs['prefix'][2][n] else
                    'counts' if local[n] and not exact[n] else '')
            if not name or name in examples or k-j<2 or len(row['x'])>24 or j>20 or len(pred)>12:
                continue
            fields = dict(source=bank[row['x']],prefix=bank[row['y'][:j]],reference=reference,
                          writer=pred,counts=runs['local'][0][n])
            for label,data in fields.items():
                png(directory,name+'_'+label,data)
                widths[name+'_'+label]=len(data)
            standalone,ended = net.generate_phrase(ink[row['x']],c,ink[row['y'][:j]],ink[qids])
            require(np.array_equal(np.packbits(standalone.numpy(),axis=1),pred) and ended==stop[n],
                    'Standalone CPU generation differs from archived output')
            examples[name]=dict(row=i,site=n,boundary=c,source=row['source'],prefix=row['target'][:j],
                reference=row['target'][j:k],chapter=row['g'],stopped=bool(ended))
        require(first_wrong+ended_short+grew_long+different_later+capped+exact.sum()==len(sites),'Failure partition')
        report[split]=dict(exact=int(exact.sum()),wrong_first_glyph=first_wrong,
            exact_emitted_prefix_but_shorter=ended_short,correct_phrase_followed_by_extra_ink=grew_long,
            correct_first_then_other_ink=different_later,exact_cells_without_stop=capped,
            drawable_from_reference_codes=drawable,exact_despite_noncanonical_input_codes=not_drawable_exact,
            local_only=int((local&~exact).sum()),writer_only=int((exact&~local).sum()),
            distinct_physical_phrases=len(frequencies),most_frequent_output_count=frequencies.most_common(1)[0][1])
        chart[split]=measured['arms']
    require(len(examples)==2,'Missing qualifying illustration')
    table=[['Generation at a supplied site','Discovery (5,376)','Replication (5,413)']]
    for arm,label in [('original','Original sentence editor'),('prefix','Modern prefix only'),('source','Classical source and modern prefix'),('local','Local whole-phrase counts')]:
        table.append([label,*[str(chart[s][arm]['exact']) for s in chart]])
    scores=[[round(100*chart[s][a]['first_glyph_exact']/chart[s][a]['sites'],2) for s in chart]
            for a in ('source','local')]
    write_new(directory/'figure.json',dict(table=table,accuracy_percent=scores,examples=examples,widths=widths))
    write_new(directory/'audit.json',report)
    print(examples,report)


if __name__=='__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN/'report')
    run(parser.parse_args().output)
