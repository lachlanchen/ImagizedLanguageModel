"""Ask the deployed visual writer to distinguish professional judgments on Chinese news."""
from collections import Counter
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.calibrate_translation_graders import PUB, ROOT, panel, base
from scripts.measure_paired_visual_writer import CoveredWriter, restore, MODEL, pack, pictured_input, path_scores, png
from scripts.audit_glyph_draft_editor import unpack

ARCHIVE = ROOT / 'publication/ilm-nature/evidence/wmt_news_visual_answers.zip'


@torch.inference_mode()
def main():
    cases, selection = panel()
    assert base.read(PUB / 'panel.json') == json.loads(json.dumps(selection))
    if (PUB / 'writer.json').exists():
        raise SystemExit('Preserved WMT visual writer assay exists')
    torch.set_num_threads(2)
    net = restore('broad', 'cpu')
    face = base.Ink(net.palette)
    inv = list(map(set, base.read(base.BASE / 'inventories.json')))
    names = base.read(base.BASE / 'readback.json')
    def encode(text, language):
        if not 1 <= len(text) <= 96:
            return 'length', []
        row = face.encode(text)
        if face.bad.intersection(text) or not set(row) <= inv[language]:
            return 'repertoire', []
        return 'represented', row

    engine = CoveredWriter(0)
    source_counts, candidate_counts = Counter(), Counter()
    rows, comparisons = [], []
    temp = ARCHIVE.with_suffix('.partial')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        for i, (source, good, major) in enumerate(cases):
            status, cells = encode(source, 1)
            source_counts[status] += 1
            if not cells:
                continue
            physical = net.palette[cells]
            assert net.addresses(physical) == cells
            draft = engine.translate(physical)[0]
            pixels, stopped = net.translate([physical], 0, [draft])[0]
            out = net.addresses(pixels)
            record = dict(index=i, stop=bool(stopped), length=len(out), cells=len(cells),
                          readback=''.join(names[str(k)] for k in out))
            for k, img in [('source', physical), ('answer', pixels)]:
                blob = png(img)
                z.writestr(f'{i}/{k}.png', blob)
                record[k] = sha256(blob).hexdigest()
            targets = [encode(s, 0) for s in (good, major)]
            candidate_counts['/'.join(status for status, _ in targets)] += 1
            record['exact_reference'] = bool(stopped and out == targets[0][1] and targets[0][0]=='represented')
            if all(status == 'represented' for status, _ in targets):
                comparisons.append((i, cells, net.addresses(draft) or [0], [v+[1] for _, v in targets]))
            rows.append(record)
        z.writestr('records.json', json.dumps(rows, ensure_ascii=False))
    scores = []
    assert len(comparisons) >= 2, 'Need two distinct sources to run the exchange'
    for j, (i, _, _, options) in enumerate(comparisons):
        item = dict(index=i)
        for key, context in [('own', j), ('exchange', (j+1) % len(comparisons))]:
            other, source_cells, draft_cells, _ = comparisons[context]
            src = pack([source_cells]*2, 'cpu')
            field = (255*net.marker(net.features, src).sigmoid()*src.ne(0).unsqueeze(-1)).round().byte()
            ll, width = path_scores(net.model, (net.features, net.inv, net.local), pictured_input(src, field),
                                    pack([draft_cells]*2, 'cpu'), pack(options, 'cpu'), 0)
            item[key] = (ll / width).tolist()
            item[key+'_source'] = other
        scores.append(item)
    replay = []
    with ZipFile(temp) as z:
        for row in rows:
            i = row['index']
            blob = z.read(f'{i}/source.png')
            assert sha256(blob).hexdigest() == row['source']
            assert sha256(z.read(f'{i}/answer.png')).hexdigest() == row['answer']
            if len(replay)<8:
                out, stop = net.translate([unpack(blob, row['cells'])], 0)[0]
                replay.append(sha256(png(out)).hexdigest() == row['answer'] and bool(stop)==row['stop'])
    temp.replace(ARCHIVE)
    margins = {key: np.array([r[key][0]-r[key][1] for r in scores]) for key in ('own','exchange')}
    correct = {k: v>0 for k,v in margins.items()}
    receipt = dict(panel=base.sha(PUB/'panel.json'), fit=base.sha(MODEL/'broad/fit.json'), archive=base.sha(ARCHIVE),
                   scope_ratios=dict(all=len(cases), source=dict(source_counts), candidates=dict(candidate_counts)),
                   scores=scores, free=len(rows), stopped=sum(r['stop'] for r in rows),
                   exact_good=sum(r['exact_reference'] for r in rows), replay=replay, rankable=len(scores),
                   human_preferred={k:int(v.sum()) for k,v in correct.items()},
                   ties={k:int((v==0).sum()) for k,v in margins.items()},
                   gained=int((correct['own'] & ~correct['exchange']).sum()),
                   lost=int((~correct['own'] & correct['exchange']).sum()))
    base.dump(PUB/'writer.json',receipt)
    print({k:v for k,v in receipt.items() if k!='scores'}, flush=True)


if __name__=='__main__':
    main()
