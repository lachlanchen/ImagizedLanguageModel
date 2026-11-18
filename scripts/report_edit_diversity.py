"""Export measured values and original generated pixels for Figure 50."""
from zipfile import ZipFile
import shutil

import numpy as np

from scripts.compare_edit_diversity import ARMS
from scripts.prepare_edit_diversity import COHORTS, ROOT, WORK, read_split
from scripts.train_visual_edit_translation import load, require, sha, write_new
from scripts.train_visual_extractive_reader import save_page


def report():
    dest = ROOT/'references/evidence/edit_diversity_20261118'
    dest.mkdir()
    for path in sorted(WORK.rglob('*.json')):
        if path.name != 'rows.json':
            shutil.copy2(path, dest / '_'.join(path.relative_to(WORK).parts))
    table = [['Training and decoder', 'Discovery', 'Replication']]
    for label, arm, decoder in [('48,000 repeated', 'repeat', 'translation'), ('192,000 distinct', 'diverse', 'translation'),
                                ('Local counts, smaller corpus', 'repeat', 'local'),
                                ('Local counts, larger corpus', 'diverse', 'local')]:
        values = [load(WORK/arm/name/'report.json')['outcome'][decoder] for name in COHORTS]
        table.append([label, *[f"{v['exact']}   /   {100*v['cer']:.2f}%" for v in values]])
    teacher = [load(WORK/'diverse'/f'{s}_teacher.json') for s in COHORTS]
    accuracy = [[round(100*v[field]/v['source_draw_positions'], 2) for v in teacher]
                for field in ('emitted_glyph_correct', 'true_code_render_correct')]
    widths, chosen = {}, {}
    prep = load(WORK/'preparation.json')
    require(sha(WORK/'pixels.npy') == prep['pixels_sha256'], 'Bitmap bank changed')
    tiles = np.load(WORK/'pixels.npy')
    records = list(read_split(COHORTS[0], prep))
    rows = load(WORK/'diverse/discovery/rows.json')
    examples = [('success', next(r for r in rows if r['translation']['exact'] and r['hard'])),
                ('failure', next(r for r in rows if not r['translation']['exact']
                                 and max(r['length'], r['translation']['cells'], r['copy']['cells']) <= 14))]
    for label, row in examples:
        i = row['index']
        c = records[i]
        chosen[label] = dict(index=i, source=c['source'], target=c['target'], chapter=c['g'])
        images = {'source': tiles[c['x']], 'reference': tiles[c['y']]}
        for arm in ARMS:
            with ZipFile(WORK/arm/'discovery/streams.zip') as z:
                images[arm] = np.frombuffer(z.read(f'translation/{i}.bin'), np.uint8).reshape(-1,128)
        for field, img in images.items():
            require(0 < len(img) <= 32, 'Unexpected example width')
            key = f'{label}_{field}'
            widths[key] = len(img)
            save_page(dest / (key+'.png'), img)
    write_new(dest/'figure.json', dict(table=table, drawing_accuracy_percent=accuracy, widths=widths, examples=chosen))


if __name__ == '__main__':
    report()
