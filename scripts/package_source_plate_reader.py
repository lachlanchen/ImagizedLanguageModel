"""Package qualified weights and fixed-order source examples from local evidence."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_source_plate_reader import DATA, MODELS, OUT as READER, ROOT
from scripts.fit_source_plate_spans import OUT
from scripts.train_visual_extractive_reader import save_page

EVIDENCE = ROOT / 'references/evidence/historical_plates_20261118'


def export():
    results = load(OUT / 'result.json')
    require(results['qualified'], 'Historical path did not meet the fixed gate')
    paths = {'reader': READER / 'source_plate.safetensors', 'span': OUT / 'stage_spans.safetensors'}
    require([sha(p) for p in paths.values()] == [results['source_reader_sha256'], results['weights_sha256']],
            'Component weights changed')
    weights = {p + '.' + k: v for p, file in paths.items() for k, v in load_file(str(file)).items()}
    dest = MODELS / 'historical_reader.safetensors'
    require(not dest.exists(), 'Preserve the existing package')
    save_file(weights, str(dest), metadata=dict(format='ilm-historical-spans-v1',
              base_sha256=sha(MODELS / 'reader_writer.safetensors'), writer_sha256=sha(MODELS / 'distilled_writer.safetensors')))
    EVIDENCE.mkdir()
    from scripts.fit_source_plate_coverage import OUT as COVERAGE
    for name, file in [('spans', OUT), ('coverage', COVERAGE), ('reader', READER),
                       ('base', ROOT / 'artifacts/source_plate_baseline_v1')]:
        (EVIDENCE / (name + '.json')).write_bytes((file / 'result.json').read_bytes())
    (EVIDENCE / 'corpus.json').write_bytes((DATA / 'report.json').read_bytes())
    px = np.load(DATA / 'pixels.npy')
    with np.load(OUT / 'intact.npz') as z:
        outputs = z['packed']
    examples = MODELS / 'examples'
    examples.mkdir(exist_ok=True)
    manifest = {}
    demo = load(DATA / 'yan/source.json')
    for name, split, index in [('yan', 'train', demo['index']), ('ma', 'development', 2), ('miss', 'development', 1216)]:
        with np.load(DATA / (split + '.npz')) as z:
            ctx, rows, n = z['context'][index], z['row'][index], int(z['length'][index])
            gold = px[z['target'][index, :n]]
        card = load(DATA / (split + '_cards.json'))[index // 16]
        grid = np.unpackbits(px[ctx[24:44]], axis=1).reshape(4, 5, 32, 32).transpose(0, 2, 1, 3).reshape(128, 160)
        Image.fromarray(255 - 255 * grid).save(examples / (name + '_plate.png'))
        save_page(examples / (name + '_question.png'), px[ctx[:24]])
        selected = [card[i] for i in rows[:n]]
        manifest[name] = dict(split=split, index=index,
                              source=[{k:v for k,v in row.items() if k in ('char','stage','label','entry','raster','svg')}
                                      for row in card], requested=[r['label'] for r in selected])
        if split == 'development':
            actual = outputs[index, :n]
            manifest[name]['emitted'] = [next(row['label'] for row in card if np.array_equal(px[row['tile']], glyph))
                                         for glyph in actual]
            save_page(EVIDENCE / (name + '_answer.png'), actual)
        if name == 'miss':
            save_page(EVIDENCE / 'miss_reference.png', gold)
    write_new(examples / 'source.json', manifest)
    write_new(EVIDENCE / 'package.json', dict(weights_sha256=sha(dest), examples=manifest,
              corpus_report_note='The initial report predates exporting the training-family Yan example at card 302.'))

if __name__ == '__main__':
    export()
