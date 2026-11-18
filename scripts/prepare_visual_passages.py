"""Index the original physical cells for each distinct modern training passage."""
import hashlib
import json
from pathlib import Path
import os
import tempfile

import numpy as np

from scripts.diagnose_visual_context_gradients import load, require, sha
from scripts.prepare_visual_sequence_corpus import CMRC, FONT, Rasterizer, context_hash
from scripts.train_continuous_image_language import DATA, ROOT, REG

MANIFEST = ROOT / 'data/visual_passages_v1/manifest.json'


def main():
    require(not MANIFEST.exists(), 'Preserve existing passage index')
    report = load(DATA/'report.json')
    require(sha(DATA/'report.json') == load(REG)['input_sha256'][str((DATA/'report.json').relative_to(ROOT))], 'Training source changed')
    source = CMRC/'cmrc2018_train.json'
    require(sha(source) == report['source_sha256'][str(source.relative_to(ROOT))] and
            sha(FONT) == report['font_sha256'], 'Training text or rasterizer font changed')
    contexts = {q['id']: p['context'] for a in load(source)['data']
                for p in a['paragraphs'] for q in p['qas']}
    renderer = Rasterizer()
    prefix = '資料：'
    seen, seen_pixels, passages = set(), set(), []
    record_index = book_targets = answer_targets = prompt_positions = 0
    for shard in report['shards']:
        if shard['split'] != 'train':
            continue
        paths = {}
        for suffix in ('_sources.json','_pixels.npz'):
            name, = [n for n in shard['files'] if n.endswith(suffix)]
            path = DATA/name
            require(sha(path) == shard['files'][name], 'Original shard changed')
            paths[suffix] = path
        origins = load(paths['_sources.json'])
        with np.load(paths['_pixels.npz'], allow_pickle=False) as arrays:
            ink = arrays['ink']
            offsets, prompts = arrays['record_glyph_offsets'], arrays['prompt_glyphs']
            require(len(origins) == len(prompts) == len(offsets)-1, 'Source alignment')
            for local, (origin, count) in enumerate(zip(origins,prompts)):
                index = record_index
                record_index += 1
                first, last = offsets[local:local+2]
                if origin['kind'] == 'book':
                    require(count == 0, 'Book prompt boundary')
                    book_targets += int(last-first-1)
                    continue
                require(origin['kind'] == 'qa' and count > 0, 'Unexpected source role')
                answer_targets += int(last-first-count)
                prompt_positions += int(count)
                group = origin['group']
                if group in seen:
                    continue
                text = contexts[origin['id']]
                require(context_hash(text) == group, 'Training group mismatch')
                seen.add(group)
                end = len(prefix)+len(text)
                require(1 < len(text) and end < count, 'Incomplete passage')
                packed = ink[first+len(prefix):first+end].tobytes()
                actual = hashlib.sha256(packed).hexdigest()
                require(b''.join(renderer.tile(c) for c in text) == packed, 'Original passage pixels differ')
                if actual in seen_pixels:
                    continue
                seen_pixels.add(actual)
                passages.append(dict(corpus_index=index, start=len(prefix), length=len(text),
                                     ink_sha256=actual, group=group))
    require((record_index, len(seen), book_targets, answer_targets, prompt_positions) ==
            (13785, 2127, 2400020, 171457, 4678984), 'Original exposure counts differ')
    result = dict(schema='visual-training-passages-v1', corpus_sha256=sha(DATA/'report.json'),
                  source_sha256=sha(source), font_sha256=sha(FONT), passages=passages,
                  original_records=record_index, original_distinct_qa_groups=len(seen),
                  original_book_targets=book_targets, original_qa_answer_targets=answer_targets,
                  original_qa_prompt_positions=prompt_positions,
                  unique_images_in_sequence=sum(r['length'] for r in passages),
                  next_image_targets=sum(r['length']-1 for r in passages))
    MANIFEST.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=MANIFEST.parent) as tmp:
        staging = Path(tmp)/MANIFEST.name
        staging.write_text(json.dumps(result,indent=2)+'\n')
        os.link(staging,MANIFEST)
    print(json.dumps({k:v for k,v in result.items() if k!='passages'}))


if __name__ == '__main__':
    main()
