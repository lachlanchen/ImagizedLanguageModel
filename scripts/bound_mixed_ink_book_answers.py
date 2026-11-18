"""Count next-sentence targets outside the output support of the mixed reader."""
import re
from collections import Counter

import numpy as np

from scripts.diagnose_visual_context_gradients import sha, write_new
from scripts.prepare_visual_sequence_corpus import BOOKS, NAMESPACE, Rasterizer, text_hash
from scripts.train_visual_extractive_reader import gold_occurrences


def main():
    import json
    records = [json.loads(line) for line in BOOKS.read_text().splitlines()]
    titles = sorted({r['source_title'] for r in records}, key=lambda t:text_hash(NAMESPACE+'\0'+t))[:2]
    render = Rasterizer()
    results = {}
    for title in titles:
        counts = Counter()
        for record in records:
            if record['source_title']!=title:
                continue
            # Stay inside the given physical source record. Keep sentence-final punctuation.
            segments = [s.strip() for s in re.split(r'(?<=[。！？])|\n+',record['text']) if s.strip()]
            previous = None
            for i,s in enumerate(segments):
                try:
                    current = np.frombuffer(b''.join(render.tile(c) for c in s),np.uint8).reshape(-1,128)
                except ValueError:
                    current = None
                if i:
                    counts['pairs'] += 1
                    if current is None or previous is None:
                        counts['unrenderable'] += 1
                    else:
                        counts['pictured_pairs'] += 1
                        short = len(current)<=8
                        contiguous = bool(gold_occurrences(previous,current))
                        counts['within_eight'] += short
                        counts['contiguous'] += contiguous
                        counts['either_possible'] += short or contiguous
                        counts['structurally_impossible'] += not (short or contiguous)
                        if not (short or contiguous):
                            known = set(map(bytes,previous))
                            counts['impossible_with_absent_ink'] += any(bytes(cell) not in known for cell in current)
                previous = current
        results[title] = dict(counts)
    out = BOOKS.parents[2]/'artifacts/mixed_ink_reader_v1/book_support.json'
    write_new(out,dict(source_sha256=sha(BOOKS),titles=titles,results=results,
                       generative_maximum=8,unit='adjacent segments within each original record',
                       bound='Either target length <=8 or target is a contiguous physical span of the source.'))
    print(results)


if __name__=='__main__':
    main()
