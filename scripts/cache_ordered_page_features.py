"""Encode preselected visual questions once, binding source pixels and contextual features."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import fcntl
import json
import os
from time import monotonic

import torch
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from scripts.fit_font_invariant_reader import WORK as READER, ROOT, sha
from scripts.ordered_image_corpus import corpus

WORK = ROOT/"artifacts/ordered_page_writer_v1"
PUBLIC = ROOT/"references/evidence/ordered_page_writer_20261119"


def fingerprint(rows):
    h = sha256()
    for r in rows:
        for v in (r.task.encode(),r.case.encode(),r.passage,r.question,r.target):
            h.update(len(v).to_bytes(8,"big"))
            h.update(v)
    return h.hexdigest()


def prepare():
    WORK.mkdir()
    start = monotonic()
    fit = json.loads((READER/"fit.json").read_text())
    for name,val in fit["weights"].items():
        assert sha(READER/name)==val
    reader = FontInvariantReader.restore(READER,"cuda:0")
    info = {}
    for split in ("train","development"):
        rows = corpus(split)
        features = contextualize(reader,[(r.passage,r.question) for r in rows],batch_size=18)
        out = WORK/(split+".pt")
        torch.save(features,out)
        info[split] = dict(rows=dict(Counter(r.task for r in rows)), corpus_sha256=fingerprint(rows),
                           features_sha256=sha(out))
        print(split,info[split],flush=True)
    info.update(seconds=monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
                fit_sha256=sha(READER/"fit.json"),
                sources={str(p.relative_to(ROOT)):sha(p) for p in
                         (Path(__file__),ROOT/"scripts/ordered_image_corpus.py",
                          ROOT/"ilm/visual_lm/ordered_page_features.py")})
    assert info["seconds"]<1200 and info["peak_bytes"]<16*2**30
    (WORK/"features.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    assert os.environ["CUDA_VISIBLE_DEVICES"]=="0"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        prepare()
