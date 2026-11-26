"""Fit the observed distribution of optical spans and freeze the training counts."""
from collections import Counter
from pathlib import Path
from time import monotonic
import gzip
import json
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer
from scripts.check_optical_span_feedback import segment
from scripts.reserve_optical_span_rollout import PUB, PROTOCOL, ROOT, dump, read, sha
from scripts.reserve_target_glyph_book_panel import library
from scripts.train_readout_objective import TEACHERS


def fit(rows, t, folder):
    tokenizer = MarianTokenizer.from_pretrained(folder, local_files_only=True)
    counts = Counter()
    discarded = 0
    for j in range(0,len(rows),256):
        group = rows[j:j+256]
        batch = tokenizer(text_target=[r["text"][t] for r in group]).input_ids
        for row, pieces in zip(group,batch,strict=True):
            spans = segment(tokenizer,row,t,pieces)
            for i,(cells, piece) in enumerate(zip(spans,pieces[:-1],strict=True)):
                if piece in (tokenizer.unk_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id):
                    discarded += 1
                else:
                    counts[int(i>0),piece,cells] += 1
    return [[phase,k,list(s),n] for (phase,k,s),n in sorted(counts.items())],discarded


def main():
    rows = library("train")
    binding = dict(panel=sha(PUB/"panel.json"),protocol=sha(PROTOCOL),script=sha(Path(__file__)),
                   training=sha(ROOT/"references/evidence/visual_library_v1.json"))
    for t,lang in enumerate(("en","zh")):
        path = PUB/(lang+"-span-counts.json.gz")
        rec = PUB/(lang+"-fit.json")
        if rec.exists():
            assert sha(path)==read(rec)["weights"] and binding==read(rec)["binding"]
            continue
        start = monotonic()
        name,rev = TEACHERS[t]
        folder = snapshot_download("Helsinki-NLP/opus-mt-"+name,revision=rev,local_files_only=True)
        entries,discarded = fit(rows,t,folder)
        path.write_bytes(gzip.compress(json.dumps(entries,separators=(",",":")).encode(),mtime=0))
        dump(rec,dict(binding=binding,weights=sha(path),revision=rev,model="Helsinki-NLP/opus-mt-"+name,
                      pages=len(rows),retained=sum(x[3] for x in entries),discarded=discarded,
                      types=len({tuple(x[2]) for x in entries}),seconds=monotonic()-start))
        print(lang,read(rec),flush=True)


if __name__=="__main__":
    main()
