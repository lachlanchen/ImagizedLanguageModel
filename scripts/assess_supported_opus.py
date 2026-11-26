"""Test the finished writers on disjoint OPUS inputs and archive their raw responses."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import torch

from scripts.evaluate_optical_span_rollout import blind_votes,configure_cuda_cli,gpu_owner,summarize
from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.reserve_independent_optical_actions import RUN as CORPUS
from scripts.reserve_source_supported_writing import BASE,PUB,RUN,ROOT,read,sha,dump
from scripts.write_source_supported import SupportedWriter
from scripts.write_neural_glyphs import save_ink,grid_bits

ARCHIVE=ROOT/"publication/ilm-nature/evidence/supported_opus_outputs.zip"
ARMS=("parent","aligned","shuffled")


def reserve():
    path=RUN/"opus.json"
    if path.exists() and (PUB/"opus-panel.json").exists():
        assert sha(path)==read(PUB/"opus-panel.json")["panel"]
        return read(path)
    test=ROOT/"artifacts/bilingual_ink_v1/held.json"
    assert sha(test)==read(ROOT/"references/evidence/bilingual_ink_v1/corpus.json")["files"]["held.json"]
    test_rows=read(test)
    names={int(k):v for k,v in read(BASE/"readback.json").items()}
    names[0]=""
    norm=lambda s:"".join(c for c in s.casefold() if c.isalnum())
    excluded=set()
    train=np.load(CORPUS/"cells.npy",mmap_mode="r")
    for t in (0,1):
        normal={norm(r["text"][t]) for r in test_rows}
        seen=set()
        exact={tuple(r["cells"][t]) for r in test_rows}
        seen_exact=set()
        for raw in train[:,t]:
            ids=tuple(raw[:np.count_nonzero(raw)].tolist())
            if ids in exact:
                seen_exact.add(ids)
            key=norm("".join(names[c] for c in ids))
            if key in normal:
                seen.add(key)
        excluded.update(i for i,r in enumerate(test_rows) if tuple(r["cells"][t]) in seen_exact or norm(r["text"][t]) in seen)
    rows=[r for i,r in enumerate(test_rows) if i not in excluded]
    assert rows
    dump(path,rows)
    dump(PUB/"opus-panel.json",dict(panel=sha(path),test=sha(test),train_pairs=len(train),held=len(test_rows),
             excluded=sorted(excluded),pairs=len(rows),protocol=sha(ROOT/"references/supported_writing_opus_transfer.md")))
    return rows


def requests(pairs):
    return [dict(t=t,source=r["text"][1-t],ref=r["text"][t],doc="OPUS test",id=i,
                 foil=pairs[(i+len(pairs)//2)%len(pairs)]["text"][t]) for i,r in enumerate(pairs) for t in (0,1)]


def generate():
    pairs=reserve()
    pal=np.load(BASE/"palette.npy")
    bits=np.unpackbits(pal,axis=1).reshape(-1,32,32).astype(bool)
    names={bytes(pal[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    rows=requests(pairs)
    answers=[{} for _ in rows]
    tmp=ARCHIVE.with_suffix(".tmp")
    receipt=PUB/"opus-generation.json"
    if receipt.exists():
        assert read(receipt)["archive"]==sha(ARCHIVE)
        return
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        images=[]
        for i,r in enumerate(rows):
            ids=pairs[r["id"]]["cells"][1-r["t"]]
            buf=BytesIO()
            save_ink(bits[ids],buf,image_format="PNG")
            blob=buf.getvalue()
            z.writestr(f"{i:04d}/source.png",blob)
            images.append(grid_bits(BytesIO(blob),len(ids)))
        for arm in ARMS:
            writer=SupportedWriter(arm,"cpu")
            for t in (0,1):
                order=list(range(t,len(rows),2))
                for j in range(0,len(order),16):
                    block=order[j:j+16]
                    for i,(ink,stop) in zip(block,writer.write([images[i] for i in block],t),strict=True):
                        buf=BytesIO()
                        save_ink(ink,buf,image_format="PNG")
                        blob=buf.getvalue()
                        z.writestr(f"{i:04d}/{arm}.png",blob)
                        text="".join(names[bytes(v)] for v in grid_bits_out(blob,len(ink)))
                        answers[i][arm]=dict(text=text,stop=stop,cells=len(ink),sha256=sha256(blob).hexdigest())
        z.writestr("records.json",json.dumps([dict(**r,answers=a) for r,a in zip(rows,answers,strict=True)],ensure_ascii=False))
    tmp.replace(ARCHIVE)
    dump(receipt,dict(archive=sha(ARCHIVE),panel=sha(RUN/"opus.json"),requests=len(rows),images=len(rows)*len(ARMS)))


def grade():
    assert sha(ARCHIVE)==read(PUB/"opus-generation.json")["archive"]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
    answers=[dict(**r["answers"],reference=dict(text=r["ref"],stop=True),foil=dict(text=r["foil"],stop=True)) for r in rows]
    votes=blind_votes(rows,answers)
    grades=summarize(rows,answers,votes)
    dump(PUB/"opus-assessment.json",dict(archive=sha(ARCHIVE),votes=votes,primary=grades))
    print(grades["total"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("generate","assess"))
    args=cli.parse_args()
    torch.set_num_threads(2)
    if args.stage=="generate":
        generate()
    elif not (PUB/"opus-assessment.json").exists():
        configure_cuda_cli()
        with gpu_owner():
            grade()
