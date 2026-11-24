"""Build the bank from training ink and score all frozen source positions."""

from time import monotonic
import numpy as np
import torch

from ilm.visual_lm.attested_gloss import AttestedGlosses
from ilm.visual_lm.gloss_focus import cell_features,encoding_windows
from scripts.answer_visual_gloss import RELEASE
from scripts.prepare_asked_gloss import PRIOR,PUBLIC,ROOT,WORK,read,dump,sha
from scripts.train_asked_gloss import packed_batch
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face,FACES
from scripts.measure_writer_energy import snapshot, consumed
from scripts.write_visual_gloss import restore


def build(writer):
    reg=read(ROOT/"references/evidence/visual_word_senses_20261124/split.json")
    if sha(PRIOR/"train.npz")!=reg["data"]["train"]["inputs"] or sha(PRIOR/"pixels.npy")!=reg["pixels"]:
        raise ValueError("Training image binding changed")
    raw=np.load(PRIOR/"pixels.npy")[0]
    bits=np.unpackbits(raw,axis=-1)[:,writer.landmarks.cpu().numpy()]
    codes=(bits*(2**np.arange(bits.shape[1]))).sum(-1)
    codes[0]=writer.stop
    with np.load(PRIOR/"train.npz") as source:
        bank=np.unique(codes[source["y"]],axis=0).astype(np.uint8)
    f=RELEASE/"attested.npz"
    np.savez_compressed(f,codes=bank)
    dump(RELEASE/"attested.json",dict(sha256=sha(f),training=reg["data"]["train"]["inputs"],glosses=len(bank)))


@torch.inference_mode()
def run():
    started=monotonic()
    writer=restore(device="cuda")
    build(writer)
    bank=AttestedGlosses(RELEASE/"attested.npz",writer.stop,"cuda")
    original=read(PUBLIC/"result.json")
    ordered=read(PUBLIC/"order_result.json")
    assert sha(PUBLIC/"predictions.json")==original["predictions"] and sha(PUBLIC/"order_positions.npz")==ordered["positions"]
    assert sha(PUBLIC/"emitted.npz")==original["ink"]
    selections=np.load(PUBLIC/"order_positions.npz")["ordered_contrast"]
    rows=read(PUBLIC/"predictions.json")
    energy=snapshot();torch.cuda.reset_peak_memory_stats()
    face=Face(FACES[0]); chars={face.cell(c):c for c in "abcdefghijklmnopqrstuvwxyz-"}
    changed=[];free_count=0;total=0;paired={};repaired=lost=0
    archive=np.load(PUBLIC/"emitted.npz")
    offsets,ink,stopped=(archive[k] for k in ("offsets","ink","stopped"))
    for j,r in enumerate(rows):
        pred=list(r["all_words"])
        raw,lengths=packed_batch([r["page"]],face,"cuda")
        ctx=encoding_windows(writer,cell_features(raw))[0]
        for i,ix in enumerate(r["indices"]):
            a,b=offsets[ix:ix+2]
            bits=np.unpackbits(ink[a:b],axis=-1)[:,writer.landmarks.cpu().numpy()]
            code=tuple((bits*(2**np.arange(bits.shape[1]))).sum(-1).tolist())
            total+=1
            if code in bank.members and stopped[ix]:
                free_count+=1
                continue
            chosen=bank.best(writer,ctx[i]); pic=np.packbits(writer.draw(chosen).cpu().numpy(),axis=-1)
            new="".join(chars[c.tobytes()] for c in pic)
            for k,tr in enumerate(r["focus"]):
                if i==tr:
                    repaired+=int(new in r["gloss"][k] and pred[i] not in r["gloss"][k])
                    lost+=int(new not in r["gloss"][k] and pred[i] in r["gloss"][k])
            changed.append(dict(row=j,cell=i,old=pred[i],new=new,ink=pic.tobytes().hex()))
            pred[i]=new
        for book in ("combined",r["book"]):
            scores=paired.setdefault(book,dict(pages=0,all_eight=0,free_all_eight=0,known_at_focus=0,free_at_focus=0))
            scores["pages"]+=1
            for col, words in [("all_eight",pred),("free_all_eight",r["all_words"])]:
                lexical=all(words[i] in r["gloss"][k] for k,i in enumerate(r["focus"]))
                scores[col]+=int(lexical and np.all(selections[j,:4]==r["focus"]))
                scores["known_at_focus" if col=="all_eight" else "free_at_focus"]+=int(lexical)
        if monotonic()-started>900 or torch.cuda.max_memory_allocated()>8*2**30:
            raise RuntimeError("Resource limit")
    archive.close()
    dump(PUBLIC/"attested_result.json",dict(unchanged=free_count,positions=total,repaired_true_cells=repaired,
        lost_true_cells=lost,books=paired,changes=changed,seconds=monotonic()-started,energy=consumed(energy,snapshot()),
        peak_bytes=torch.cuda.max_memory_allocated(),bank=sha(RELEASE/"attested.npz")))
    print(total,len(changed),paired["combined"])


if __name__=="__main__":
    with gpu_owner():run()
