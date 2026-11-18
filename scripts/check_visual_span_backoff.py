"""Compare the runtime selector with its archived test emissions and the same text-LLM sample."""
from pathlib import Path
import numpy as np

from ilm.visual_lm.span_backoff import SpanBackoff
from scripts.audit_traditional_passage_training import archived
from scripts.benchmark_visual_inference_qwen import LOCATION, REG as QWEN
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.fit_visual_span_backoff import PATH
from scripts.train_visual_inference_writer import PUBLIC,WORK,Corpus,RESTORE,load,memory_bank,sha,write_json


def audit():
    backoff=load(PUBLIC/"backoff.json");primary=load(PUBLIC/"result.json")
    for n,v in backoff["files"].items():assert sha(PATH/n)==v
    net=SpanBackoff(PATH);coeff=np.load(PATH/"coefficients.npy")
    bank=memory_bank();ds=Corpus(bank,"development")
    read=lambda n:archived(WORK/n,primary["archives"][n])
    ix=np.load(WORK/"official_cloze_rows.npy")
    neural,ends=read("paired/official_restoration_intact.npz")
    masked,_=read("official_cloze_requests.npz");target,_=read("official_cloze_targets.npz")
    saved,stops=archived(PATH/"official.npz",backoff["files"]["official.npz"])
    for i,q,n,e,out,end in zip(ix,masked,neural,ends,saved,stops,strict=True):
        p=b"".join(bank.tiles[t] for t in ds.p[i])
        assert net.choose(p,q[128*len(RESTORE):],ds.mask_tile(),n,e,coeff)==(out,end)
    ids=load(QWEN)["official_rows"]
    positions={v:k for k,v in enumerate(ix.tolist())}
    chosen=np.array([positions[i] for i in ids])
    hybrid=[stops[i] and saved[i]==target[i] for i in chosen]
    import json
    from scripts.benchmark_visual_inference_qwen import cases
    ledger=[json.loads(l) for l in (LOCATION/"answers.jsonl").read_text().splitlines()]
    teacher=[s["stop"]=="stop" and s["text"]==r[4] for r,s in zip(cases(),ledger,strict=True) if r[1]=="restoration"]
    score=paired_interval(hybrid,teacher,np.arange(len(chosen))//3)
    write_json(PUBLIC/"backoff_audit.json",dict(verified_official=len(ix),qwen_sample=len(ids),
        hybrid=int(sum(hybrid)),qwen=sum(teacher),hybrid_minus_qwen=score,
        backoff_sha256=sha(PUBLIC/"backoff.json"),source_sha256=sha(Path(__file__))))
    print(len(ix),sum(hybrid),sum(teacher),score)


if __name__=="__main__":
    audit()
