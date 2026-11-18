"""Recount independent physical emissions and measure the whole-window copying baseline."""
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

from scripts.audit_traditional_passage_training import archived
from scripts.prepare_paired_stance_images import DEST, Rasterizer
from scripts.train_paired_passage import unpack_sequences
from scripts.train_paired_stance_writer import PUBLIC, SPLIT, WORK, load, sha, write_json


def figure_prefix(value,name):
    if len(value)<384:
        raise ValueError("Example lacks three glyphs")
    packed=np.frombuffer(value[:384],np.uint8).reshape(3,128)
    bits=np.unpackbits(packed,axis=1).reshape(3,32,32)
    Image.fromarray(255*(1-np.concatenate(bits,axis=1))).save(PUBLIC/(name+".png"))


def audit():
    report=load(PUBLIC/"result.json")
    r=Rasterizer()
    signs={b"".join(r.tile(c) for c in text) for text in ("肯定。","否定。")}
    splits=load(SPLIT)
    output={}
    for key,orig in (("validation","train"),("official","development")):
        path=DEST/orig
        rows=[2*i+delta for i in splits[key] for delta in (0,1)]
        bank=np.load(path/"pixels.npy")
        targets,passages,questions=[[bank[s].tobytes() for s in unpack_sequences(path/(k+".npz"))] for k in ("answers","passages","questions")]
        baseline=Counter()
        details={}
        for arm in ("matched","reassigned","parent"):
            name=arm+"/"+key+"_full.npz"
            imgs,ends=archived(WORK/name,report["archives"][name])
            erased_name=arm+"/"+key+"_question_erased.npz"
            erased,_=archived(WORK/erased_name,report["archives"][erased_name])
            counts=Counter()
            exact_sign=[]; complete=[]; body=[]
            for j,i in enumerate(rows):
                pred=imgs[j];ref=targets[i];p=passages[i]
                correct=pred[:384]==ref[:384]
                exact=bool(ends[j]) and pred==ref
                quote=bool(ends[j]) and pred[:384] in signs and pred[384:]==ref[384:]
                entire=p==ref[384:]
                exact_sign.append(correct);complete.append(exact);body.append(quote)
                counts["position_same_when_question_removed"]+=pred[:384]==erased[j][:384]
                counts["full_output_same_when_question_removed"]+=pred==erased[j]
                counts["full_exact_with_whole_window"]+=exact and entire
                counts["full_exact_with_proper_subspan"]+=exact and not entire
                counts["proper_subspan_body_exact"]+=quote and not entire
                counts["correct_body_wrong_position"]+=quote and not correct
                counts["correct_position_wrong_body_or_stop"]+=correct and not quote
                counts["body_equals_input_window"]+=bool(ends[j]) and pred[:384] in signs and pred[384:]==p
                if arm=="matched":
                    baseline["human_quote_is_full_window"]+=entire
                    baseline["whole_window_correct_both"]+=i%2==1 and entire and targets[i-1][384:]==passages[i-1]
                    instruction="这段材料对下面的问题持什么立场？写明立场，再引一句材料原话。"
                    seen={r.tile(c) for c in instruction}
                    data=p+questions[i]
                    seen.update(data[n:n+128] for n in range(0,len(data),128))
                    absent=any(ref[n:n+128] not in seen for n in (0,128))
                    baseline["missing_position_tile"]+=absent
                    counts["correct_missing_position"]+=correct and absent
            for m,flags in (("position",exact_sign),("full",complete),("body",body)):
                assert {"responses":len(rows),"single":sum(flags),"both":sum(a and b for a,b in zip(flags[::2],flags[1::2],strict=True))}==report["arms"][arm][key+"_full"][m]
            if key=="official" and arm in ("matched","reassigned"):
                for j in (0,1):
                    figure_prefix(imgs[j],arm+str(j))
            details[arm]=dict(counts)
        output[key]=dict(baseline=dict(baseline),scores=details)
    write_json(PUBLIC/"audit.json",dict(partitions=output,result_sha256=sha(PUBLIC/"result.json"),source_sha256=sha(Path(__file__))))
    print(output)


if __name__=="__main__":
    audit()
