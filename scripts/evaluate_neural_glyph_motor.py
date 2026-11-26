"""Run the deployed raw-pixel writer on each saved source image, without target history."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from scripts.write_neural_glyphs import MODEL, grid_bits, load_model, save_ink
from scripts.train_readout_objective import PUB, ROOT, RUN, read, sha, dump
from scripts.evaluate_teacher_translation import paired

OUT=ROOT/"artifacts/neural_glyph_motor_v1"
EVID=ROOT/"references/evidence/neural_glyph_motor_v1"


def evaluate():
    net=load_model("cpu")
    scores=read(PUB/"evaluation.json")
    generated=RUN/"generated.json"
    if sha(generated)!=scores["raw"]: raise ValueError("Changed reference emissions")
    baseline=read(generated)["answers"]
    rows=paired(read(RUN/"fresh.json"))
    evidence=[]
    tally=Counter()
    per_language={}
    for i,row in enumerate(rows):
        lang=("en","zh")[row["t"]]
        p=RUN/"pixels"/f"{i:04d}"
        x=grid_bits(p/"input.png",len(row["input"]))
        ink,n,stop=net.write(torch.from_numpy(x)[None],torch.tensor([len(x)]),row["t"])
        result=ink[0,:int(n[0])].numpy()
        out=OUT/f"{i:04d}.png"
        save_ink(result,out)
        raw=baseline[i]["ilm"]
        if sha(p/"ilm.png")!=raw["png"]: raise ValueError("Changed control image")
        exact=sha(out)==raw["png"] and bool(stop[0])==raw["stop"] and len(result)==len(raw["cells"])
        credit=bool(scores["source"]["records"][i]["ilm"]) if exact else None
        info=dict(index=i,source_cells=len(x),direction=lang,input_png=sha(p/"input.png"),output_png=sha(out),
                  cells=len(result),stop=bool(stop[0]),packed_sha256=sha256(np.packbits(result).tobytes()).hexdigest(),
                  control_png=raw["png"],identical_complete=exact,source_only_credit=credit)
        evidence.append(info)
        for total in (tally,per_language.setdefault(lang,Counter())):
            total.update(dict(n=1,identical=int(exact),credit_preserved=int(credit is True),
                              ungraded=int(credit is None),stops=int(stop[0])))
    archive=ROOT/"publication/ilm-nature/evidence/neural_motor_answers.zip"
    with ZipFile(archive,"w",compression=ZIP_DEFLATED) as z:
        for r in evidence:
            k=f'{r["index"]:04d}'
            z.write(OUT/(k+".png"),k+"/output.png")
            z.write(RUN/"pixels"/k/"input.png",k+"/input.png")
    dump(EVID/"evaluation.json",dict(weights=sha(MODEL/"weights.safetensors"),program=sha(Path(__file__)),
         source_scores=sha(PUB/"evaluation.json"),archive=sha(archive),tally=tally,directions=per_language,
         retained=tally["identical"]/len(rows)>=.99,records=evidence))
    print(tally,flush=True)


if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    EVID.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2)
    evaluate()
