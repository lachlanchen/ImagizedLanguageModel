"""Assess full free translations and archive all neural answer images before interpreting scores."""
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch
from ilm.visual_lm.portable_writer import save_grid
from ilm.visual_lm.target_glyph_teaching import MODELS
from scripts.check_field_writer_roundtrip import cells
from scripts.evaluate_optical_state_reader import replacement
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_readout_objective import summarize
from scripts.evaluate_teacher_translation import paired
from scripts.evaluate_visual_library import blind_votes
from scripts.readout_work_interval import work_interval
from scripts.replay_target_glyph_output import MODES, ARCHIVE, write_condition
from scripts.reserve_target_glyph_book_panel import BASE, PUB, ROOT, RUN, dump, read, sha
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner
from scripts.write_neural_glyphs import grid_bits


def requests():
    if sha(RUN/"panel.json")!=read(PUB/"panel.json")["sha256"]:
        raise ValueError("Reserved panel differs")
    pairs = read(RUN/"panel.json")
    rows = paired(pairs)
    for i,r in enumerate(rows):
        j=i//2
        r["foil"]=pairs[((j//24+1)%4)*24+j%24]["text"][r["t"]]
    return rows


def generate(rows):
    palette = np.load(BASE/"palette.npy")
    readback = read(BASE/"readback.json")
    alphabet = {bytes(b):readback[str(i)] for i,b in enumerate(palette) if i>=2 and str(i) in readback}
    inputs,answers = [],[{} for _ in rows]
    for i,r in enumerate(rows):
        folder = RUN/"png"/f"{i:04d}"
        folder.mkdir(parents=True,exist_ok=True)
        file = folder/"input.png"
        save_grid(palette[r["input"]].tobytes(),file)
        inputs.append(grid_bits(file,len(r["input"])))
    for m in MODES:
        src = replacement(rows) if m=="exchange" else list(range(len(rows)))
        for i,png,n,stop in write_condition(m,"cuda",inputs,[r["t"] for r in rows],src,24):
            out = RUN/"png"/f"{i:04d}"/(m+".png")
            out.write_bytes(png)
            actual = cells(out.read_bytes(),n).reshape(n,1024)
            chars = [alphabet.get(b.tobytes(),"�") for b in np.packbits(actual,axis=1)]
            answers[i][m] = dict(text="".join(chars),cells=n,unknown=chars.count("�"),stop=stop,
                                 source_index=src[i],png=sha(out))
    return answers


def package(rows, answers, votes):
    temp = ARCHIVE.with_suffix(".tmp")
    with ZipFile(temp,"w",compression=ZIP_DEFLATED) as z:
        for i,r in enumerate(answers):
            for m in ("input",*MODES):
                path = RUN/"png"/f"{i:04d}"/(m+".png")
                if m!="input" and sha(path)!=r[m]["png"]:
                    raise ValueError("PNG hash has changed")
                z.write(path,f"{i:04d}/{m}.png")
        records = [dict(id=i,work=r["doc"],source=r["source"],reference=r["ref"],target_language=("en","zh")[r["t"]],
                        answers=a,source_only=votes[i*2:i*2+2]) for i,(r,a) in enumerate(zip(rows,answers,strict=True))]
        z.writestr("request-outcomes.json",json.dumps(records,ensure_ascii=False))
    temp.replace(ARCHIVE)
    return sha(ARCHIVE)


def main():
    configure_cuda_cli()
    rows = requests()
    for m,v in read(PUB/"training.json").items():
        if sha(MODELS/m/"fit.json")!=v: raise ValueError("Training binding")
    bind = {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__), ROOT/"scripts/replay_target_glyph_output.py",
            PUB/"training.json",RUN/"panel.json")}
    raw,graded,blind = (RUN/(k+".json") for k in ("raw","graded","blind"))
    with gpu_owner():
        torch.set_num_threads(2)
        if not raw.exists(): dump(raw, dict(binding=bind,answers=generate(rows)))
        r = read(raw)
        if r["binding"]!=bind: raise ValueError("Changed generation binding")
        print("All 768 answer PNGs saved",flush=True)
        torch.cuda.empty_cache()
        if not graded.exists():
            a = r["answers"]
            g = semantic(rows,a,MODES)
            dump(graded,dict(raw=sha(raw),answers=a,votes=g["votes"],judge=g["judge"],comparator=g["qwen"]))
        g = read(graded)
        if g["raw"]!=sha(raw): raise ValueError("Different raw answers")
        if not blind.exists(): dump(blind,dict(graded=sha(graded),votes=blind_votes(rows,g["answers"])))
        b = read(blind)
        if b["graded"]!=sha(graded): raise ValueError("Different graded answers")
    a = g["answers"]
    result = summarize(rows,a,b["votes"])
    result["paired_95"] = work_interval(result["works"],"guided","likelihood")
    total = result["total"]
    stops = {m:sum(r[m]["stop"] for r in a) for m in MODES}
    criteria = dict(controls=total["guided"]>max(total["start"],total["likelihood"]),
                    interval=result["paired_95"][0]>0, exchange=total["guided"]>total["exchange"],
                    stop=stops["guided"]>=.95*len(rows))
    dump(PUB/"evaluation.json",dict(binding=bind,raw=sha(raw),graded=sha(graded),votes=sha(blind),
         archive=package(rows,a,b["votes"]),primary=result,reference_informed=summarize(rows,a,g["votes"]),
         criteria=criteria,passed=all(criteria.values()),stops=stops))
    print(total, result["paired_95"], criteria, flush=True)


if __name__=="__main__":
    main()
