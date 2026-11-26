"""Run the fixed print-transfer panel and preserve the images before semantic assessment."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import json
import numpy as np
import torch

from scripts.check_field_writer_roundtrip import cells
from scripts.evaluate_optical_state_reader import replacement
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_readout_objective import summarize
from scripts.evaluate_teacher_translation import paired
from scripts.evaluate_visual_library import blind_votes
from scripts.readout_work_interval import work_interval
from scripts.replay_target_glyph_output import write_condition
from scripts.reserve_print_translation import BASE, PUB, ROOT, RUN, dump, read, sha
from scripts.train_field_to_glyph import configure_cuda_cli, gpu_owner
from scripts.train_rehearsal_typeface import FONTS, render_face
from scripts.write_neural_glyphs import grid_bits, save_ink
from scripts.write_print_translation import normalize_print
from ilm.visual_lm.portable_writer import save_grid

MODES = ("guided", "likelihood", "serif", "adapted", "exchange")
FONT = FONTS / "opentype/noto/NotoSerifCJK-Regular.ttc"
ARCHIVE = ROOT / "publication/ilm-nature/evidence/print_translation_answers.zip"


def requests():
    if sha(RUN/"panel.json") != read(PUB/"panel.json")["sha256"]:
        raise ValueError("Panel changed")
    pairs = read(RUN/"panel.json")
    rows = paired(pairs)
    for i, row in enumerate(rows):
        group, position = divmod(i//2, 24)
        row["foil"] = pairs[((group+1)%4)*24+position]["text"][row["t"]]
    return rows


def generate(rows):
    face = render_face(FONT, 2)
    palette = np.load(BASE/"palette.npy")
    letters = read(BASE/"readback.json")
    alphabet = {bytes(p):letters[str(i)] for i,p in enumerate(palette) if i >= 2 and str(i) in letters}
    canonical, serif = [], []
    for i, row in enumerate(rows):
        folder = RUN/"images"/f"{i:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        save_grid(palette[row["input"]].tobytes(), folder/"canonical.png")
        save_grid(b"".join(face.tile(c) for c in row["source"]), folder/"serif-input.png")
        n = len(row["input"])
        canonical.append(grid_bits(folder/"canonical.png", n))
        serif.append(grid_bits(folder/"serif-input.png", n))
    adapted = normalize_print(serif, "cpu")
    pixel = []
    answers = [{} for _ in rows]
    for i,(a,b,c) in enumerate(zip(canonical,serif,adapted,strict=True)):
        save_ink(c, RUN/"images"/f"{i:04d}"/"adapted-input.png")
        pixel.append(dict(n=len(a), serif=int((a==b).all((1,2)).sum()), adapted=int((a==c).all((1,2)).sum())))
    for mode in MODES:
        grids = canonical if mode in ("guided","likelihood") else serif if mode=="serif" else adapted
        source = replacement(rows) if mode=="exchange" else list(range(len(rows)))
        if mode=="exchange" and any(i==j or rows[i]["doc"]!=rows[j]["doc"] or rows[i]["t"]!=rows[j]["t"] for i,j in enumerate(source)):
            raise ValueError("Invalid exchange")
        writer = "likelihood" if mode=="likelihood" else "guided"
        for i,png,n,stop in write_condition(writer, "cuda", grids, [r["t"] for r in rows],source,24):
            path = RUN/"images"/f"{i:04d}"/(mode+".png")
            path.write_bytes(png)
            bits = cells(png,n).reshape(n,1024)
            transcript = "".join(alphabet.get(bytes(b),"�") for b in np.packbits(bits,axis=1))
            answers[i][mode] = dict(text=transcript,cells=n,stop=stop,unknown=transcript.count("�"),
                                    png=sha(path),source_index=source[i])
    return dict(answers=answers,cells=pixel)


def artifacts(rows, generated):
    dst=ARCHIVE.with_suffix(".tmp")
    with ZipFile(dst,"w",compression=ZIP_DEFLATED) as z:
        for i,r in enumerate(generated["answers"]):
            for name in ("canonical","serif-input","adapted-input",*MODES):
                path=RUN/"images"/f"{i:04d}"/(name+".png")
                if name in MODES and sha(path)!=r[name]["png"]: raise ValueError("Changed answer")
                z.write(path,f"{i:04d}/{name}.png")
        outcomes=[dict(id=i,source=row["source"],reference=row["ref"],work=row["doc"],
                       to=("en","zh")[row["t"]],answers=answers,source_cells=pixels)
                  for i,(row,answers,pixels) in enumerate(zip(rows,generated["answers"],generated["cells"],strict=True))]
        z.writestr("outcomes.json",json.dumps(outcomes,ensure_ascii=False))
    dst.replace(ARCHIVE)
    return sha(ARCHIVE)


def replay():
    result=read(PUB/"images.json")
    if sha(ARCHIVE)!=result["archive"]: raise ValueError("Archive changed")
    matches=Counter()
    differences=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("outcomes.json"))
        def source(name):
            return [grid_bits(BytesIO(z.read(f'{i:04d}/{name}.png')), len(r["source"])) for i,r in enumerate(rows)]
        canonical, serif=source("canonical"),source("serif-input")
        adapted=normalize_print(serif,"cpu")
        assert all(np.array_equal(a,b) for a,b in zip(adapted,source("adapted-input"),strict=True))
        for m in MODES:
            grids=canonical if m in ("likelihood","guided") else serif if m=="serif" else adapted
            actor="likelihood" if m=="likelihood" else "guided"
            for i,png,n,stop in write_condition(actor,"cpu",grids,[("en","zh").index(r["to"]) for r in rows],
                                                  [r["answers"][m]["source_index"] for r in rows],12):
                a=rows[i]["answers"][m]
                original=z.read(f"{i:04d}/{m}.png")
                if sha256(original).hexdigest()!=a["png"]: raise ValueError("Changed original")
                if (png,n,stop)==(original,a["cells"],a["stop"]): matches[m]+=1
                else:
                    dst=PUB/f"replay-{i:04d}-{m}.png"
                    dst.write_bytes(png)
                    differences.append(dict(index=i,mode=m,sha256=sha(dst),cells=n,stop=stop))
    dump(PUB/"replay.json",dict(archive=result["archive"],matches=matches,differences=differences))
    print("Replayed",matches,"differences",len(differences),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.replay:
        replay()
        return
    configure_cuda_cli()
    rows=requests()
    binding={str(p.relative_to(ROOT)):sha(p) for p in (PUB/"panel.json",ROOT/"references/print_translation_study.md",
        ROOT/"scripts/evaluate_print_translation.py",ROOT/"scripts/write_print_translation.py",
        ROOT/"models/target_glyph_teaching/guided/fit.json",ROOT/"models/target_glyph_teaching/likelihood/fit.json",
        ROOT/"models/book_breadth/typeface/fit.json")}
    raw=RUN/"raw.json"
    with gpu_owner():
        if not raw.exists(): dump(raw,dict(binding=binding,**generate(rows)))
        generated=read(raw)
        if generated["binding"]!=binding: raise ValueError("Changed generation bindings")
        if not (PUB/"images.json").exists():
            dump(PUB/"images.json",dict(binding=binding,font_sha256=sha(FONT),archive=artifacts(rows,generated),raw=sha(raw)))
        if sha(ARCHIVE)!=read(PUB/"images.json")["archive"]: raise ValueError("Changed raw archive")
        print("Frozen",len(rows)*len(MODES),"GPU answers",flush=True)
        torch.cuda.empty_cache()
        graded,blind=RUN/"reference.json",RUN/"blind.json"
        if not graded.exists():
            answers=generated["answers"]
            g=semantic(rows,answers,MODES)
            dump(graded,dict(raw=sha(raw),answers=answers,votes=g["votes"],judge=g["judge"],comparator=g["qwen"]))
        doc=read(graded)
        if doc["raw"]!=sha(raw): raise ValueError("Changed answers")
        if not blind.exists(): dump(blind,dict(graded=sha(graded),votes=blind_votes(rows,doc["answers"])))
        votes=read(blind)
        if votes["graded"]!=sha(graded): raise ValueError("Changed graded answers")
    primary=summarize(rows,doc["answers"],votes["votes"])
    sensitivity=summarize(rows,doc["answers"],doc["votes"])
    primary["paired_interval"]=work_interval(primary["works"],"adapted","serif")
    report=dict(primary=primary,reference_informed=sensitivity,archive=sha(ARCHIVE),
                primary_votes=votes["votes"],reference_informed_votes=doc["votes"],
                comparators=[{k:r[k] for k in ("qwen","reference","foil")} for r in doc["answers"]],
                judges={k:doc[k] for k in ("judge","comparator")})
    dump(PUB/"assessment.json",report)
    print(primary["total"],primary["paired_interval"],flush=True)


if __name__=="__main__":
    main()
