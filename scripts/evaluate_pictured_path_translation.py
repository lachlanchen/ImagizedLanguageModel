"""Compare image-derived source paths on unseen books with frozen visual output weights."""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer

from ilm.visual_lm.pictured_path_writer import COUNTS, OpticalSpanWriter, PicturedPathWriter
from ilm.visual_lm.portable_writer import save_grid
from scripts.evaluate_optical_span_rollout import (BASE, blind_votes, configure_cuda_cli, gpu_owner, grid_bits,
    paired, replacement, save_ink, semantic, summarize, work_interval, write_condition)
from scripts.reserve_pictured_path_translation import PROTOCOL, PUB, ROOT, RUN, dump, read, sha

ARCHIVE = ROOT / "publication/ilm-nature/evidence/pictured_path_answers.zip"
MODES = ("path", "field", "exchange", "native", "single")


class SuppliedSource:
    def __init__(self, reader, paths):
        self.reader = reader
        self.paths = paths

    def states(self, features, source):
        return self.reader.encode(self.paths, source.device)


def requests():
    assert sha(RUN / "panel.json") == read(PUB / "panel.json")["sha256"]
    pairs = read(RUN / "panel.json")
    rows = paired(pairs)
    for i, row in enumerate(rows):
        row["foil"] = pairs[(i//2+24)%len(pairs)]["text"][row["t"]]
    return rows


def emit(rows, grids, device):
    identity = list(range(len(rows)))
    directions = [r["t"] for r in rows]
    for i, png, count, stop in write_condition("guided", device, grids, directions, identity, 12):
        yield i, "single", png, count, stop, None, None, None
    swapped = replacement(rows)
    for t, lang in enumerate(("en", "zh")):
        idx = [i for i in identity if directions[i] == t]
        fit = read(COUNTS / (lang + "-fit.json"))
        folder = snapshot_download(fit["model"], revision=fit["revision"], local_files_only=True)
        tokenizer = MarianTokenizer.from_pretrained(folder, local_files_only=True)
        for model_type, modes in ((OpticalSpanWriter, ("field",)), (PicturedPathWriter, ("path", "exchange", "native"))):
            writer = model_type(lang, device)
            reader = writer.reader
            for mode in modes:
                for j in range(0, len(idx), 24):
                    group = idx[j:j+24]
                    inputs = [grids[swapped[i] if mode == "exchange" else i] for i in group]
                    path_matches = [None] * len(group)
                    if mode in ("path", "native"):
                        native = [tokenizer(rows[i]["source"]).input_ids[:-1] for i in group]
                        if mode == "native":
                            writer.reader = SuppliedSource(reader, native)
                        else:
                            # Compare the actual downsampled visual paths with native source addresses.
                            features = torch.as_tensor(np.concatenate(inputs), device=device, dtype=torch.float32)
                            pooled = torch.nn.functional.avg_pool2d(features[:,None], 2).flatten(1)
                            indices = torch.arange(1,1+len(pooled),device=device).split(list(map(len,inputs)))
                            handles = torch.nn.utils.rnn.pad_sequence(indices,batch_first=True)
                            inferred = reader.addresses(torch.nn.functional.pad(pooled,(0,0,1,0)),handles)
                            path_matches = [x==y for x,y in zip(inferred,native,strict=True)]
                    try:
                        outputs = writer.write(inputs)
                    finally:
                        writer.reader = reader
                    for i,(ink,stop,edges),match in zip(group,outputs,path_matches,strict=True):
                        buf = BytesIO()
                        save_ink(ink, buf, image_format="PNG")
                        yield i, mode, buf.getvalue(), len(ink), stop, edges, match, writer.cost
            del writer, reader


def binding():
    files = [Path(__file__), PROTOCOL, ROOT/"references/pictured_path_translation_controls.md", PUB/"panel.json",
             ROOT/"ilm/visual_lm/pictured_path_writer.py", ROOT/"ilm/visual_lm/optical_span_rollout.py",
             ROOT/"scripts/fit_pictured_source_spans.py", COUNTS/"source-paths.json", COUNTS/"generation.json",
             ROOT/"models/target_glyph_teaching/guided/fit.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def generate(rows):
    start = monotonic()
    palette = np.load(BASE/"palette.npy")
    labels = read(BASE/"readback.json")
    names = {bytes(v):labels[str(i)] for i,v in enumerate(palette) if i>=2 and str(i) in labels}
    answers = [{} for _ in rows]
    tmp = ARCHIVE.with_suffix(".tmp")
    costs = {}
    swaps = replacement(rows)
    with ZipFile(tmp, "w", compression=ZIP_DEFLATED) as z:
        grids = []
        for i,row in enumerate(rows):
            p = RUN/f"{i:04d}.png"
            save_grid(palette[row["input"]].tobytes(), p)
            z.write(p,f"{i:04d}/source.png")
            grids.append(grid_bits(p,len(row["input"])))
        torch.cuda.reset_peak_memory_stats()
        for i,m,png,n,stop,edges,match,cost in emit(rows,grids,"cuda"):
            z.writestr(f"{i:04d}/{m}.png",png)
            if cost:
                costs[f"{rows[i]['t']}/{m}"] = cost
            ink = grid_bits_out(png,n)
            answers[i][m] = dict(text="".join(names.get(bytes(v),"�") for v in ink),stop=bool(stop),cells=n,
                                 boundaries=edges,source=swaps[i] if m=="exchange" else i,
                                 native_path_equal=match,sha256=sha256(png).hexdigest())
        assert all(set(a)==set(MODES) for a in answers)
        records = [dict(source=r["source"],reference=r["ref"],work=r["doc"],direction=("en","zh")[r["t"]],answers=a)
                   for r,a in zip(rows,answers,strict=True)]
        z.writestr("records.json",json.dumps(records, ensure_ascii=False))
    peak = torch.cuda.max_memory_allocated()
    elapsed = monotonic()-start
    if peak > 20*2**30 or elapsed > 7200:
        raise RuntimeError("Writer resource ceiling exceeded")
    tmp.replace(ARCHIVE)
    dump(PUB/"generation.json",dict(binding=binding(),archive=sha(ARCHIVE),costs=costs,
                                    seconds=elapsed,peak_bytes=peak,answers=len(rows)*len(MODES)))


def grid_bits_out(png,n):
    from scripts.check_field_writer_roundtrip import cells
    return np.packbits(cells(png,n).reshape(n,1024),axis=1)


def replay(rows):
    same = Counter()
    diffs = []
    with ZipFile(ARCHIVE) as z:
        reviewed = json.loads(z.read("records.json"))
        grids = [grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),len(r["input"])) for i,r in enumerate(rows)]
        for i,m,png,n,stop,edges,exact,_ in emit(rows,grids,"cpu"):
            old = reviewed[i]["answers"][m]
            if (png,n,stop,edges,exact)==(z.read(f"{i:04d}/{m}.png"),old["cells"],old["stop"],old["boundaries"],old["native_path_equal"]):
                same[m]+=1
            else:
                diffs.append(dict(case=i,mode=m,sha256=sha256(png).hexdigest()))
                (PUB/f"cpu-{i:04d}-{m}.png").write_bytes(png)
    dump(PUB/"replay.json",dict(archive=sha(ARCHIVE),same=same,differences=diffs))
    print("CPU repetitions",same,"differences",len(diffs),flush=True)


def assess(rows):
    with ZipFile(ARCHIVE) as z:
        answers = [r["answers"] for r in json.loads(z.read("records.json"))]
    cache = RUN/"reference.json"
    if not cache.exists():
        evaluation = semantic(rows,answers,MODES)
        dump(cache,dict(archive=sha(ARCHIVE),answers=answers,votes=evaluation["votes"],
                        comparator=evaluation["qwen"],judge=evaluation["judge"]))
    data = read(cache)
    assert data["archive"]==sha(ARCHIVE)
    answers = data["answers"]
    votes = blind_votes(rows,answers)
    primary = summarize(rows,answers,votes)
    interval = work_interval(primary["works"],"path","field")
    passed = (interval[0]>0 and all(d["path"]>=d["field"] for d in primary["directions"].values())
              and primary["total"]["path"]>primary["total"]["exchange"])
    diagnostics = Counter()
    for a in answers:
        x,y=a["path"],a["native"]
        exact = x["native_path_equal"]
        equal = all(x[k]==y[k] for k in ("sha256","cells","stop","boundaries"))
        diagnostics[f"path_{exact}_output_{equal}"]+=1
    dump(PUB/"assessment.json",dict(archive=data["archive"],primary=primary,interval=interval,passed=passed,
        diagnostics=diagnostics,votes=votes,secondary_votes=data["votes"],reference_informed=summarize(rows,answers,data["votes"]),
        judge=data["judge"],comparator=data["comparator"],controls=[{k:a[k] for k in ("qwen","reference","foil")} for a in answers]))
    print(primary["total"],interval,diagnostics,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("generate","assess","replay"))
    action = parser.parse_args().action
    torch.set_num_threads(2)
    rows=requests()
    record=PUB/"generation.json"
    if action=="generate":
        if record.exists():
            assert read(record)["binding"]==binding() and read(record)["archive"]==sha(ARCHIVE)
            return
        configure_cuda_cli()
        with gpu_owner(): generate(rows)
    else:
        assert read(record)["binding"]==binding() and read(record)["archive"]==sha(ARCHIVE)
        if action=="replay":
            replay(rows)
        elif not (PUB/"assessment.json").exists():
            configure_cuda_cli()
            with gpu_owner(): assess(rows)


if __name__=="__main__":
    main()
