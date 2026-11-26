"""Free writing on the prospectively reserved role-grounding books and controlled families."""

from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
import torch

from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.audit_library_role_attention import audit
from scripts.evaluate_visual_library import BASE, blind_votes, cohort, pack, paired, semantic, stopped
from scripts.prepare_teacher_translation import dump
from scripts.sample_visual_library import library, read, sha
from scripts.train_role_grounding import ARMS, MODELS, PUB, ROOT, RUN
from scripts.train_visual_prefix import gpu_owner, material
from scripts.verse_gpu_runtime import configure_cuda_cli
from scripts.write_ordered_translation import restore

WRITERS = {"start": "library-broad", **{arm: "role-" + arm for arm in ARMS}}


def requests():
    file = RUN / "unseen-books.json"
    evidence = read(PUB / file.name)
    if sha(file) != evidence["rows_sha256"]:
        raise ValueError("Changed reserved book images")
    pairs = read(file)
    rows = paired(pairs)
    documents = evidence["titles"]
    rng = np.random.default_rng(620413)
    for j, title in enumerate(documents):
        group = [i for i, r in enumerate(pairs) if r["doc"] == title]
        other = [i for i, r in enumerate(pairs) if r["doc"] == documents[(j+1) % len(documents)]]
        for i, k in zip(group, rng.permutation(other), strict=True):
            for t in (0, 1):
                rows[2*i+t]["foil"] = pairs[k]["text"][t]
    return rows, cohort()[144:1088]


@torch.inference_mode()
def generate(rows):
    features, inv, _ = material("cuda")
    palette = np.load(BASE / "palette.npy")
    index = {bytes(v):i for i,v in enumerate(palette) if i >= 2}
    labels = read(BASE / "readback.json")
    sources = []
    for i, row in enumerate(rows):
        path = RUN / "pixels" / f"{i:04d}" / "input.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_grid(palette[row["input"]].tobytes(), path)
        cells = [index[bytes(v)] for v in read_image(path, len(row["input"]))]
        if cells != row["input"]:
            raise ValueError("Changed source image")
        sources.append(cells)
    out = [{} for r in rows]
    for mode, model in WRITERS.items():
        net = restore(model, "cuda")
        for t in (0, 1):
            ix = [i for i,r in enumerate(rows) if r["t"] == t]
            for j in range(0, len(ix), 48):
                ids = ix[j:j+48]
                seq = net.generate(features, pack([sources[i] for i in ids], "cuda"), inv[t], t).cpu()
                for i, result in zip(ids, seq, strict=True):
                    cells, stop = stopped(result)
                    bitmap = palette[np.asarray(cells, dtype=int)].tobytes()
                    path = RUN / "pixels" / f"{i:04d}" / (mode+".png")
                    save_grid(bitmap, path)
                    out[i][mode] = dict(cells=cells, text="".join(labels[str(c)] for c in cells),
                                        stop=stop, packed=sha256(bitmap).hexdigest(), png=sha(path),
                                        exact=stop and cells==rows[i]["target"],
                                        similarity=stop*Levenshtein.normalized_similarity(cells, rows[i]["target"]))
    return out


def scores(rows, answers, votes):
    per_work = defaultdict(Counter)
    records = []
    if len(votes) != 2*len(rows):
        raise ValueError("Missing assessments")
    for i, (r, a) in enumerate(zip(rows, answers, strict=True)):
        ok = {m:v["stop"] and votes[2*i][m] and votes[2*i+1][m] for m,v in a.items()}
        per_work[r["doc"]].update(ok)
        per_work[r["doc"]]["n"] += 1
        records.append(ok)
    intervals = {}
    for m in ("likelihood", "shuffled", "start"):
        ds = [(v["grounded"]-v[m])/v["n"] for v in per_work.values()]
        intervals[m] = np.quantile([np.mean(vals) for vals in product(ds, repeat=len(ds))], [.025,.975]).tolist()
    return dict(total=sum(per_work.values(), Counter()), works=per_work, records=records, paired_95pct=intervals)


def main():
    configure_cuda_cli()
    panel, controls = requests()
    rows = panel + controls
    n = len(panel)
    if n != 384 or len(controls) != 944:
        raise ValueError("Incomplete evaluation")
    paths = [Path(__file__).resolve(), PUB / "training.json", PUB / "unseen-books.json",
             ROOT / "scripts/evaluate_proposition_translation.py",
             ROOT / "scripts/evaluate_visual_library.py", ROOT / "scripts/write_ordered_translation.py"]
    binding = {str(p.relative_to(ROOT)):sha(p) for p in paths}
    for arm, digest in read(PUB / "training.json")["fits"].items():
        if sha(MODELS / arm / "fit.json") != digest:
            raise ValueError("Changed trained weights")
    raw, guided, blind = [RUN / f for f in ("generated.json", "reference.json", "source.json")]
    with gpu_owner():
        if not raw.exists():
            dump(raw, dict(binding=binding, rows=rows, answers=generate(rows)))
        gen = read(raw)
        if gen["binding"] != binding or gen["rows"] != rows:
            raise ValueError("Changed generated evidence")
        print("Generated or verified every image", flush=True)
        if not guided.exists():
            torch.cuda.empty_cache()
            cand = [dict(a) for a in gen["answers"][:n]]
            result = semantic(panel, cand, tuple(WRITERS))
            dump(guided, dict(raw=sha(raw), answers=cand, votes=result["votes"],
                              judge=result["judge"], qwen=result["qwen"]))
        ref = read(guided)
        if ref["raw"] != sha(raw):
            raise ValueError("Mismatched reference-informed assessment")
        print("Reference-informed votes complete", flush=True)
        if not blind.exists():
            dump(blind, dict(reference=sha(guided), votes=blind_votes(panel, ref["answers"])))
        mask = read(blind)
        if mask["reference"] != sha(guided):
            raise ValueError("Mismatched source-only assessment")
    source = scores(panel, ref["answers"], mask["votes"])
    informed = scores(panel, ref["answers"], ref["votes"])
    out = gen["answers"]
    optical = {m:dict(n=n, exact=sum(a[m]["exact"] for a in out[:n]), stops=sum(a[m]["stop"] for a in out[:n]),
                      similarity=float(np.mean([a[m]["similarity"] for a in out[:n]])),
                      families=sum(all(a[m]["exact"] for a in out[i:i+8]) for i in range(n,len(out),8)))
               for m in WRITERS}
    path = RUN / "dev.npy"
    if sha(path) != read(PUB / "alignment.json")["files"][path.name]:
        raise ValueError("Development annotation hash")
    tags, pairs, optics = torch.from_numpy(np.load(path)), library("dev"), material("cpu")
    diagnostics = {m:audit(restore(model, "cpu"), pairs, tags, optics) for m,model in WRITERS.items()}
    checks = dict(intervals=all(source["paired_95pct"][m][0]>0 for m in ("likelihood","shuffled")),
                  corroboration=all(informed["total"]["grounded"] > informed["total"][m] for m in ("likelihood","shuffled")),
                  families=optical["grounded"]["families"]>=108, stops=optical["grounded"]["stops"]==n)
    result = dict(binding=binding, files={p.name:sha(p) for p in (raw,guided,blind)}, optical=optical,
                  development=diagnostics, source_only=source, reference_informed=informed,
                  qualification=checks, passed=all(checks.values()))
    dump(PUB / "evaluation.json", result)
    print(optical, source["total"], informed["total"], checks, flush=True)


if __name__ == "__main__":
    main()
