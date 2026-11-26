"""Assess free translations on new books and paired propositions after both final fits."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
from zipfile import ZipFile, ZIP_DEFLATED
import json
import numpy as np
import torch

from scripts.train_paired_branch_writer import (
    ROOT,
    PUB,
    WORK,
    MODEL,
    base,
    restore,
    qualified_ids,
    bindings,
    gpu_owner,
    configure_cuda_cli,
)
from scripts.evaluate_visual_training_breadth import (
    ARCHIVE as CONTROL,
    PUB as CONTROL_PUB,
    png,
    blind_votes,
    chat,
    server,
    QWEN,
    JUDGE,
    MODELS,
    STORE,
)
from scripts.audit_glyph_draft_editor import unpack
from scripts.write_covered_phrases import CoveredWriter

MODES = ("parent", "positive", "paired")
ARCHIVE = ROOT / "publication/ilm-nature/evidence/paired_branch_writers.zip"


def cases():
    assert base.read(PUB / "books.json")["panel"] == base.sha(WORK / "panel.json")
    book = base.read(WORK / "panel.json")
    rows = []
    for i, r in enumerate(book):
        for t in (0, 1):
            rows.append(
                dict(
                    group="book",
                    doc=r["doc"],
                    t=t,
                    source=r["text"][1 - t],
                    ref=r["text"][t],
                    foil=book[(i + 12) % len(book)]["text"][t],
                )
            )
    families = base.read(WORK / "families.json")
    reserve = base.read(PUB / "primary.json")["reserved"]
    for i in reserve:
        for t in (0, 1):
            for s in (0, 1):
                rows.append(
                    dict(
                        group="paired",
                        i=i,
                        t=t,
                        side=s,
                        source=families[i][s][1 - t],
                        ref=families[i][s][t],
                        foil=families[i][1 - s][t],
                    )
                )
    assert base.read(CONTROL_PUB / "generation.json")["archive"] == base.sha(CONTROL)
    with ZipFile(CONTROL) as z:
        original = json.loads(z.read("records.json"))[96:]
        rows.extend(
            dict(
                group="role",
                t=r["t"],
                source=r["source"],
                ref=r["ref"],
                expected=r["answers"]["broad"]["png"],
            )
            for r in original
        )
    return rows


@torch.inference_mode()
def generate():
    rows = cases()
    net = {m: restore(m, "cuda") for m in MODES}
    face = base.Ink(net["parent"].palette)
    names = base.read(base.BASE / "readback.json")
    temp = ARCHIVE.with_suffix(".partial")
    with ZipFile(temp, "w", ZIP_DEFLATED) as z:
        for t in (0, 1):
            engine = CoveredWriter(t)
            ix = [i for i, r in enumerate(rows) if r["t"] == t]
            for k in range(0, len(ix), 32):
                group = ix[k : k + 32]
                imgs = [
                    net["parent"].palette[face.encode(rows[i]["source"])] for i in group
                ]
                drafts = [engine.translate(img)[0] for img in imgs]
                for i, img in zip(group, imgs, strict=True):
                    blob = png(img)
                    z.writestr(f"{i}/source.png", blob)
                    rows[i].update(source_sha=sha256(blob).hexdigest(), answers={})
                for m in MODES:
                    outputs = net[m].translate(imgs, t, drafts)
                    for i, (pixels, stop) in zip(group, outputs, strict=True):
                        g = net[m].addresses(pixels)
                        data = png(pixels)
                        z.writestr(f"{i}/{m}.png", data)
                        r = rows[i]
                        r["answers"][m] = dict(
                            text="".join(names[str(x)] for x in g),
                            stop=bool(stop),
                            exact=bool(stop and g == face.encode(r["ref"])),
                            n=len(g),
                            sha=sha256(data).hexdigest(),
                        )
                        if r["group"] == "role" and m == "parent":
                            assert r["answers"][m]["sha"] == r["expected"]
        assert not face.bad
        z.writestr("records.json", json.dumps(rows, ensure_ascii=False))
    temp.replace(ARCHIVE)
    base.dump(
        PUB / "free-generation.json",
        dict(
            archive=base.sha(ARCHIVE),
            bindings=bindings(),
            n=len(rows),
            fits={m: base.sha(MODEL / m / "fit.json") for m in MODES[1:]},
        ),
    )


def paired_interval(x):
    rng = np.random.default_rng(38219)
    return np.quantile(
        np.mean(rng.choice(x, (20000, len(x)), replace=True), axis=1), [0.025, 0.975]
    ).tolist()


def assess():
    assert base.read(PUB / "free-generation.json")["archive"] == base.sha(ARCHIVE)
    for m in (QWEN, JUDGE):
        assert base.sha(STORE / "blobs" / ("sha256-" + MODELS[m])) == MODELS[m]
    palette = np.load(base.BASE / "palette.npy")
    names = base.read(base.BASE / "readback.json")
    letters = {
        v.tobytes(): names[str(i)] for i, v in enumerate(palette) if str(i) in names
    }
    with ZipFile(ARCHIVE) as z:
        rows = json.loads(z.read("records.json"))
        for i, r in enumerate(rows):
            for m, v in r["answers"].items():
                data = z.read(f"{i}/{m}.png")
                assert (
                    sha256(data).hexdigest() == v["sha"]
                    and "".join(letters[x.tobytes()] for x in unpack(data, v["n"]))
                    == v["text"]
                )
    book = [r for r in rows if r["group"] == "book"]
    paired = [r for r in rows if r["group"] == "paired"]
    roles = [r for r in rows if r["group"] == "role"]
    assert (len(book), len(paired), len(roles)) == (96, 512, 472)
    cache = WORK / "qwen-books.json"
    if not cache.exists():
        with server() as url, ThreadPoolExecutor(4) as pool:

            def ask(r):
                return chat(
                    url,
                    QWEN,
                    "Translate into "
                    + ("English", "Mandarin")[r["t"]]
                    + ". Return only the translation.\n"
                    + r["source"],
                )

            base.dump(
                cache,
                dict(
                    books=base.sha(WORK / "panel.json"),
                    answers=list(pool.map(ask, book)),
                ),
            )
    q = base.read(cache)
    assert q["books"] == base.sha(WORK / "panel.json")
    candidates = []
    for r, answer in zip(book, q["answers"], strict=True):
        candidates.append(
            dict(
                r["answers"],
                qwen=answer,
                reference=dict(text=r["ref"], stop=True),
                foil=dict(text=r["foil"], stop=True),
            )
        )
    for i, r in enumerate(paired):
        other = paired[i ^ 1]
        assert (
            other["i"] == r["i"] and r["t"] == other["t"] and r["side"] != other["side"]
        )
        candidates.append(
            dict(
                r["answers"],
                **{m + "_other": other["answers"][m] for m in MODES},
                reference=dict(text=r["ref"], stop=True),
                foil=dict(text=r["foil"], stop=True),
            )
        )
    votes = blind_votes(book + paired, candidates)
    grades = [
        {k: a["stop"] and votes[2 * i][k] and votes[2 * i + 1][k] for k, a in g.items()}
        for i, g in enumerate(candidates)
    ]
    by_work = {}
    for i, r in enumerate(book):
        by_work.setdefault(r["doc"], Counter()).update(grades[i])
    book_delta = [(v["paired"] - v["positive"]) / 24 for v in by_work.values()]
    book_interval = np.quantile(
        [np.mean(x) for x in product(book_delta, repeat=4)], [0.025, 0.975]
    ).tolist()
    full = {}
    role = {}
    for m in MODES:
        # Both grader orders must reject the rival answer, including when the two wordings coincide.
        individual = [
            g[m]
            and not votes[2 * i][m + "_other"]
            and not votes[2 * i + 1][m + "_other"]
            for i, g in enumerate(grades)
            if i >= len(book)
        ]
        full[m] = np.asarray(individual).reshape(-1, 4).all(1).astype(int)
        hit = np.array([r["answers"][m]["exact"] for r in roles])
        role[m] = dict(
            exact=int(hit.sum()),
            polarity=int(hit.reshape(-1, 2).all(1).sum()),
            roles=int(hit.reshape(-1, 4).all(1).sum()),
        )
    interval = paired_interval(full["paired"] - full["positive"])
    passed = (
        book_interval[0] > 0
        and interval[0] > 0
        and all(role["paired"][k] >= role["parent"][k] for k in ("polarity", "roles"))
    )
    base.dump(
        PUB / "free-assessment.json",
        dict(
            archive=base.sha(ARCHIVE),
            book=by_work,
            qwen=q["answers"],
            votes=votes,
            grades=grades,
            full_families={m: int(v.sum()) for m, v in full.items()},
            family_interval=interval,
            book_interval=book_interval,
            roles=role,
            passed=bool(passed),
            models={m: MODELS[m] for m in (JUDGE, QWEN)},
        ),
    )
    print(
        "full families",
        {m: int(v.sum()) for m, v in full.items()},
        "books",
        by_work,
        "decision",
        passed,
        flush=True,
    )


if __name__ == "__main__":
    qualified_ids()
    configure_cuda_cli()
    with gpu_owner():
        if not (PUB / "free-generation.json").exists():
            generate()
        if not (PUB / "free-assessment.json").exists():
            assess()
