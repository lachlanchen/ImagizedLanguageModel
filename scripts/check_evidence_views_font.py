"""Repeat the same Chinese and English questions after changing only the rendering font."""

from hashlib import sha256
from pathlib import Path
from collections import defaultdict
import argparse
import torch
from PIL import ImageFont
from fontTools.ttLib import TTFont
from ilm.visual_lm.book_prefix import observations
from scripts.evaluate_evidence_views import groups, official_index
from scripts.prepare_branch_ink import WORK as BRANCH, PUB as BRANCH_PUB
from scripts.prepare_evidence_views import Alphabet, MODEL, PUB
from scripts.prepare_open_exam import read, dump, sha
from scripts.train_paired_evidence import gpu_owner
from scripts.write_open_exam import load
from scripts.write_visual_discourse import Writer
from scripts.write_recurrent_evidence import generate


def main(root, report):
    judge, _, _, dev, _ = official_index()
    path = BRANCH / "held.json"
    if sha(path) != read(BRANCH_PUB / "corpus.json")["files"]["held.json"]:
        raise ValueError("Changed Chinese test")
    sets = {
        "en": groups(judge, dev)["coqa"],
        "zh": [
            dict(doc=r["doc"], page=r["p"], question=r["q"], refs=[r["answer"]])
            for r in read(path)
            if r["cohort"] == "paired"
        ],
    }
    rows = []
    for lang, items in sets.items():
        keep = sorted(
            {r["doc"] for r in items},
            key=lambda d: sha256(("glyph typeface " + d).encode()).digest(),
        )[:32]
        rows.extend(dict(r, lang=lang) for r in items if r["doc"] in keep)
    assert len(rows) == 128
    font = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")
    result = {}
    with gpu_owner(), torch.inference_mode():
        writer = Writer(device="cuda")
        for face in ("sans", "serif"):
            a = Alphabet()
            if face == "serif":
                a.face.font = ImageFont.truetype(str(font), 26, index=2)
                with TTFont(font, fontNumber=2) as ft:
                    a.face.coverage = set(ft.getBestCmap())
            seq = [(a.encode(r["question"]), a.encode(r["page"])) for r in rows]
            x = torch.from_numpy(observations({"tiles": a.array()}, "ink")).cuda()
            result[face] = {}
            for arm in ("parent", "whole", "highlight"):
                model = load(arm, "cuda", root)
                output = []
                for i in range(0, len(rows), 12):
                    qs, ps = map(list, zip(*seq[i : i + 12]))
                    out = generate(
                        writer.model,
                        model,
                        x,
                        a.n,
                        qs,
                        ps,
                        [p + q for p, q in zip(ps, qs)],
                        64,
                        geometry=(320, 4096),
                    )
                    for r, (ids, stopped) in zip(rows[i : i + 12], out, strict=True):
                        text = "".join(a.chars[j] for j in ids)
                        ok = (
                            text in r["refs"]
                            if r["lang"] == "zh"
                            else judge.normalize_answer(text)
                            in list(map(judge.normalize_answer, r["refs"]))
                        )
                        output.append((r["lang"], r["doc"], stopped and ok))
                vals = {}
                for lang in sets:
                    docs = defaultdict(list)
                    for l, d, ok in output:
                        if l == lang:
                            docs[d].append(ok)
                    assert all(len(g) == 2 for g in docs.values())
                    vals[lang] = dict(
                        exact=sum(sum(g) for g in docs.values()),
                        questions=64,
                        both=sum(all(g) for g in docs.values()),
                        pairs=32,
                    )
                result[face][arm] = vals
    dump(report / "font-transfer.json", dict(serif_sha256=sha(font), scores=result))


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL)
    p.add_argument("--report", type=Path, default=PUB)
    a = p.parse_args()
    main(a.model_dir, a.report)
