"""Measure autonomous complete answers, fact edits and the paired region intervention."""

from collections import defaultdict
import copy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import sys
from urllib.request import urlopen
import numpy as np
import torch

from scripts.prepare_region_writer import WORK, MODEL, PUB, Ink, regions
from scripts.prepare_evidence_views import ROOT
from scripts.prepare_open_exam import read, dump, sha
from scripts.prepare_grounded_phrasings import phrase_key
from scripts.write_region_writer import answer, model
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import packed
from scripts.score_dialogue_ink_references import official_index
from scripts.evaluate_evidence_views import groups


def e2e_scorer():
    aux = WORK / "tools/tgen/data.py"
    aux.parent.mkdir(parents=True, exist_ok=True)
    if not aux.exists():
        with urlopen(
            "https://raw.githubusercontent.com/UFAL-DSG/tgen/3c43c0e29faa7ea3857a6e490d9c28a8daafc7d0/tgen/data.py",
            timeout=30,
        ) as r:
            aux.write_bytes(r.read())
    if sha(aux) != "680716a261f47026808031a8cfef409e10a36203c034f2944d9d54add35f2000":
        raise ValueError("Changed TGen data adapter")
    sys.path.insert(0, str(aux.parent.parent))
    p = ROOT / "data/raw/e2e_cleaned/slot_error.py"
    if (
        sha(p)
        != read(ROOT / "references/evidence/semantic_page_v1/corpus.json")["files"][
            p.name
        ]
    ):
        raise ValueError("Changed slot scorer")
    spec = spec_from_file_location("slot_error_official", p)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)

    def score(text, facts):
        mr = m.DA.parse_diligent_da(", ".join(f"{k}[{v}]" for k, v in facts.items()))
        actual, expected = m.reclassify_mr(text, mr)
        errors = m.check_output(expected, actual)[:4]
        return list(map(int, errors))

    return score


def changed(rows):
    output = []
    for i, row in enumerate(rows):
        task = row["task"]
        key = "food" if task == "restaurant" else next(iter(row["facts"]))
        old = row["facts"].get(key, "")
        if not old or old.casefold() in row["q"].casefold():
            continue
        donors = rows[i + 1 :] + rows[:i]
        for d in donors:
            new = d["facts"].get(key)
            if (
                d["task"] != task
                or not new
                or new.casefold()
                in (" ".join(row["blocks"]) + " " + row["q"]).casefold()
            ):
                continue
            if task == "statements":
                # Substitute like-for-like numeric strings or capitalized names of comparable size.
                if len(old) > 40 or len(new) > 40:
                    continue
                if not (
                    (old.isdecimal() and new.isdecimal())
                    or (
                        old.istitle()
                        and new.istitle()
                        and len(old.split()) == len(new.split())
                    )
                ):
                    continue
                if not any(old in s for s in row["answers"]):
                    continue
            if new in old or old in new:
                continue
            clone = copy.deepcopy(row)
            clone["original"] = i
            clone["old"] = old
            clone["new"] = new
            if task == "restaurant":
                token = f"food = {old}"
                repl = f"food = {new}"
            elif task == "catalogue":
                token = key + " " + old
                repl = key + " " + new
            else:
                token = old
                repl = new
            for j in clone["valid"]:
                clone["blocks"][j] = clone["blocks"][j].replace(token, repl)
            if (
                clone["blocks"] == row["blocks"]
                or any(len(b) > 256 for b in clone["blocks"])
                or sum(map(len, clone["blocks"])) > 1536
            ):
                continue
            clone["answers"] = [s.replace(old, new) for s in row["answers"] if old in s]
            clone["facts"][key] = new
            output.append(clone)
            break
    return output


def offpage():
    judge, _, _, dev, _ = official_index()
    rows = groups(judge, dev)["composition"]
    ink = Ink()
    out = []
    excluded = 0
    for r in rows:
        q, p = r["question"], r["page"]
        forms = [judge.normalize_answer(s) for s in (q, p)]
        forms_physical = ["".join(map(chr, ink.encode(s))) for s in (q, p)]
        if any(
            any(judge.normalize_answer(t) in s for s in forms)
            or any("".join(map(chr, ink.encode(t))) in s for s in forms_physical)
            for t in r["refs"]
        ):
            continue
        ps = regions(p)
        if not (0 < len(q) <= 192 and 1 <= len(ps) <= 24 and sum(map(len, ps)) <= 1536):
            excluded += 1
            continue
        out.append(
            dict(
                task="coqa",
                family=r["doc"],
                q=q,
                blocks=ps,
                valid=[],
                answers=r["refs"],
                facts={},
            )
        )
    assert len(out) + excluded == 116
    return out, excluded, judge.normalize_answer


def score_rows(outputs, rows, slot_scorer, norm):
    scores = []
    for r, (text, ids, stop, pick, h) in zip(rows, outputs, strict=True):
        t = r["task"]
        refs = r["answers"]
        normalize = norm if t == "coqa" else phrase_key
        exact = (
            stop
            and bool(normalize(text))
            and any(normalize(text) == normalize(ref) for ref in refs)
        )
        has = lambda x: bool(phrase_key(x)) and phrase_key(x) in phrase_key(text)
        errors = slot_scorer(text, r["facts"]) if t == "restaurant" else []
        fact = stop and (
            (not any(errors))
            if errors
            else (has(next(iter(r["facts"].values()))) if r["facts"] else False)
        )
        scores.append(
            dict(
                task=t,
                family=r["family"],
                exact=exact,
                grounded_proxy=fact,
                stop=stop,
                cells=len(ids),
                selected=pick,
                region_hit=(
                    pick in r["valid"] if pick is not None and r["valid"] else None
                ),
                text=text,
                sha256=h,
                slots=errors,
                fact_total=len(r["facts"]),
                exact_old_new=(stop and has(r["new"]) and not has(r["old"]))
                if "old" in r
                else None,
            )
        )
    return scores


def run_group(net, sel, mode, records, ink, book, score, norm):
    qs = [ink.encode(r["q"]) for r in records]
    blocks = [[ink.encode(s) for s in r["blocks"]] for r in records]
    tiles = ink.array()
    byid = {i: c for i, c in enumerate(book)}
    for c, i in ink.chars.items():
        byid.setdefault(i, c)
    outputs = []
    for j in range(0, len(qs), 12):
        rows = records[j : j + 12]
        forced = [r["valid"][0] for r in rows] if mode == "oracle" else None
        raw, picks = answer(
            net,
            sel,
            "direct" if mode == "direct" else "selected",
            tiles,
            len(book),
            qs[j : j + 12],
            blocks[j : j + 12],
            "cuda",
            supplied=forced,
        )
        for (ids, end), k in zip(raw, picks):
            h = sha256(tiles[np.asarray(ids, int)].tobytes()).hexdigest()
            outputs.append(("".join(byid[i] for i in ids), ids, end, k, h))
    scored = score_rows(outputs, records, score, norm)
    for s, (_, ids, _, _, _), q, ps in zip(scored, outputs, qs, blocks):
        run = "".join(map(chr, ids))
        s["ink_hex"] = tiles[np.asarray(ids, int)].tobytes().hex()
        s["visible_reply"] = bool(run) and any(
            run in "".join(map(chr, z)) for z in [q, *ps]
        )
    return scored


def summarize(rows):
    bytask = defaultdict(list)
    for r in rows:
        bytask[r["task"]].append(r)
    return {
        t: dict(
            n=len(g),
            exact=sum(r["exact"] for r in g),
            grounded_proxy=sum(r["grounded_proxy"] for r in g),
            selected_correct=sum(r["region_hit"] is True for r in g),
            stopped=sum(r["stop"] for r in g),
            exact_absent=sum(r["exact"] and not r["visible_reply"] for r in g),
            median_cells=float(np.median([r["cells"] for r in g])),
            slot_errors=sum(sum(r["slots"]) for r in g),
            slots=sum(r["fact_total"] for r in g) if t == "restaurant" else None,
        )
        for t, g in bytask.items()
    }


def main():
    rows = read(WORK / "dev-source.json")
    if (
        sha(WORK / "dev-source.json")
        != read(PUB / "corpus.json")["files"]["dev-source.json"]
    ):
        raise ValueError("Changed development sources")
    edits = changed(rows)
    external, excluded, norm = offpage()
    score = e2e_scorer()
    # Measure whether a retained human answer realizes all annotated restaurant slots.
    human = {
        r["family"]: min(sum(score(s, r["facts"])) for s in r["answers"])
        for r in rows
        if r["task"] == "restaurant"
    }
    result = {}
    all_out = {}
    book = read(MODEL / "readback.json")
    ink = Ink(np.load(MODEL / "palette.npy"))
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        sel = model("selector", "cuda")
        for mode in ("direct", "selected", "oracle"):
            net = model("direct" if mode == "direct" else "selected", "cuda")
            scored = run_group(net, sel, mode, rows, ink, book, score, norm)
            altered = run_group(net, sel, mode, edits, ink, book, score, norm)
            prior = []
            for e, s in zip(edits, altered):
                old = scored[e["original"]]
                prior.append(
                    dict(
                        task=e["task"],
                        both=old["grounded_proxy"] and s["grounded_proxy"],
                        changes=old["sha256"] != s["sha256"],
                        new_without_old=s["exact_old_new"],
                    )
                )
            outdomain = (
                run_group(net, sel, mode, external, ink, book, score, norm)
                if mode != "oracle"
                else []
            )
            all_out[mode] = scored
            packed(
                WORK / (mode + "-predictions.json.gz"),
                dict(original=scored, edited=altered, coqa=outdomain),
            )
            result[mode] = dict(
                original=summarize(scored),
                edited=summarize(altered),
                changes={
                    t: {
                        k: sum(bool(z[k]) for z in prior if z["task"] == t)
                        for k in ("both", "changes", "new_without_old")
                    }
                    for t in ("statements", "restaurant", "catalogue")
                },
                coqa=summarize(outdomain),
            )
            print(mode, result[mode], flush=True)
    effects = {}
    for t in ("restaurant", "catalogue", "statements"):
        pairs = [
            (g, p)
            for g, p in zip(all_out["oracle"], all_out["selected"])
            if g["task"] == t
        ]
        same = [(g, p) for g, p in pairs if g["selected"] == p["selected"]]
        assert all(
            g["sha256"] == p["sha256"] and g["stop"] == p["stop"] for g, p in same
        )
        n = sum(g["exact"] - p["exact"] for g, p in pairs)
        up = sum(g["exact"] and not p["exact"] for g, p in pairs)
        down = sum(p["exact"] and not g["exact"] for g, p in pairs)
        assert n == up - down
        effects[t] = dict(same=len(same), delta=n, rescued=up, spoiled=down)
    dump(
        PUB / "outcomes.json",
        dict(
            scores=result,
            region_effect=effects,
            coqa_outside_geometry=excluded,
            restaurant_human_perfect=sum(x == 0 for x in human.values()),
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
