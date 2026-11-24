"""Evaluate complete written answers on new conversational and list questions."""

from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scripts.answer_dialogue_surface import load, answer
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_teacher_stories import absent, new_stories
from scripts.evaluate_dialogue_ink import exchanges
from scripts.evaluate_dialogue_evidence import predict
from scripts.evaluate_optical_passage_qa import normalized, metrics, save_page
from scripts.prepare_compose_dialogue import MODEL, prepared, kind
from scripts.prepare_dialogue_ink import dump, sha
from scripts.train_dialogue_evidence import RELEASE
from scripts.train_dialogue_surface import representation, corpus
from scripts.train_span_choice import gpu_owner
from ilm.visual_lm.portable_writer import read_image
from ilm.visual_lm.causal_glyph_writer import write_strip


def assignment(rows):
    q, p = exchanges(rows)
    for i, r in enumerate(rows):
        if rows[q[i]][3] == r[3]:
            q[i] = next(
                j
                for d in range(1, len(rows))
                if (j := (p[i] + d - 1) % len(rows)) != i and rows[j][3] != r[3]
            )
        if rows[p[i]][1] == r[1]:
            raise ValueError("Identical passage control")
    return q, p


def confidence(values, rows, idx):
    grouped = defaultdict(list)
    for i in idx:
        grouped[rows[i][1]].append(values[i])
    if not grouped:
        return None
    stat = np.array([(sum(v), len(v)) for v in grouped.values()])
    rng = np.random.default_rng(420121)
    pos = rng.integers(len(stat), size=(5000, len(stat)))
    rep = stat[pos].sum(1)
    return dict(
        delta=float(stat[:, 0].sum() / stat[:, 1].sum()),
        interval95=np.quantile(rep[:, 0] / rep[:, 1], [0.025, 0.975]).tolist(),
        stories=len(stat),
    )


def score(records, refs, markers, official):
    vals = np.zeros((len(records), 3, 3))
    for i, entry in enumerate(records):
        for j, key in enumerate(("original", "question", "passage")):
            trial = entry[key]
            if trial["stop"]:
                text = trial["answer"]
                exact, f1 = metrics(text, refs[i])
                if official:
                    f1 = official.compute_turn_score(*markers[i], text)["f1"]
                vals[i, j] = [exact, f1, 1]
    return vals


def summarize(vals, subsets):
    result = {}
    grounded = (vals[:, 0, 0] == 1) & (vals[:, 1:, 0].sum(1) == 0)
    for k, ind in subsets.items():
        sel = vals[ind]
        result[k] = dict(
            n=len(ind),
            grounded=int(grounded[ind].sum()),
            exact=sel[:, :, 0].sum(0).astype(int).tolist(),
            f1=sel[:, :, 1].mean(0).tolist() if len(ind) else [],
            stopped=sel[:, :, 2].sum(0).astype(int).tolist(),
        )
    return result, grounded


def paths_for(mode, root):
    return dict(
        plain=RELEASE / mode / "plain",
        assembly=root / mode / "assembly",
        lists=root / mode / "lists",
    )


def sample_pixels(eg, row, data, chosen, paths, out):
    folder = out / "examples"
    folder.mkdir(exist_ok=True)
    face, _, _ = representation("ink")
    fields = []
    for label, s in zip(
        ("passage", "question", "history"), (data["pages"][row[1]], row[3], row[4])
    ):
        if not s:
            fields.append(np.empty((0, 128), np.uint8))
            continue
        p = folder / f"{eg}-{label}.png"
        save_page(s, p, face)
        fields.append(read_image(p, None))
    answers = {}
    for arm, r in chosen.items():
        raw, state = answer(load(paths[arm]), fields)
        if raw != face(r["answer"]) or state["stopped"] != r["stop"]:
            raise ValueError("Single-image replay failed")
        write_strip(
            np.frombuffer(raw, np.uint8).reshape(-1, 128), folder / f"{eg}-{arm}.png"
        )
        answers[arm] = r | dict(actions=state["actions"])
    return dict(
        question=row[3], target=row[5], index=row[2], page=row[1], answers=answers
    )


def evaluate(root, out):
    out.mkdir(exist_ok=False, parents=True)
    material = prepared()
    old = read_data()
    cohorts = {
        "fresh": (material["fresh"], None, None),
        "lists": (material["lists"], None, None),
    }
    for group in ("long", "longer"):
        cohorts[group] = new_stories(old, group)
    trained = {normalized(r[5]) for r in old["rows"] if r[0] == "train"} | {
        normalized(r[5]) for r in material["extra"]["rows"]
    }
    result = {}
    for group, (data, markers, official) in cohorts.items():
        rows = data["rows"]
        refs = (
            [official.gold_data[m] for m in markers]
            if official
            else [[r[5]] for r in rows]
        )
        fields = [(data["pages"][r[1]], r[3], r[4]) for r in rows]
        hard = np.array([absent(v, s) for v, s in zip(refs, fields)])
        classes = np.array([kind(r[5], s) for r, s in zip(rows, fields)])
        subsets = {
            "all": np.arange(len(rows)),
            "difficult": np.flatnonzero(hard),
            "ordinary": np.flatnonzero(~hard),
            "assembly": np.flatnonzero(hard & (classes == "assembly")),
            "spell": np.flatnonzero(hard & (classes == "spell")),
            "new_wording": np.flatnonzero(
                hard
                & np.array(
                    [not any(normalized(a) in trained for a in rr) for rr in refs]
                )
            ),
            "within_word_limit": np.array(
                [i for i, r in enumerate(rows) if len(r[5].split()) <= 16], dtype=int
            ),
        }
        swaps = assignment(rows)
        g = result[group] = dict(
            stories=len(set(r[1] for r in rows)),
            controls=["original", "question_and_history", "passage"],
            models={},
            comparisons={},
            examples={},
        )
        for mode in ("ink", "byte"):
            face, unit, sep = representation(mode)
            decode = {face(chr(i)): chr(i) for i in range(32, 127)}
            paths = paths_for(mode, root)
            records = {}
            values = {}
            grounds = {}
            ds = corpus(data, "fresh", load(paths["plain"], "cuda"), face, unit)
            for arm, folder in paths.items():
                rec = predict(
                    folder, ds, load(folder, "cuda"), sep, unit, decode, rows, swaps
                )
                records[arm] = rec
                values[arm] = score(rec, refs, markers, official)
                summary, grounds[arm] = summarize(values[arm], subsets)
                file = out / f"{group}-{mode}-{arm}.json.gz"
                file.write_bytes(
                    gzip.compress(
                        json.dumps(rec, separators=(",", ":")).encode(), mtime=0
                    )
                )
                g["models"][mode + "/" + arm] = dict(
                    scores=summary,
                    weights=sha(folder / "writer.safetensors"),
                    outputs=sha(file),
                )
            for before, after in [
                ("plain", "assembly"),
                ("assembly", "lists"),
                ("plain", "lists"),
            ]:
                a = values[after][:, 0, 1] - values[before][:, 0, 1]
                change = {}
                for label in ("all", "difficult"):
                    ix = subsets[label]
                    b, t = grounds[before][ix], grounds[after][ix]
                    change[label] = dict(
                        f1=confidence(a, rows, ix),
                        new=int((t & ~b).sum()),
                        lost=int((b & ~t).sum()),
                    )
                g["comparisons"][mode + "/" + before + "-" + after] = change
            if mode == "ink" and group in ("fresh", "lists"):
                eligible = hard if group == "fresh" else np.ones(len(rows), bool)
                for typ, mask in [
                    (
                        "gain",
                        ~grounds["plain"] & (grounds["assembly"] | grounds["lists"]),
                    ),
                    (
                        "miss",
                        ~grounds["plain"] & ~grounds["assembly"] & ~grounds["lists"],
                    ),
                ]:
                    ids = np.flatnonzero(eligible & mask)
                    if len(ids):
                        i = ids[0]
                        tag = group + "-" + typ
                        g["examples"][typ] = sample_pixels(
                            tag,
                            rows[i],
                            data,
                            {a: v[i]["original"] for a, v in records.items()},
                            paths,
                            out,
                        )
        print(
            group,
            {k: v["scores"]["difficult"]["grounded"] for k, v in g["models"].items()},
            flush=True,
        )
    dump(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=MODEL)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    with gpu_owner():
        evaluate(a.models, a.output)
