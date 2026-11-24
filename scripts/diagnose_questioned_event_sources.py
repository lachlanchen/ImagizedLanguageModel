"""Separate question interpretation from competition with unannotated positions."""

from collections import defaultdict
import argparse
from pathlib import Path
import gzip
import json
import torch
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from scripts.answer_questioned_event import components, load_head, RELEASE
from scripts.prepare_questioned_events import PUB, read, dump
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import FACES, Face


@torch.inference_mode()
def measure(output):
    root = PUB / "description_evaluation"
    with gzip.open(root / "decisions.json.gz", "rt") as f:
        data = json.load(f)
    rows = read(PUB / "held.json")
    writer, syntax, lookup, _ = components("cpu")
    net = load_head(RELEASE / "described/aligned", "cpu")
    face = Face(FACES[0])
    cases = defaultdict(list)
    for ix, mode, j, qi, arm, *other in data["records"]:
        if arm == "word_aligned" and mode in (
            "webster",
            "natural_unseen",
            "natural_literal",
        ):
            cases[ix, mode].append((j, data["questions"][qi], other[:2], other[2:4]))
    group = defaultdict(list)
    observations = []
    for (ix, mode), questions in cases.items():
        row = rows[ix]
        events = row["events"]
        width = len(row["page"])
        p, l = packed_batch([row["page"]], face, "cpu")
        h = encoding_windows(writer, cell_features(p))
        a, k = net.page(h, syntax.arc_scores(h, l), lookup.predicates(h))
        verbs = [e[0] for e in events]
        objects = [e[1] for e in events]
        pix, length = packed_batch([q for j, q, _, _ in questions], face, "cpu")
        q, m = lookup.questions(cell_features(pix), length)
        sem = net.question_scores(k, q, m)
        full = net.scores(a, k, q, m)
        pred = full.flatten(1).argmax(-1)
        allowed = full[:, verbs, objects].argmax(-1)
        sem_only = sem[:, verbs].argmax(-1)
        # Counterfactual: both sides are English pictures. Supply exact action names.
        labelpix, labelsize = packed_batch(
            [e[2].replace("-", " ") for e in events], face, "cpu"
        )
        name = lookup.meaning(
            *lookup.words.sentence(cell_features(labelpix), labelsize)
        )
        query = lookup.meaning(*lookup.words.sentence(cell_features(pix), length))
        lexical = (query @ name.T).argmax(-1)
        true = torch.tensor([t[0] for t in questions])
        gold = torch.tensor([verbs[j] * width + objects[j] for j in true])
        assert pred.tolist() == [v * width + o for _, _, (v, o), _ in questions]
        vals = dict(
            full_binding=bool((pred == gold).all()),
            given_arcs_joint=bool((allowed == true).all()),
            given_verbs_semantic=bool((sem_only == true).all()),
            given_english_names=bool((lexical == true).all()),
        )
        g = ("literature" if row["group"] == "literature" else "kyoto") + "_" + mode
        group[g].append(vals)
        observations.append(dict(page=ix, mode=mode, **vals))
    summary = {
        g: dict(n=len(vals), **{k: sum(r[k] for r in vals) for k in vals[0]})
        for g, vals in group.items()
    }
    dump(output, dict(groups=summary, pages=observations))
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    torch.set_num_threads(2)
    measure(p.parse_args().output)
