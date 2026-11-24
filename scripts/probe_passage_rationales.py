"""Localize open-answer errors by intervening on human-marked English evidence."""

from collections import Counter
from hashlib import sha256
import numpy as np
import torch
from scripts.score_dialogue_ink_references import official_index
from scripts.prepare_open_exam import read, dump
from scripts.prepare_branch_ink import PUB, EXAM, MODEL
from scripts.train_paired_evidence import gpu_owner
from scripts.check_branch_ink_transfer import write
from scripts.evaluate_competing_facts import CorpusFace
from scripts.evaluate_visual_discourse import packed
from scripts.write_visual_discourse import Writer
from scripts.write_open_exam import load


def pages(source):
    rows = []
    exclusions = Counter()
    for entry in read(source)["data"]:
        story = entry["story"][:600]
        q, a = entry["questions"][0], entry["answers"][0]
        lo, hi = a["span_start"], a["span_end"]
        if not 0 <= lo < hi <= len(story) or story[lo:hi] != a["span_text"]:
            exclusions["evidence outside crop or offset mismatch"] += 1
            continue
        w = hi - lo
        # Nearest equally long, disjoint passage. Prefer following text when both fit.
        other = hi if hi + w <= len(story) else lo - w
        if other < 0 or len(q["input_text"]) > 160:
            exclusions["no disjoint equal-length comparison or question too long"] += 1
            continue
        assert q["turn_id"] == a["turn_id"] == 1
        rows.append(
            dict(
                doc=entry["id"],
                question=q["input_text"],
                answer=a["input_text"],
                full=story,
                evidence=story[lo:hi],
                disjoint=story[other : other + w],
                masked=story[:lo] + " " * w + story[hi:],
            )
        )
    return rows, exclusions


def main():
    judge, _, _, source, _ = official_index()
    rows, excluded = pages(source)
    writer = Writer(device="cuda")
    shapes = list(map(bytes, writer.palette))
    alphabet = list(writer.transcript)
    lookup = {p: i for i, p in enumerate(shapes) if i >= 2}
    face = CorpusFace()
    cache = {}

    def encode(s):
        for c in set(s):
            if c not in cache:
                try:
                    tile = face.tile(c)
                except ValueError:
                    tile = face.tile("�")
                if tile not in lookup:
                    lookup[tile] = len(shapes)
                    shapes.append(tile)
                    alphabet.append(c)
                cache[c] = lookup[tile]
        return [cache[c] for c in s]

    conditions = ("full", "evidence", "disjoint", "masked")
    prepared = {
        k: [dict(q=encode(r["question"]), s=encode(r[k])) for r in rows]
        for k in conditions
    }
    tiles = np.frombuffer(b"".join(shapes), np.uint8).reshape(-1, 128)
    norm = judge.normalize_answer
    report = {}
    for name, root in (("parent", EXAM), ("matched", EXAM), ("branch", MODEL)):
        net = load(name, "cuda", root)
        all_results = []
        report[name] = {}
        for k in conditions:
            stats = []
            outputs = write(writer, net, tiles, prepared[k])
            for r, (ids, end) in zip(rows, outputs, strict=True):
                text = "".join(alphabet[i] for i in ids)
                key = (r["doc"], 1)
                score = judge.compute_turn_score(*key, text)
                refs = judge.gold_data[key]
                ref = [norm(a) for a in refs]
                evidence = " " + norm(r["evidence"]) + " "
                missing = all(a and " " + a + " " not in evidence for a in ref)
                any_em = end and norm(text) in ref
                item = dict(
                    doc=r["doc"],
                    condition=k,
                    absent=missing,
                    stop=end,
                    exact=any_em,
                    em=score["em"] if end else 0.0,
                    f1=score["f1"] if end else 0.0,
                    output=sha256(tiles[np.asarray(ids, int)].tobytes()).hexdigest(),
                )
                all_results.append(item)
                stats.append(item)
            subsets = {}
            for g, items in (
                ("all", stats),
                ("absent_from_evidence", [r for r in stats if r["absent"]]),
            ):
                subsets[g] = dict(
                    n=len(items),
                    exact=sum(r["exact"] for r in items),
                    stopped=sum(r["stop"] for r in items),
                    official_f1=100 * np.mean([r["f1"] for r in items]),
                    official_em=100 * np.mean([r["em"] for r in items]),
                )
            report[name][k] = subsets
        packed(PUB / (name + "-rationale-outcomes.json.gz"), all_results)
    dump(
        PUB / "rationale.json",
        dict(eligible=len(rows), excluded=dict(excluded), results=report),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner(), torch.inference_mode():
        main()
