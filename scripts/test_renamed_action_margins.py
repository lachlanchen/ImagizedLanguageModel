"""Pool the frozen writer's clause margins under a reversible relabelling of two names."""

import json
from collections import Counter, defaultdict
from time import monotonic

import torch

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.evaluate_individual_clause_writer import corpora
from scripts.pointing_glyph_corpus import Row
from scripts.role_change_corpus import target
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_individual_clause_writer import PUBLIC, WORK, PARENT
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import FACES, Face


def exchange(strip, names):
    pieces = []
    pos = 0
    while pos < len(strip):
        for i, name in enumerate(names):
            if strip.startswith(name, pos):
                pieces.append(names[1 - i])
                pos += len(name)
                break
        else:
            pieces.append(strip[pos:pos + 128])
            pos += 128
    return b"".join(pieces)


def renamed_rows(scenes, face):
    rows = []
    for c in scenes:
        names = [face(n) for n in c["names"]]
        assert all(names) and names[0] != names[1]
        for w, passage in enumerate(c["pages"]):
            gold = target(c, w, False).removesuffix("。").split("，")
            rival = target(c, 1-w, False).removesuffix("。").split("，")
            page = face(passage)
            swapped = exchange(page, names)
            assert swapped != page and exchange(swapped, names) == page
            for lang, questions in c["qs"].items():
                for a, question in enumerate(questions):
                    key = f"{c['key']}/{w}/{lang}/{a}"
                    q = face(question)
                    q2 = exchange(q, names)
                    assert exchange(q2, names) == q
                    answers = [face(s + "。") for s in (gold[a], rival[a])]
                    assert [exchange(s, names) for s in answers] == answers[::-1]
                    rows.extend(Row("renamed", key, swapped, q2, y, b"") for y in answers[::-1])
    return rows


@torch.inference_mode()
def main():
    start = monotonic()
    face = Face(FACES[0])
    sets = corpora()
    original = json.loads((PUBLIC / "clause_odds.json").read_text())
    out = {}
    for arm in ("parent", "independent", "aligned"):
        folder = PARENT if arm == "parent" else WORK / f"clause_{arm}/page"
        net = PointingCausalWriter.restore(folder, "cuda")
        out[arm] = {}
        for group in ("old", "new"):
            path = WORK / f"clause_margins_{arm}_{group}.json"
            assert sha(path) == original[arm][group]["sha256"]
            margins = json.loads(path.read_text())
            rows = renamed_rows(sets[group], face)
            read, bank, pairs, labels = inputs(net.writer, rows)
            assert all(all(v < len(bank) or v in src for v in y)
                       for (src, _), y in zip(pairs, labels, strict=True))
            scores = []
            for i in range(0, len(rows), 16):
                values, _ = score_batch(net, read, bank, pairs[i:i+16], labels[i:i+16])
                scores.extend(values)
            cells = defaultdict(list)
            details = {}
            for row, g, r in zip(rows[::2], scores[::2], scores[1::2], strict=True):
                s = margins[row.key]
                pooled = (s + (g-r)) / 2
                k, _, lang, _ = row.key.rsplit("/", 3)
                cells[k, lang].append((s, pooled))
                details[row.key] = [s, g-r, pooled]
            counts = Counter()
            for z in cells.values():
                assert len(z) == 4
                counts.update(contrasts=1, requests=len(z),
                              raw=sum(s>0 for s,_ in z), pooled=sum(t>0 for _,t in z),
                              original_complete=all(s>0 for s,_ in z),
                              pooled_complete=all(t>0 for _,t in z))
            assert counts["original_complete"] == sum(c["complete"] for c in original[arm][group]["counts"].values())
            out[arm][group] = dict(counts=counts, cells=details)
        del net
    peak = torch.cuda.max_memory_allocated()
    elapsed = monotonic() - start
    record(PUBLIC / "renamed_action_scores.json", dict(arms=out, seconds=elapsed, peak_bytes=peak))
    assert elapsed < 300 and peak < 20 * 1024**3


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        main()
