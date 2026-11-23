"""Compare the stopped correct clause to the same clause naming the other participant."""

import json
from collections import defaultdict, Counter
import torch
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter
from scripts.audit_book_sequence_probability import score_batch
from scripts.pointing_glyph_corpus import Row
from scripts.role_change_corpus import target
from scripts.evaluate_individual_clause_writer import corpora
from scripts.train_individual_clause_writer import PUBLIC, WORK, PARENT
from scripts.train_pointing_causal_writer import inputs
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


@torch.inference_mode()
def run():
    face = Face(FACES[0])
    report = {}
    sets = corpora()
    for arm in ("parent", "independent", "aligned"):
        base = PARENT if arm == "parent" else WORK / f"clause_{arm}" / "page"
        net = PointingCausalWriter.restore(base, "cuda")
        results = {}
        for group in ("old", "new", "credits"):
            rows = []
            for c in sets[group]:
                ys = [
                    [
                        face(clause + "。")
                        for clause in target(c, w, False).removesuffix("。").split("，")
                    ]
                    for w in (0, 1)
                ]
                for w, p in enumerate(c["pages"]):
                    for lang, qs in c["qs"].items():
                        for a, q in enumerate(qs):
                            key = f"{c['key']}/{w}/{lang}/{a}"
                            rows.extend(
                                Row(group, key, face(p), face(q), ys[v][a], b"")
                                for v in (w, 1 - w)
                            )
            read, bank, pairs, labels = inputs(net.writer, rows)
            keep = [
                i
                for i, ((src, _), y) in enumerate(zip(pairs, labels, strict=True))
                if all(v < len(bank) or v in src for v in y)
            ]
            vals = [None] * len(rows)
            for j in range(0, len(keep), 20):
                ix = keep[j : j + 20]
                scores, _ = score_batch(
                    net, read, bank, [pairs[i] for i in ix], [labels[i] for i in ix]
                )
                for i, s in zip(ix, scores, strict=True):
                    vals[i] = s
            detail = {}
            by = defaultdict(Counter)
            flags = defaultdict(list)
            for r, s, t in zip(rows[::2], vals[::2], vals[1::2], strict=True):
                assert (s is None) == (t is None)
                d = None if s is None else s - t
                k, w, lang, a = r.key.rsplit("/", 3)
                detail[r.key] = d
                ok = d is not None and d > 0
                by[lang].update(requests=1, preferred=ok, unsupported=s is None)
                flags[k, lang].append(ok)
            for (k, lang), x in flags.items():
                assert len(x) == 4
                by[lang].update(contrasts=1, complete=all(x))
            file = WORK / f"clause_margins_{arm}_{group}.json"
            record(file, detail)
            results[group] = dict(counts=by, sha256=sha(file))
        report[arm] = results
        del net
    record(PUBLIC / "clause_odds.json", report)


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        run()
