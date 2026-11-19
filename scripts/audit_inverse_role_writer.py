"""Recount original stopped images and locate the first error in Chinese who-answers."""
from collections import Counter
from functools import lru_cache
import hashlib
import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.check_role_writer_historical import FORMS
from scripts.train_inverse_role_writer import DRCD, PUBLIC, ROOT, WHO, WORK, load, sha, write_json
from scripts.prepare_pictured_entry_answers import Rasterizer


def source_start(passage, answer):
    if not answer:
        return None
    start = passage.find(answer)
    while start >= 0 and start % 128:
        start = passage.find(answer, start + 1)
    return None if start < 0 else start // 128


def audit():
    result = load(WORK / "result.json")
    registration = load(PUBLIC / "registration.json")
    assert sha(PUBLIC / "registration.json") == result["registration"]
    assert all(sha(ROOT / path) == digest for path, digest in registration.items())
    render = Rasterizer()
    @lru_cache(maxsize=4096)
    def ink(text):
        return b"".join(render.tile(c) for c in text)
    raw = ROOT / "data/raw/drcd/DRCD_dev.json"
    dataset = load(raw)["data"]
    records = {qa["id"]: (para["context"], qa) for article in dataset
               for para in article["paragraphs"] for qa in para["qas"]}
    ids = load(DRCD / "development/question_ids.json")
    with np.load(WHO / "cohort.npz") as z:
        who, pairs = z["development"], z["development_pairs"]
    forms = load(FORMS)
    by_hash = {forms[c]["ink_sha256"]: c for c in "晋燕蔡魏秦楚"}
    for c in by_hash.values():
        assert sha(ROOT.parent / "incoder" / forms[c]["source_local_path"]) == forms[c]["sha256"]

    output = {}
    for arm, scores in result["arms"].items():
        def read(test):
            return archived(WORK / arm / (test + ".npz"), scores[test]["sha256"])
        rec = {}
        for test, names in (("historic_identify", "秦楚"), ("historic_four", "晋燕蔡魏")):
            if test not in scores:
                continue
            images, stops = read(test)
            predicted = [by_hash.get(hashlib.sha256(x).hexdigest()) if end else None
                         for x, end in zip(images, stops, strict=True)]
            n = len(names)
            expected = [names[(i // n + i % n) % n] for i in range(len(images))]
            ok = np.equal(predicted, expected).reshape(-1,n,n)
            complete = int(ok.all(axis=1).sum())
            assert int(ok.sum()) == scores[test]["exact"]
            assert complete == scores[test]["both_sources" if n == 2 else "all_rotations"]
            source_slots = Counter()
            for i, actual in enumerate(predicted):
                key = str((names.index(actual) - i // n) % n + 1) if actual in list(names) else "other"
                source_slots[key] += 1
            rec[test] = dict(exact=int(ok.sum()), complete=complete, selected_clause=dict(source_slots),
                             first_orientation=predicted[:n], second_orientation=predicted[n:2*n])

        if "drcd" in scores:
            fit = result["fits"][arm]
            assert sha(WORK / arm / "final.safetensors") == fit["sha256"]
            images, stops = read("drcd")
            truths = []
            for key in ids:
                context, qa = records[key]
                truths.append({ink(a["text"]) for a in qa["answers"] if a["text"] and a["text"] in context})
            ok = np.array([bool(s and x in valid) for s,x,valid in zip(stops,images,truths,strict=True)])
            values = [int(ok.sum()),int(ok[who].sum()),int(ok[who][pairs].all(1).sum())]
            assert values == [scores["drcd"][k] for k in ("exact","who","who_pairs")]
            counts = Counter()
            examples = {}
            for i in who:
                question_id = ids[i]
                context, qa = records[question_id]
                pix = images[i]
                candidates = truths[i]
                start = source_start(ink(context),pix)
                if ok[i]:
                    label = "exact"
                elif not pix:
                    label = "empty"
                elif any(pix[:128] == ref[:128] for ref in candidates):
                    label = "wrong_after_correct_first_glyph"
                else:
                    label = "wrong_first_glyph"
                counts[label] += 1
                if not ok[i] and start is not None:
                    counts["incorrect_contiguous_source_span"] += 1
                if not ok[i] and not stops[i]:
                    counts["incorrect_unstopped"] += 1
                if start is not None and label not in examples:
                    emitted = context[start:start+len(pix)//128]
                    assert ink(emitted)==pix
                    examples[label] = dict(id=question_id, question=qa["question"], emitted=emitted,
                                           references=[a["text"] for a in qa["answers"]], stopped=bool(stops[i]))
            rec["drcd"] = dict(exact=values[0], who=values[1], pairs=values[2], who_counts=dict(counts),
                               first_examples=examples)
        output[arm] = rec
    path = PUBLIC / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    write_json(path,dict(arms=output, raw_development=sha(raw), scores_sha256=sha(WORK/"result.json"),
                         program_sha256=sha(__file__)))


if __name__ == "__main__":
    audit()
