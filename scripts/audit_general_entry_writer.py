"""Score saved glyphs independently and identify which pictured entry the writer used."""
from collections import Counter
import json

import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.compare_traditional_passage_training import TraditionalPassages
from scripts.evaluate_same_passage_writer import compare_cluster
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_general_entry_writer import AUTHORED, ENTRIES, PARENT, ROOT, WORK
from scripts.train_grounded_glyph_writer import Split
from scripts.train_paired_passage import unpack_sequences
from scripts.train_same_passage_writer import PAIRS

EVIDENCE = ROOT / "references/evidence/general_entry_writer_20261119"


def recount():
    result = load(WORK / "result.json")
    folder = ENTRIES / "development"
    meta = load(folder / "index.json")
    train = load(ENTRIES / "train/index.json")
    forms = {m["raster"] for m in meta}
    assert len(forms) == 568 and forms.isdisjoint(m["raster"] for m in train)
    assert {m["family"] for m in meta}.isdisjoint(m["family"] for m in train)
    bank = np.load(folder / "pixels.npy")
    targets = [bank[y].tobytes() for y in unpack_sequences(folder / "answers.npz")]
    cards = [{bytes(bank[i]) for i in seq} for seq in unpack_sequences(folder / "cards.npz")]
    memory = {bytes(x) for x in np.load(PARENT / "memory.npy")}
    by_card = [dict() for _ in cards]
    for m, y in zip(meta, targets, strict=True):
        form = y[-256:-128]
        card = m["card"]
        assert form in cards[card] and form not in memory
        by_card[card][form] = (m["head"], m["stage"])
    assert all(len(v) == 4 for v in by_card)
    authored = [json.loads(l)["row"] for l in AUTHORED.read_text().splitlines()]
    pairs = np.load(PAIRS / "development.npy")
    ids = np.unique(pairs[:, 1:3])
    dr = TraditionalPassages("development")
    refs = {i: {dr.bank[dr.passages[dr.groups[i]][l:r]].tobytes()
                for l, r in dr.intervals[i]} for i in ids}
    du = Split("development")
    by_arm, diagnoses = {}, {}
    for arm, scores in result["arms"].items():
        by_arm[arm] = {}
        for mode, score in scores.items():
            key = arm + "/" + mode + ".npz"
            streams, end = archived(WORK / key, result["archives"][key])
            assert int(end.sum()) == score["stopped"] and len(end) == score["records"]
            if mode == "drcd":
                ok = {i: bool(e and x in refs[i]) for i, x, e in zip(ids, streams, end, strict=True)}
                assert sum(ok.values()) == score["exact"]
                assert sum(ok[i] and ok[j] for i, j in pairs[:, 1:3]) == score["both"]
                continue
            if mode == "dureader":
                correct = [bool(e and x == du.tiles[y].tobytes())
                           for x, y, e in zip(streams, du.y, end, strict=True)]
                assert sum(correct) == score["exact"]
                for group in set(du.groups):
                    assert sum(c for c, g in zip(correct, du.groups) if group == g) == score[group + "_exact"]
                continue
            rows = authored if mode == "independent" else range(len(meta))
            ok = np.array([e and x == targets[r] for x, e, r in zip(streams, end, rows, strict=True)])
            assert sum(ok) == score["exact"]
            assert ok.reshape(-1, 4 if mode == "independent" else 8).all(1).sum() == score["all_in_group"]
            diagnosis = Counter()
            for x, e, r in zip(streams, end, rows, strict=True):
                m = meta[r]
                terminal = x[-256:-128]
                source = by_card[m["card"]].get(terminal) if e else None
                if source is None:
                    condition = "outside_four_forms_or_unstopped"
                else:
                    h, s = source
                    condition = ("same_head" if h == m["head"] else "other_head") + (
                            "_right_stage" if s == m["stage"] else "_other_stage")
                diagnosis[condition] += 1
                diagnosis["correct_opening_head"] += int(x[:128] == targets[r][:128])
            assert diagnosis["same_head_right_stage"] == score["terminal_form"]
            diagnoses[arm + "/" + mode] = dict(diagnosis)
            by_arm[arm][mode] = ok
    intervals = {}
    for mode, paired in by_arm["paired"].items():
        rows = authored if mode == "independent" else range(len(meta))
        ci = compare_cluster(paired, by_arm["scrambled"][mode], [meta[r]["card"] for r in rows])
        ci["cards"] = ci.pop("passages")
        intervals[mode] = ci
    EVIDENCE.mkdir(exist_ok=True)
    write_json(EVIDENCE / "result.json", result)
    write_json(EVIDENCE / "audit.json", dict(contrasts=intervals, form_selection=diagnoses,
               held_out_forms=len(forms), files_recounted=len(result["archives"]), program_sha256=sha(__file__)))
    print(diagnoses["paired/simple"], intervals["simple"])


if __name__ == "__main__":
    recount()
