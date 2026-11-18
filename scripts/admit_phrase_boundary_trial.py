"""Freeze natural contrasts, original multi-cell challenges and untouched regressions."""

from scripts.prepare_phrase_boundary_trial import (
    EVIDENCE, EVENT, INITIAL, INVENTORY, OLD, OUT, PREVIOUS, ROOT, Rasterizer, clauses,
    load, read_rows, sha, write_new,
)
from scripts.prepare_predicate_likelihood import PREDICATES
from scripts.prepare_visual_boundary_candidates import OTHER

# Sense and surrounding proposition were read before inspecting model outputs.
# The count of fresh natural contrasts is limited by the remaining archive.
NATURAL_PAIRS = [(0, 43), (1, 36), (7, 16), (10, 19), (2, 29), (3, 21)]
SINGLE = [6, 9, 12, 23, 24, 30, 31, 32, 37, 40, 44, 46, 49, 50, 52, 53, 55]
AUTHORED = [
    ("去", "两营相去九里", "两座军营相距九里", "距离", "将军辞众而去", "将军告别众人离开了", "离开"),
    ("去", "东门与桥相去八十步", "东门和桥相距八十步", "距离", "舟人舍舟相携而去", "船夫们弃船结伴离开", "离开"),
    ("举", "县令荐举乡中贤者", "县令推荐乡里的贤人", "推荐", "县令徐举案上之杯", "县令慢慢举起桌上的杯子", "举起"),
    ("伐", "戍卒入谷伐榆", "守军进山谷砍伐榆树", "砍伐", "戍卒奉命伐燕", "守军奉命攻打燕国", "攻打"),
]


def run():
    proposed = load(OUT / "proposed.json")
    screen = load(OUT / "screen.json")
    assert sha(OUT / "proposed.json") == screen["proposed_sha"]
    held = []
    for pair, group in enumerate(NATURAL_PAIRS):
        a, b = [proposed[i] for i in group]
        assert a["predicate"] == b["predicate"] and a["gloss"] != b["gloss"]
        held.extend(dict(proposed[i], cohort="archive_pair", pair=pair) for i in group)
    render = Rasterizer()
    def ink(text):
        return b"".join(map(render.tile, text))
    new_ink = {ink(x) for _, a, b, _, c, d, _ in AUTHORED for x in (a, b, c, d)}
    assert len(new_ink) == len(AUTHORED) * 4
    lengths = {len(x) // 128 for x in new_ink}
    def check(row):
        for k in ("source", "new", "old", "reference", "target"):
            if not isinstance(row.get(k), str):
                continue
            for s in clauses(row[k]):
                if len(s) in lengths:
                    try:
                        image = ink(s)
                    except ValueError:
                        continue
                    assert image not in new_ink, s
    for split in ("train", "heldout"):
        for row in read_rows(split):
            check(row)
    for name, roster, _ in OLD:
        for row in load(ROOT / "artifacts" / name / (roster + ".json")):
            check(row)
    for folder, names in [(EVENT, ("train", "roles", "broad")), (INITIAL, ("train", "proposed", "test")),
                          (PREVIOUS, ("train", "proposed", "test")), (OUT, ("proposed",))]:
        for name in names:
            for row in load(folder / (name + ".json")):
                check(row)
    for row in load(INVENTORY)["records"]:
        check(row)
    for j, (p, s, t, g, s2, t2, g2) in enumerate(AUTHORED, len(NATURAL_PAIRS)):
        for x, y, sense in [(s, t, g), (s2, t2, g2)]:
            held.append(dict(source=x, target=y, chapter=f"authored/{j}", predicate=p,
                             gloss=sense, cohort="authored_pair", pair=j))
    held.extend(dict(proposed[i], cohort="archive_single", pair=None) for i in SINGLE)
    assert len({r["source"] for r in held}) == len(held)
    assert all(r["gloss"] in (PREDICATES | OTHER)[r["predicate"]] for r in held)
    EVIDENCE.mkdir()
    write_new(EVIDENCE / "test.json", held)
    write_new(EVIDENCE / "admission.json", dict(screen, test_sha=sha(EVIDENCE / "test.json"),
              broad_sha=sha(EVENT / "broad.json"), natural_pairs=6, authored_pairs=4,
              archive_single=len(SINGLE), training_exclusions=19))


if __name__ == "__main__":
    run()
