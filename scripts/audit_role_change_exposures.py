"""Check source and answer exposure on the actual two training schedules."""

from collections import Counter, defaultdict
from hashlib import sha256
import json

from scripts.prepare_role_change_actions import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record
from scripts.train_role_change_writer import training
from scripts.train_typeface_writer import Face, FACES


def audit():
    face = Face(FACES[0])
    totals = {}
    for arm in ("independent", "aligned"):
        pages = sha256()
        conditional = defaultdict(Counter)
        by_scene = defaultdict(Counter)
        sizes = sha256()
        for row in training(arm, face):
            for value in (row.task.encode(), row.key.encode(), row.p, row.q):
                pages.update(len(value).to_bytes(4, "big"))
                pages.update(value)
            y = sha256(row.y).digest()
            sizes.update(len(row.y).to_bytes(4, "big"))
            if row.task in ("book", "historical"):
                by_scene[row.task, row.key][y] += 1
            else:
                scene, world, language, order_and_repeat = row.key.rsplit("/", 3)
                grouping = (scene, language, order_and_repeat[0])
                by_scene[grouping][y] += 1
                conditional[grouping, world][y] += 1
        counts = Counter(tuple(sorted(a.values())) for a in conditional.values())
        expected = (1, 1) if arm == "independent" else (2,)
        assert counts == {expected: 22304}, counts
        totals[arm] = (pages.digest(), sizes.digest(), by_scene)
        assert json.loads((WORK / arm / "corpus.json").read_text())["rows"] == 49340
    assert totals["aligned"] == totals["independent"]
    record(
        PUBLIC / "exposure_audit.json",
        dict(
            presentations=49340,
            edited_inputs=22304,
            training_scenes=2788,
            replay=4732,
            ordered_inputs_sha=pages.hexdigest(),
            target_lengths_sha=sizes.hexdigest(),
            equal_answers_per_scene_order_and_language=True,
            replay_labels_identical=True,
        ),
    )


if __name__ == "__main__":
    audit()
