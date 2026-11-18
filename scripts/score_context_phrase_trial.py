"""Grade supplied alternatives and unprompted translations under the registered rule."""

import csv
from scripts.reserve_context_phrase_trial import EVIDENCE, load, write_new
from scripts.train_context_phrase_trial import sha


def paired(scores):
    strict = []
    interaction = []
    sides = 0
    for i, ((aa, ab), (ba, bb)) in enumerate(scores):
        sides += int(aa > ab) + int(bb > ba)
        if aa > ab and bb > ba:
            strict.append(i)
        if aa - ab + bb - ba > 0:
            interaction.append(i)
    return dict(strict=strict, sides=sides, interaction=interaction)


def main():
    folder = EVIDENCE / "reading"
    with (folder / "credits.tsv").open() as f:
        grades = list(csv.DictReader(f, delimiter="\t"))
    packet = load(folder / "review.json")
    mapping = load(folder / "mapping.json")
    result = load(EVIDENCE / "evaluation.json")
    natural = set(load(EVIDENCE / "reading_roster.json"))
    pairs = result["pair_rows"]
    selected = natural.union(*pairs)
    assert len(grades) == len(packet) == len(mapping) == len(selected)
    assert {r["row"] for r in mapping} == selected
    decisions = {}
    for method, v in result["methods"].items():
        whole, detail = set(), set()
        for j, (m, p, g) in enumerate(zip(mapping, packet, grades)):
            assert int(g["j"]) == j
            for label, passed in (("whole", whole), ("detail", detail)):
                credits = {int(x) for x in g[label].split(",") if x != "-"}
                assert credits <= set(range(len(p["choices"])))
                if m["options"][method] in credits:
                    passed.add(m["row"])
        decisions[method] = dict(
            faithful=sorted(whole & natural),
            both_whole=[i for i, p in enumerate(pairs) if set(p) <= whole],
            both_detail=[i for i, p in enumerate(pairs) if set(p) <= detail],
            archive_edits=sum(v["edits"][:192]),
            cap=sum(not stopped for stopped in v.get("stops", [])),
            train_starts=sum(sum(p) for p in v.get("free_training_starts", [])),
        )
    forced = {arm: {group: paired(ms) for group, ms in groups.items()}
              for arm, groups in result["phrase_scores"].items()}
    write_new(EVIDENCE / "scores.json", dict(
        free=decisions, forced=forced,
        registered_success=(
            len(forced["crossed"]["test"]["strict"]) > len(forced["rehearsal"]["test"]["strict"])
            and len(decisions["crossed"]["faithful"]) > len(decisions["rehearsal"]["faithful"])
        ),
        frozen_grades_sha=sha(folder / "credits.tsv"),
        packet_sha=sha(folder / "review.json"), mapping_sha=sha(folder / "mapping.json"),
        evaluation_sha=sha(EVIDENCE / "evaluation.json"),
    ))


if __name__ == "__main__":
    main()
