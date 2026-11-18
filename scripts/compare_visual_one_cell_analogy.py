"""Measure how much of the local transfer probe an image analogy can explain."""

from rapidfuzz.distance import Levenshtein
from scripts.train_screened_visual_targets import EVIDENCE, OUT, load, write_new
from scripts.evaluate_paired_student import read_streams


def substitute(query, examples):
    eligible = []
    for source, target in examples:
        if (
            len(source) != len(query)
            or Levenshtein.distance(source, query, score_cutoff=1) != 1
        ):
            continue
        slot = next(i for i in range(len(source)) if source[i] != query[i])
        old = source[slot]
        if source.count(old) == 1 and target.count(old) == 1:
            index = target.index(old)
            eligible.append(target[:index] + (query[slot],) + target[index + 1 :])
    return eligible[0] if len(eligible) == 1 else None


def run():
    def read(name, folder):
        return [tuple(map(bytes, v)) for v in read_streams(name, OUT / folder)]

    examples = list(
        zip(
            read("source", "training_replay"),
            read("screened_target", "training_replay"),
        )
    )
    queries = read("source", "changes")
    expected = read("target", "changes")
    natural = read("source", "evaluation")
    reg = load(EVIDENCE / "change_probes.json")["cases"]
    results = {}
    for group in ("shift", "number"):
        ids = [i for i, v in enumerate(reg) if v["kind"] == group]
        pred = [substitute(queries[i], examples) for i in ids]
        results[group] = dict(
            count=len(ids),
            emitted=sum(v is not None for v in pred),
            exact=sum(p == expected[i] for p, i in zip(pred, ids)),
        )
    archive = load(EVIDENCE / "registration.json")["archive"]
    results["archive_emitted"] = sum(
        substitute(x, examples) is not None for x in natural[:archive]
    )
    results["authored_emitted"] = sum(
        substitute(x, examples) is not None for x in natural[archive:]
    )
    write_new(EVIDENCE / "one_cell_analogy.json", results)
    print(results)


if __name__ == "__main__":
    run()
