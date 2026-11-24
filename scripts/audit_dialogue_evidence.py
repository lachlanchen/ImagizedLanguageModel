"""Recount published visual dialogue scores from stopped autonomous response strings."""

from collections import defaultdict
import gzip
import json
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_teacher_stories import new_stories, absent
from scripts.qualify_dialogue_teacher import PUB, WORK, sha
from scripts.evaluate_optical_passage_qa import normalized


def audit():
    registry = read_data()
    result = json.loads((PUB / "replication.json").read_text())
    files = answers = 0
    for group in ("long", "longer"):
        ds, markers, evaluator = new_stories(registry, group)
        for seed, entry in result["by_seed"].items():
            for arm in ("plain", "human"):
                recorded = entry["groups"][group][arm]
                path = PUB / "responses" / seed / f"{group}-{arm}.json.gz"
                if sha(path) != recorded["outputs_sha256"]:
                    raise ValueError(path)
                counts = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
                for row, marker, out in zip(
                    ds["rows"],
                    markers,
                    json.loads(gzip.decompress(path.read_bytes())),
                    strict=True,
                ):
                    refs = evaluator.gold_data[marker]
                    groups = ["all"]
                    if absent(refs, (ds["pages"][row[1]], row[3], row[4])):
                        groups.append("absent")
                    for name, r in out.items():
                        stopped = r["stop"]
                        text = r["answer"]
                        score = evaluator.compute_turn_score(*marker, text)
                        exact = any(normalized(a) == normalized(text) for a in refs)
                        for cohort in groups:
                            c = counts[cohort, name]
                            c[0] += 1
                            c[1] += stopped
                            c[2] += stopped and exact
                            if stopped:
                                c[3] += score["em"]
                                c[4] += score["f1"]
                        answers += 1
                for (cohort, key), (n, stops, exact, em, f1) in counts.items():
                    saved = recorded[cohort][key]
                    if (
                        (n, stops, exact)
                        != (saved["turns"], saved["stopped"], saved["any_exact"])
                        or abs(em / n - saved["em"]) > 1e-12
                        or abs(f1 / n - saved["f1"]) > 1e-12
                    ):
                        raise ValueError((path, cohort, key))
                files += 1
    for prefix in ("", "current_last"):
        path = WORK / prefix / "qualification.jsonl"
        result = json.loads((PUB / prefix / "qualification.json").read_text())
        if result["admitted"]:
            raise ValueError("Teacher was recorded as admitted")
        if path.exists() and (
            result["decisions"] != sha(path)
            or len(path.read_text().splitlines()) != result["requests"]
        ):
            raise ValueError(path)
    print(
        f"Verified {files} decision files and recounted {answers} actual model responses."
    )


if __name__ == "__main__":
    audit()
