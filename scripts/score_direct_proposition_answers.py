"""Count complete direct answers and accepted statements from saved human-question results."""
import json
from collections import Counter

from scripts.evaluate_grounded_propositions import load_streams
from scripts.evaluate_human_proposition_questions import cases
from scripts.grounded_proposition_corpus import PUBLIC, WORK, record, sha


def score():
    rows, metadata, _, _ = cases()
    results = {}
    for condition in ("predicted", "oracle"):
        path = WORK / f"evidence_human_{condition}.npz"
        evaluation = json.loads((PUBLIC / "evidence_result.json").read_text())
        if sha(path) != evaluation["evidence"]["human_" + condition]["sha256"]:
            raise ValueError("Saved image sequence failed verification")
        answers, stops = load_streams(path)
        readings = json.loads((WORK / f"evidence_human_{condition}_readings.json").read_text())
        counts = Counter()
        for row, meta, response, stopped, reading in zip(rows, metadata, answers, stops, readings, strict=True):
            if meta["key"] != reading["key"]:
                raise ValueError("Reading does not match the input")
            short = stopped and response == row.y
            counts["examples"] += 1
            counts["exact_direct_bitmap"] += short
            counts["accepted_complete_statement"] += reading["statement"]
            counts["either"] += bool(short or reading["statement"])
        results[condition] = dict(counts)
    record(PUBLIC / "direct_answer_scores.json", results)
    print(json.dumps(results))


if __name__ == "__main__":
    score()
