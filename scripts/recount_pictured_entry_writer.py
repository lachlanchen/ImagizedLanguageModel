"""Verify every reported complete answer from the exported bits and source targets."""

import numpy as np
from scripts.evaluate_paired_student import read_streams
from scripts.train_pictured_entry_writer import DEFAULT, OUT, corpus, load, sha
from scripts.prepare_pictured_entry_answers import write_json


def main():
    study = load(OUT / "evaluation.json")
    refs = corpus("development")["answers"]
    n = len(refs)
    counts = {}
    checked = 0
    for cond, arms in study["conditions"].items():
        counts[cond] = {}
        for arm, v in arms.items():
            name = cond + "_" + arm
            assert sha(OUT / "outputs" / (name + ".npz")) == v["stream_sha"]
            outputs = read_streams(name, OUT / "outputs")
            assert len(outputs) == len(v["stopped"]) == n
            ids = []
            for i, (out, stopped) in enumerate(zip(outputs, v["stopped"])):
                expected = (
                    refs[i][:128] + refs[i ^ 4][128:]
                    if cond == "relocated"
                    else refs[i]
                )
                if stopped and out.tobytes() == expected:
                    ids.append(i)
            assert ids == v["exact_rows"] and len(ids) == v["exact"]
            complete = set(ids)
            all_eight = sum(
                all(j in complete for j in range(i, i + 8)) for i in range(0, n, 8)
            )
            assert all_eight == v["all_eight"]
            counts[cond][arm] = len(ids)
            checked += n
    write_json(
        OUT / "recount.json",
        dict(
            checked_streams=checked,
            counts=counts,
            evaluation_sha=sha(OUT / "evaluation.json"),
            dataset_sha=sha(DEFAULT / "report.json"),
        ),
    )


if __name__ == "__main__":
    main()
