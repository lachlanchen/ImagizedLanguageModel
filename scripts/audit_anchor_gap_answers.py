"""Check the evidence for learned translations against the complete physical training corpus."""

import numpy as np
from scripts.evaluate_anchor_gap_writer import (
    DATA,
    OUT,
    EVIDENCE,
    load,
    sha,
    write_new,
    read_streams,
)


def run():
    scored = load(EVIDENCE / "scores.json")["faithful_rows"]
    finite = load(EVIDENCE / "finite_scores.json")["faithful"]
    sources = read_streams("source", DATA / "controls")
    generated = read_streams("learned", OUT)
    path = DATA / "training.npz"
    assert sha(path) == load(EVIDENCE / "fit.json")["training_sha256"]
    with np.load(path) as z:
        ix = {bytes(v): i for i, v in enumerate(z["cells"])}
        seq, offsets = z["target"], z["y_offsets"]

    def contains(a, b):
        return any(a[i : i + len(b)] == b for i in range(len(a) - len(b) + 1))

    credited = []
    for row in scored["learned"]:
        output = [ix.get(bytes(v), -1) for v in generated[row]]
        whole = within = 0
        for a, b in zip(offsets, offsets[1:]):
            dst = seq[a:b].tolist()
            whole += dst == output
            within += contains(dst, output)
        credited.append(
            dict(
                row=row,
                length=len(output),
                identical_to_source=np.array_equal(generated[row], sources[row]),
                training_exact_complete_targets=whole,
                training_rows_containing_complete_output=within,
            )
        )
    nonref = set(finite)
    for k, v in scored.items():
        if k != "reference":
            nonref.update(v)
    write_new(
        EVIDENCE / "generation_audit.json",
        dict(
            credited=credited,
            training_rows=len(offsets) - 1,
            any_generated_or_source_faithful=sorted(nonref),
            reviewed=24,
        ),
    )


if __name__ == "__main__":
    run()
