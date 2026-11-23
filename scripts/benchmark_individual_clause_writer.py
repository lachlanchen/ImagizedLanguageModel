"""Measure warm image inference and check outputs against the saved semantic reading set."""

from statistics import median
from time import monotonic
import json
import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.measure_writer_energy import snapshot, consumed
from scripts.read_individual_clause_replies import samples
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_individual_clause_writer import PUBLIC, WORK
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face, FACES


def run():
    folder = WORK / "clause_aligned/page"
    net = PreparedGlyphResponder(folder, "cuda")
    face = Face(FACES[0])
    source = WORK / "aligned_new_separate.npz"
    assert (
        sha(source)
        == json.loads((PUBLIC / "aligned_evaluation.json").read_text())["groups"][
            "new"
        ]["separate"]["images_sha"]
    )
    xs, stops = load_streams(source)
    jobs = []
    for c, w, lang, a, i, _ in samples():
        p = tensor(face(c["pages"][w]))
        qs = c["qs"][lang]
        jobs.append(
            ([(p, tensor(face(qs[j]))) for j in (a, 1 - a)], (i + a, i + 1 - a))
        )
    net.answer_many(jobs[0][0])
    torch.cuda.synchronize()
    latency = []
    mismatches = 0
    energy = snapshot()
    for pairs, idx in jobs:
        start = monotonic()
        ys = net.answer_many(pairs)
        torch.cuda.synchronize()
        latency.append(monotonic() - start)
        for (im, e), i in zip(ys, idx, strict=True):
            if im.numpy().tobytes() != xs[i] or bool(e) != stops[i]:
                mismatches += 1
    used = consumed(energy, snapshot())
    record(
        PUBLIC / "warm_inference.json",
        dict(
            requests=len(jobs),
            replies=2 * len(jobs),
            image_mismatches=mismatches,
            seconds_sum=sum(latency),
            seconds_median=median(latency),
            device_energy=used,
        ),
    )
    if mismatches:
        raise ValueError("Warm replay differs from the reviewed images")


if __name__ == "__main__":
    with gpu_owner():
        torch.set_num_threads(2)
        run()
