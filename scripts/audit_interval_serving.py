"""Check batch-one and batched output pixels and record the benchmark GPU placement."""

import json
import re
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.interval_teacher_corpus import PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha


def audit():
    run = WORK / "latency"
    scoring = json.loads((PUBLIC / "result.json").read_text())["arms"]
    placement = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", (run / "server.log").read_text())
    if not placement:
        raise ValueError("No Qwen placement receipt")
    differences = {}
    for arm in ("replay", "aligned", "distilled"):
        a = WORK / "evaluation" / arm / "fresh.npz"
        b = run / (arm + ".npz")
        assert sha(a) == scoring[arm]["scores"]["fresh"]["sha256"]
        assert sha(b) == json.loads((run / (arm + ".json")).read_text())["stream_hash"]
        pixels, stops = load_streams(a)
        single, ends = load_streams(b)
        assert len(pixels) == len(stops) == len(single) == len(ends) == 196
        differences[arm] = [i for i, (p,q,s,t) in enumerate(zip(pixels,single,stops,ends,strict=True)) if p != q or bool(s) != bool(t)]
    record(PUBLIC / "serving_audit.json", dict(cases=196, differing_indices=differences, qwen_gpu_layer_counts=placement))


if __name__ == "__main__":
    audit()
