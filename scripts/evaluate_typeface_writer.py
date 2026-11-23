"""Save every physical response from the held-typeface and retention comparisons."""

import argparse
import json
from time import monotonic

import torch
from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_interval_teacher import trials
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import WORK, PUBLIC, BASE
from scripts.typeface_writer_cases import sets

CONTROLS = (
    "cmrc",
    "drcd",
    "historical",
    "historical_rephrased",
    "yan",
    "open_single_span",
    "open_distributed_ink",
    "open_absent_content_ink",
)


@torch.inference_mode()
def evaluate(arms):
    styled, sources = sets()
    groups = trials()
    cohorts = styled | {k: groups[k] for k in CONTROLS}
    record(
        WORK / "test_ids.json",
        {k: [c.question_id for c in cases] for k, cases in styled.items()},
    )
    record(
        PUBLIC / "sources.json",
        sources | {"test_ids_sha256": sha(WORK / "test_ids.json")},
    )
    torch.set_num_threads(2)
    for arm in arms:
        folder = WORK / "evaluation" / arm
        folder.mkdir(exist_ok=False, parents=True)
        path = BASE if arm in ("parent", "map_only") else WORK / arm / "page"
        weights = json.loads((path.parent / "page_fit.json").read_text())["weights"]
        assert all(sha(path / n) == h for n, h in weights.items())
        now = monotonic()
        model = PreparedGlyphResponder(path, "cuda")
        if arm == "map_only":
            assert (
                sha(WORK / "typeface.safetensors")
                == json.loads((PUBLIC / "correspondence.json").read_text())["sha256"]
            )
            model.net.writer.attach_typeface_map(WORK / "typeface.safetensors")
        pairs = sorted(
            {(c.p, c.q) for cases in cohorts.values() for c in cases},
            key=lambda pair: (len(pair[0]) + 2 * len(pair[1]), pair),
        )
        outputs = {}
        torch.cuda.reset_peak_memory_stats()
        for i in range(0, len(pairs), 16):
            batch = pairs[i : i + 16]
            ys = model.answer_many([(tensor(p), tensor(q)) for p, q in batch])
            outputs.update(
                (key, (y.numpy().tobytes(), bool(stop)))
                for key, (y, stop) in zip(batch, ys, strict=True)
            )
        files = {}
        for name, cases in cohorts.items():
            ys, ends = zip(*(outputs[c.p, c.q] for c in cases), strict=True)
            f = folder / (name + ".npz")
            store_streams(f, ys, ends)
            files[name] = sha(f)
        record(
            folder / "streams.json",
            dict(
                unique=len(pairs),
                seconds=monotonic() - now,
                peak_bytes=torch.cuda.max_memory_allocated(),
                files=files,
            ),
        )
        print(arm, len(pairs), round(monotonic() - now, 1), flush=True)
        del model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "arms", nargs="+", choices=("parent", "map_only", "replay", "mapped")
    )
    with gpu_owner():
        evaluate(ap.parse_args().arms)
