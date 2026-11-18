"""Generate independent translations and a shuffled, method-masked reading packet."""

import fcntl
import random
import time
import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file

from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from ilm.visual_lm.ordered_copy_anchors import forced_matches
from scripts.evaluate_paired_student import read_streams, save_streams
from scripts.prepare_phrase_boundary_trial import Rasterizer, load, sha, write_new
from scripts.train_anchor_gap_writer import OUT, DATA, ROOT
from scripts.train_causal_glyph_translator import read_rows

EVIDENCE = ROOT / "references/evidence/anchor_gap_20261119"
ROSTER = ROOT / "references/evidence/anchor_writer_holdout_20261119.json"
CONTROLS = ROOT / "references/evidence/anchor_writer_controls_20261119.json"


def reading(rows, selected, streams, *, evidence=EVIDENCE, extra_chars=""):
    chars = set("".join(r["source"] + r["target"] for r in rows))
    chars.update(extra_chars)
    for r in read_rows("train"):
        chars.update(r["new"] + r["source"])
    render, lex = Rasterizer(), {}
    for ch in sorted(chars):
        try:
            lex.setdefault(render.tile(ch), ch)
        except ValueError:
            continue
    rng = random.Random(167251)
    rng.shuffle(selected)
    packet, mapping, pixels = [], [], []
    for i in selected:
        values = {k: v[i].tobytes() for k, v in streams.items()}
        choices = list(dict.fromkeys(values.values()))
        rng.shuffle(choices)
        texts = []
        for output in choices:
            cells = np.frombuffer(output, np.uint8).reshape(-1, 128)
            pixels.append(cells)
            texts.append("".join(lex.get(bytes(cell), "▧") for cell in cells))
        packet.append(dict(source=rows[i]["source"], choices=texts))
        mapping.append(
            dict(row=i, options={k: choices.index(v) for k, v in values.items()})
        )
    write_new(evidence / "review.json", packet)
    write_new(evidence / "mapping.json", mapping)
    save_streams("review_pixels", pixels, evidence)


def evaluate():
    EVIDENCE.mkdir()
    control = load(CONTROLS)
    assert sha(ROSTER) == control["roster_sha"]
    rows = load(ROSTER)["rows"]
    d = DATA / "controls"
    for name, h in control["streams"].items():
        assert sha(d / (name + ".npz")) == h
    streams = {k: read_streams(k, d) for k in control["streams"]}
    path = OUT / "writer.safetensors"
    fit = load(OUT / "fit.json")
    assert sha(path) == fit["checkpoint_sha256"]
    write_new(EVIDENCE / "fit.json", fit)
    state = load_file(str(path))
    model = AnchorGapWriter(state["landmarks"], state["exemplars"])
    model.load_state_dict(state)
    model.eval().cuda()
    torch.set_num_threads(2)
    records = []
    start = time.monotonic()
    for policy in ("learned", "all", "supplied"):
        outputs, stats = [], []
        for s in range(0, len(rows), 16):
            src = streams["source"][s : s + 16]
            gold = streams["reference"][s : s + 16]
            lengths = torch.tensor(list(map(len, src)), device="cuda")
            physical = np.zeros((len(src), max(map(len, src)), 128), np.uint8)
            aligned = np.zeros(physical.shape[:2], bool)
            for j, (x, y) in enumerate(zip(src, gold)):
                physical[j, : len(x)] = x
                _, pairs = forced_matches(list(map(bytes, x)), list(map(bytes, y)))
                for k, _ in pairs:
                    aligned[j, k] = True
            pics = torch.from_numpy(np.unpackbits(physical, axis=-1)).cuda().bool()
            supplied = (
                torch.from_numpy(aligned).cuda() if policy == "supplied" else None
            )
            generated, counts, stops, selected = model.write(
                pics, lengths, policy=policy, supplied=supplied
            )
            packed = np.packbits(generated.cpu().numpy(), axis=-1)
            selected = selected.cpu().numpy()
            for j, (y, count, stop) in enumerate(
                zip(gold, counts.tolist(), stops.tolist())
            ):
                v = packed[j, :count]
                outputs.append(v)
                stats.append(
                    dict(
                        row=s + j,
                        band=rows[s + j]["length_band"],
                        cells=count,
                        stop=stop,
                        edits=Levenshtein.distance(
                            tuple(map(bytes, v)), tuple(map(bytes, y))
                        ),
                        true_source_positions=int(aligned[j].sum()),
                        selected=int(selected[j].sum()),
                        selected_true=int((aligned[j] & selected[j]).sum()),
                    )
                )
        save_streams(policy, outputs, OUT)
        streams[policy] = outputs
        records.append(
            dict(policy=policy, per_case=stats, sha256=sha(OUT / (policy + ".npz")))
        )
        print(policy, "done", round(time.monotonic() - start, 1), flush=True)
    write_new(
        EVIDENCE / "generation.json",
        dict(
            checkpoint_sha256=sha(path),
            roster_sha256=sha(ROSTER),
            controls_sha256=sha(CONTROLS),
            seconds=time.monotonic() - start,
            streams=records,
        ),
    )
    reading(rows, control["reader_rows"].copy(), streams)


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate()
