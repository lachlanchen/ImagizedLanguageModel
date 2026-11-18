"""Test the trained clause writer with the correct preceding pictures and one real error."""

from collections import Counter

import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_translator import (
    CausalGlyphTranslator,
    EXEMPLAR,
    DRAW,
    END,
)
from ilm.visual_lm.idiom_ink import visual_input
from scripts.evaluate_causal_glyph_translator import select
from scripts.train_causal_glyph_translator import OUT, tiles, sha, write_new
from scripts.align_visual_translation import ROOT, load, require
import fcntl


@torch.inference_mode()
def predict(net, source, lens, history):
    memory, keys, mask, h = net.read(visual_input(source, net.landmarks), lens)
    x = net.embed(visual_input(history, net.landmarks))
    q, _ = net.history(torch.cat((net.start.expand(len(x), 1, -1), x), 1), h)
    state, a, _ = net.plan_steps(
        q, memory, keys, mask, lens, lens.new_full(lens.shape, -0.65, dtype=torch.float)
    )
    modes = net.route(state).argmax(-1)
    rows = torch.arange(len(lens), device=lens.device)[:, None]
    out = source[rows, a.argmax(-1)].clone()
    common = modes == EXEMPLAR
    out[common] = net.exemplars[
        net.prototype_scores(state[common], net.image_keys()).argmax(-1)
    ]
    paint = modes == DRAW
    if paint.any():
        bits = net.draw.choose(state[paint])
        v = net.motor(bits) > 0
        v[:, net.landmarks] = bits.bool()
        out[paint] = v
    return out.cpu().numpy(), modes.cpu().numpy()


def main():
    evaldir = OUT / "evaluation"
    meta = load(evaldir / "summary.json")
    ck = OUT / "translator.safetensors"
    require(
        sha(ck) == meta["checkpoint"]
        and sha(evaldir / "writer.npz") == meta["files_sha256"]["writer.npz"],
        "Changed evaluation",
    )
    weights = load_file(str(ck))
    net = CausalGlyphTranslator(weights["landmarks"], weights["exemplars"])
    net.load_state_dict(weights)
    net.cuda().eval()
    bank = tiles()
    rows, _ = select()
    pixels = torch.from_numpy(np.unpackbits(bank, axis=1)).bool().cuda()
    with np.load(evaldir / "writer.npz") as lib:
        packed, offset = lib["packed"], lib["offsets"]
    summary = Counter()
    details = []
    for start in range(0, len(rows), 24):
        block = rows[start : start + 24]

        def pad(field):
            ids = np.zeros((len(block), max(len(r[field]) for r in block)), np.int32)
            for i, r in enumerate(block):
                ids[i, : len(r[field])] = r[field]
            return pixels[torch.from_numpy(ids).long().cuda()]

        src, target = pad("x"), pad("y")
        lengths = torch.tensor([len(r["x"]) for r in block], device="cuda")
        teacher, modes = predict(net, src, lengths, target)
        replacement = target.clone()
        indices = []
        for i, r in enumerate(block):
            j = start + i
            emitted = packed[offset[j] : offset[j + 1]]
            gold = bank[r["y"]]
            upto = min(len(gold), len(emitted))
            diff = np.flatnonzero((emitted[:upto] != gold[:upto]).any(axis=1))
            at = int(diff[0]) if len(diff) else upto
            indices.append(at)
            if at < upto:
                replacement[i, at] = (
                    torch.from_numpy(np.unpackbits(emitted[at]))
                    .to(target.device)
                    .bool()
                )
            # Replaying teacher context must match free generation up to the first error.
            forced = np.packbits(teacher[i, :at], axis=1)
            if not np.array_equal(forced, emitted[:at]) or (modes[i, :at] == END).any():
                summary["prefix_replay_mismatch_cases"] += 1
        altered, alt_modes = predict(net, src, lengths, replacement)
        for i, r in enumerate(block):
            n = len(r["y"])
            gold = bank[r["y"]]
            j = start + i
            free = packed[offset[j] : offset[j + 1]]
            at = indices[i]
            correct = (np.packbits(teacher[i, :n], axis=1) == gold).all(1) & (
                modes[i, :n] != END
            )
            alt = (np.packbits(altered[i, :n], axis=1) == gold).all(1) & (
                alt_modes[i, :n] != END
            )
            summary.update(
                dict(
                    glyph_positions=n,
                    clean_correct=int(correct.sum()),
                    end_positions=1,
                    clean_correct_end=int(modes[i, n] == END),
                    first_correct=int(correct[0]),
                    first_positions=1,
                    free_correct_at_position=int(
                        (gold[: min(n, len(free))] == free[: min(n, len(free))])
                        .all(1)
                        .sum()
                    ),
                    complete_gold_history=int(correct.all() and modes[i, n] == END),
                )
            )
            if at + 1 < n and at < len(free):
                summary.update(
                    dict(
                        perturbations=1,
                        next_correct_clean=int(correct[at + 1]),
                        next_correct_altered=int(alt[at + 1]),
                        next_lost=int(correct[at + 1] and not alt[at + 1]),
                        next_gained=int(alt[at + 1] and not correct[at + 1]),
                    )
                )
            details.append(
                dict(
                    first_free_difference=at,
                    reference_length=n,
                    clean_correct=int(correct.sum()),
                )
            )
    write_new(
        evaldir / "history.json",
        dict(counts=summary, checkpoint=sha(ck), details=details),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
