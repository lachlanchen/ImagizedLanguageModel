"""Measure the contribution of recurrent writing history to visual copying."""
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.idiom_ink import GROUPS, LANDMARKS, STOP, PicturedIdiom, bit_image, visual_input
from scripts.train_visual_writing_control import GlyphControl, OUT as CONTROL
from scripts.measure_physical_writer_states import OUT as STATES, PARENT, ROOT, load, require, sha, write_new

OUT = ROOT / "artifacts/glyph_boundary_reset_v1"


@torch.inference_mode()
def copy_images(writer, states, reset, beam=4):
    batch, glyphs, width = states.shape
    rows = torch.arange(batch, device=states.device)[:, None]
    copied = []
    hidden = states[:, 0]
    previous = torch.full((batch,), 17, device=states.device)
    feedback = states.new_zeros(batch, LANDMARKS)
    for slot in range(glyphs):
        context = states[:, slot]
        if reset:
            hidden = context
            previous = torch.full_like(previous, 17)
            feedback = torch.zeros_like(feedback)
        conditioning = context[:, None].expand(-1, beam, -1)
        hidden = hidden[:, None].expand(-1, beam, -1).contiguous()
        previous = previous[:, None].expand(-1, beam)
        feedback = feedback[:, None].expand(-1, beam, -1)
        score = states.new_full((batch, beam), -torch.inf)
        score[:, 0] = 0
        patterns = previous.new_zeros(batch, beam, GROUPS)
        for offset in range(GROUPS):
            inputs = writer.inputs(previous, torch.full_like(previous, offset), feedback, conditioning)
            nxt, _ = writer.causal(inputs.reshape(batch * beam, 1, -1),
                                  hidden.reshape(1, batch * beam, width).contiguous())
            nxt = nxt.reshape(batch, beam, width)
            logits = writer.next_pattern(nxt)
            logits[:, :, STOP] = -torch.inf
            score, choices = (score[:, :, None] + logits.log_softmax(-1)).flatten(1).topk(beam)
            parent = choices // (STOP + 1)
            previous = choices % (STOP + 1)
            patterns = patterns[rows, parent]
            patterns[:, :, offset] = previous
            hidden = nxt[rows, parent]
        pixels = writer.motor(bit_image(patterns[:, 0])) > 0
        copied.append(pixels)
        hidden, previous = hidden[:, 0], previous[:, 0]
        feedback = pixels[:, writer.landmarks].float()
    return np.packbits(torch.stack(copied, dim=1).cpu().numpy(), axis=-1)


def run():
    start = time.monotonic()
    torch.set_num_threads(2)
    require("4090" in torch.cuda.get_device_name(), "Expected registered GPU")
    receipt = load(CONTROL / "result.json")
    require(sha(CONTROL / "sources.npz") == receipt["source_sha256"], "Changed target inventory")
    weights_path = PARENT / "diverse/weights.pt"
    require(sha(weights_path) == load(STATES / "result.json")["parent_sha256"], "Changed frozen writer")
    controller_path = CONTROL / "aligned/controller.pt"
    require(sha(controller_path) == receipt["arms"]["aligned"]["weights_sha256"], "Changed controller")
    weights = torch.load(weights_path, weights_only=True, map_location="cpu")
    writer = PicturedIdiom(weights["landmarks"]).cuda().eval()
    writer.load_state_dict(weights)
    controller = GlyphControl().cuda().eval()
    controller.load_state_dict(torch.load(controller_path, weights_only=True, map_location="cpu"))
    bank = np.load(PARENT / "pool/pixels.npy")
    with torch.inference_mode():
        pixels = torch.from_numpy(np.unpackbits(bank, axis=1)).cuda().bool()
        states = controller(visual_input(pixels, writer.landmarks))
    OUT.mkdir()
    report = {}
    with np.load(CONTROL / "sources.npz") as source:
        for split in ("observed", "mixed"):
            target = source[split]
            arms = {}
            first = None
            for arm in ("carry", "reset"):
                parts = []
                for rows in np.array_split(target, (len(target) + 127) // 128):
                    controls = states[torch.from_numpy(rows.astype(np.int64)).cuda()]
                    parts.append(copy_images(writer, controls, reset=arm == "reset"))
                    require(time.monotonic() - start < 120 and torch.cuda.max_memory_allocated() < 2**30,
                            "Registered resource ceiling")
                ink = np.concatenate(parts)
                if first is None:
                    first = ink[:, 0].copy()
                else:
                    require(np.array_equal(first, ink[:, 0]), "First glyph disagrees between matched arms")
                hits = (ink == bank[target]).all(axis=2)
                prefix = np.logical_and.accumulate(hits, axis=1)
                path = OUT / f"{split}_{arm}.npz"
                np.savez_compressed(path, packed=ink, exact=prefix[:, -1])
                result = dict(forms=len(target), prefix_exact=prefix.sum(axis=0).tolist(),
                              position_exact=hits.sum(axis=0).tolist(), pixels_sha256=sha(path))
                if split == "observed":
                    rare = source["counts"] <= 5
                    result.update(rare_exact=int(prefix[rare, -1].sum()), rare_forms=int(rare.sum()))
                arms[arm] = result
            report[split] = arms
    seen = report["observed"]["reset"]
    novel = report["mixed"]["reset"]
    passed = (seen["prefix_exact"][-1] >= .9 * seen["forms"] and seen["rare_exact"] >= .9 * seen["rare_forms"]
              and novel["prefix_exact"][-1] >= .9 * novel["forms"]
              and novel["prefix_exact"][-1] > report["mixed"]["carry"]["prefix_exact"][-1])
    write_new(OUT / "result.json", dict(splits=report, passed=bool(passed), source_sha256=receipt["source_sha256"],
              program_sha256=sha(Path(__file__)), seconds=time.monotonic()-start,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), device=torch.cuda.get_device_name()))
    print(report, flush=True)


if __name__ == "__main__":
    with (OUT.parent / "ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
