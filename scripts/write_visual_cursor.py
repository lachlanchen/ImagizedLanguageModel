"""Produce an answer image from a pictured question or a rendered text prompt."""

import argparse
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import torch
from ilm.visual_lm.source_cursor import SourceCursor
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_text, read_image, DEFAULT_FONT
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.prepare_visual_cursor import MODEL, PUBLIC, read, dump, sha
from scripts.evaluate_visual_discourse import draw
from scripts.train_visual_cursor import pad, TAIL


def load_cursor(arm, device):
    fp = MODEL / (arm + ".safetensors")
    if sha(fp) != read(PUBLIC / (arm + ".json"))["weights"]:
        raise ValueError("Changed visual cursor")
    model = SourceCursor().to(device)
    model.load_state_dict(load_file(str(fp), device=device))
    return model.eval()


@torch.inference_mode()
def generate(base, model, features, seq, n, limit, persistent):
    if not seq or any(not len(s) or len(s) > 768 for s in seq):
        raise ValueError("Use 1 to 768 occupied cells per prompt")
    if limit < 1:
        raise ValueError("Expected positive generation limit")
    device = features.device
    ids = torch.as_tensor(pad(seq), device=device)
    bank = model.shape(features)
    optical = base.encode(features)
    keys, h = model.page(bank, ids)
    previous = torch.zeros_like(ids, dtype=torch.float32)
    current = model.begin.expand(len(seq), 1, -1)
    recent = [list(s[-TAIL:]) for s in seq]
    answer = [[] for _ in seq]
    ended = [False] * len(seq)
    for _ in range(limit):
        states, h = model.recurrent(current, h)
        starts, advance, beta, stop, custom = model.emissions(states, keys, ids >= 2)
        w, s = model.paths(
            previous, advance[:, 0], starts[:, 0], beta[:, 0], ids >= 2, persistent
        )
        hist = [line[-255:] for line in recent]
        stored = torch.as_tensor(pad([line + [0] for line in hist]), device=device)
        pos = torch.tensor([len(line) for line in hist], device=device)[:, None]
        logits, ptr, gate = base.distributions(optical, stored, n, pos)
        free = logits.new_zeros((len(ids), len(features)))
        free[:, 1:n] = logits[:, 0].softmax(-1) * (1 - gate)
        free.scatter_add_(1, stored.long(), ptr[:, 0] * gate)
        mix = custom[:, 0, None]
        free = free * (1 - mix)
        free[:, 2:n] += mix * (
            model.free(states[:, 0]) @ bank[2:n].T * keys.size(-1) ** -0.5
        ).softmax(-1)
        stop = stop[:, 0]
        result = free * (w * (1 - stop))[:, None]
        source = s * (1 - stop[:, None])
        result.scatter_add_(1, ids.long(), source)
        result[:, 1] += stop
        choice = result.argmax(-1)
        previous = (
            source
            * (ids == choice[:, None])
            / result.gather(1, choice[:, None]).clamp_min(1e-30)
        )
        current = bank[choice][:, None, :]
        for i, v in enumerate(choice.tolist()):
            if not ended[i]:
                if v == 1:
                    ended[i] = True
                else:
                    answer[i].append(v)
                    recent[i].append(v)
        if all(ended):
            break
    return answer, ended


def main():
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--text")
    p.add_argument("--cells", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=("independent", "persistent"), default="persistent")
    p.add_argument("--max-glyphs", type=int, default=96)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(3)
    writer = Writer(device=a.device)
    model = load_cursor(a.arm, a.device)
    source = (
        read_image(a.image, a.cells) if a.image else read_text(a.text, DEFAULT_FONT)
    )
    shapes, seq = encode_pages(writer.palette, [source])
    features = torch.from_numpy(observations({"tiles": shapes}, "ink")).to(a.device)
    out, stop = generate(
        writer.model,
        model,
        features,
        seq,
        len(writer.palette),
        a.max_glyphs,
        a.arm == "persistent",
    )
    physical = shapes[np.asarray(out[0], dtype=int)]
    a.output.mkdir(exist_ok=False, parents=True)
    draw(a.output, "input", source)
    draw(a.output, "answer", physical)
    (a.output / "answer.bin").write_bytes(physical.tobytes())
    dump(
        a.output / "result.json",
        dict(
            cells=len(source),
            emitted=len(physical),
            stopped=stop[0],
            transcript="".join(
                writer.transcript[i] if i < len(writer.transcript) else "�"
                for i in out[0]
            ),
        ),
    )


if __name__ == "__main__":
    main()
