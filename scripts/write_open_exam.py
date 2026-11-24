"""Write an answer image from a Chinese pictured passage and pictured question."""

import argparse
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.recurrent_evidence import ComposingWriter
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.prepare_open_exam import MODEL, read, dump, sha
from scripts.write_recurrent_evidence import generate, load as initial
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.evaluate_visual_discourse import draw


def load(arm, device, model_dir=MODEL):
    if arm == "parent":
        return initial("dynamic", device)
    if arm in ("matched", "exchanged"):
        net = ComposingWriter(dynamic=True).to(device)
        dest = model_dir / arm
        w = dest / "writer.safetensors"
        if sha(w) != read(dest / "fit.json")["weights"]:
            raise ValueError("Changed exam weights")
        net.load_state_dict(load_file(str(w), device=device))
        return net.eval()
    raise ValueError(f"Unknown exam writer {arm!r}")


@torch.inference_mode()
def infer(writer, net, q, p, cap=64):
    tiles, (qi, pi) = encode_pages(writer.palette, [q, p])
    x = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(writer.device)
    [(ids, stop)] = generate(
        writer.model, net, x, len(writer.palette), [qi], [pi], [pi + qi], cap
    )
    return tiles[np.asarray(ids, int)], stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("question", "passage"):
        g = parser.add_mutually_exclusive_group(required=True)
        g.add_argument("--" + name)
        g.add_argument("--" + name + "-image", type=Path)
        parser.add_argument("--" + name + "-cells", type=int)
    parser.add_argument(
        "--arm", choices=("parent", "matched", "exchanged"), default="matched"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL,
        help="Root containing the fitted arm directories",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()

    def ink(k):
        f = getattr(a, k + "_image")
        return (
            read_image(f, getattr(a, k + "_cells"))
            if f
            else read_text(getattr(a, k), DEFAULT_FONT)
        )

    torch.set_num_threads(3)
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli
        from scripts.train_paired_evidence import gpu_owner

        configure_cuda_cli()
        owner = gpu_owner()
    else:
        owner = nullcontext()
    with owner:
        pixels, stop = infer(
            Writer(device=a.device),
            load(a.arm, a.device, a.model_dir),
            ink("question"),
            ink("passage"),
        )
    a.output.mkdir(parents=True, exist_ok=False)
    draw(a.output, "answer", pixels)
    (a.output / "answer.bin").write_bytes(pixels.tobytes())
    dump(a.output / "receipt.json", dict(cells=len(pixels), stopped=stop))
    print(a.output / "answer.png")


if __name__ == "__main__":
    main()
