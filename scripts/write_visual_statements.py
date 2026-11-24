"""Write a full English answer from a pictured question and evidence page."""

import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from ilm.visual_lm.recurrent_evidence import ComposingWriter
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import read_image, read_text, DEFAULT_FONT
from scripts.prepare_visual_statements import MODEL, PUB, read, sha, dump
from scripts.train_visual_statements import fact_page
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_paired_evidence import reader, generate as read_fact
from scripts.write_recurrent_evidence import generate
from scripts.evaluate_visual_discourse import draw


def editor(device, run_dir=None):
    p = (run_dir or MODEL) / "statement.safetensors"
    if sha(p) != read((run_dir or PUB) / "training.json")["weights"]:
        raise ValueError("Different visual realizer")
    net = ComposingWriter().to(device)
    net.load_state_dict(load_file(str(p), device=device))
    return net.eval()


def realize(base, net, x, n, qs, facts, space, cap=160):
    src = [fact_page(q, f, space) for q, f in zip(qs, facts, strict=True)]
    return generate(base, net, x, n, qs, src, src, cap)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("question", "passage"):
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--" + k)
        g.add_argument("--" + k + "-image", type=Path)
        p.add_argument("--" + k + "-cells", type=int)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--run-dir",
        type=Path,
        help="Load a separately fitted statement head and its training receipt.",
    )
    a = p.parse_args()
    if a.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(3)
    w = Writer(device=a.device)
    r = reader("evidence", a.device)
    e = editor(a.device, a.run_dir)

    def get(k):
        img = getattr(a, k + "_image")
        return (
            read_image(img, getattr(a, k + "_cells"))
            if img
            else read_text(getattr(a, k), DEFAULT_FONT)
        )

    shapes, (q, s) = encode_pages(w.palette, [get("question"), get("passage")])
    x = torch.from_numpy(observations({"tiles": shapes}, "ink")).to(a.device)
    ((fact, fstop),) = read_fact(w.model, r, x, len(w.palette), [q], [s], [s + q], 96)
    space = next(i for i, v in enumerate(w.palette) if i >= 2 and not v.any())
    ((out, end),) = realize(w.model, e, x, len(w.palette), [q], [fact], space)
    a.output.mkdir(exist_ok=False, parents=True)

    def save(k, seq):
        ink = shapes[np.asarray(seq, dtype=int)]
        draw(a.output, k, ink)
        (a.output / (k + ".bin")).write_bytes(ink.tobytes())
        return "".join(w.transcript[i] if i < len(w.transcript) else "�" for i in seq)

    txt = save("answer", out)
    evidence = save("read_fact", fact)
    dump(
        a.output / "result.json",
        dict(
            transcript=txt,
            fact=evidence,
            fact_stopped=fstop,
            stopped=end,
            cells=len(out),
        ),
    )
    print(txt)


if __name__ == "__main__":
    main()
