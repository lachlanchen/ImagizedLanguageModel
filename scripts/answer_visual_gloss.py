"""Write an English gloss from a pictured classical passage and a pictured question."""

import argparse
import json
import os

import numpy as np
from pathlib import Path

import torch
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.ordered_gloss_question import OrderedGlossFocus
from ilm.visual_lm.portable_writer import read_image
from scripts.read_optical_role import digest
from scripts.respond_pointing_glyphs import render_text
from scripts.write_visual_gloss import DEFAULT, ROOT, restore, respond

RELEASE = Path(os.environ.get("ILM_ASKED_GLOSS_RELEASE", ROOT / "models/asked_gloss"))


def load_focus(folder, device):
    file = folder / "focus.safetensors"
    if digest(file) != json.loads((folder / "weights.json").read_text())["sha256"]:
        raise ValueError("Changed pictured-question weights")
    net = OrderedGlossFocus().to(device)
    net.load_state_dict(load_file(str(file)))
    return net.eval()


@torch.inference_mode()
def answer(writer, reader, page, question, lexicon=None):
    if not len(page) or not len(question):
        raise ValueError("Both the passage and the question need at least one cell")
    device = next(reader.parameters()).device

    def features(raw):
        return cell_features(torch.as_tensor(raw, device=device)[None])

    page_length = torch.tensor([len(page)], device=device)
    q = reader.read_questions(
        features(question), torch.tensor([len(question)], device=device)
    )
    ctx = encoding_windows(writer, features(page))
    pos = reader(ctx, page_length, q[:, None])[0, 0].argmax().item()
    ink, ended = respond(writer, page, pos)
    if lexicon is not None:
        landmarks = writer.landmarks.cpu().numpy()
        bits = np.unpackbits(ink, axis=-1)[:, landmarks]
        physical = tuple((bits * (2 ** np.arange(len(landmarks)))).sum(-1).tolist())
        if not ended or physical not in lexicon.members:
            choice = lexicon.best(writer, ctx[0, pos])
            ink = np.packbits(writer.draw(choice).cpu().numpy(), axis=-1)
            ended = True
    return ink, ended, pos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("page", "question"):
        group = p.add_mutually_exclusive_group(required=True)
        group.add_argument("--" + item, type=Path)
        group.add_argument("--" + item + "-text")
        p.add_argument(
            "--" + item + "-cells",
            type=int,
            help="Occupied cells in a padded image grid",
        )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--font", type=Path, default=DEFAULT)
    p.add_argument("--font-index", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--transcribe", action="store_true")
    p.add_argument(
        "--known-glosses",
        action="store_true",
        help="If necessary, draw the most probable complete training gloss",
    )
    args = p.parse_args()

    def load(item):
        path = getattr(args, item)
        count = getattr(args, item + "_cells")
        if path:
            return read_image(path, count)
        if count is not None:
            p.error("--" + item + "-cells requires an image")
        return render_text(
            getattr(args, item + "_text"), args.font, args.font_index
        ).numpy()

    torch.set_num_threads(2)
    writer = restore(device=args.device)
    lexicon = None
    if args.known_glosses:
        from ilm.visual_lm.attested_gloss import AttestedGlosses

        bank = RELEASE / "attested.npz"
        if (
            digest(bank)
            != json.loads((RELEASE / "attested.json").read_text())["sha256"]
        ):
            raise ValueError("Changed attested gloss bank")
        lexicon = AttestedGlosses(bank, writer.stop, args.device)
    ink, stop, pos = answer(
        writer,
        load_focus(RELEASE, args.device),
        load("page"),
        load("question"),
        lexicon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(ink, args.output)
    report = dict(source_cell=pos, stopped=stop, cells=len(ink))
    if args.transcribe:
        letters = "abcdefghijklmnopqrstuvwxyz-"
        table = {
            p.numpy().tobytes(): c
            for p, c in zip(render_text(letters, DEFAULT, 2), letters)
        }
        report["text"] = "".join(table.get(p.tobytes(), "�") for p in ink)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
