"""Answer an English pictured action question about a two-line Chinese page."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.gloss_focus import cell_features, encoding_windows
from ilm.visual_lm.pictured_action_lookup import PicturedActionLookup
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_visual_gloss import load_focus
from scripts.read_optical_role import digest
from scripts.respond_pointing_glyphs import render_text
from scripts.write_visual_gloss import ROOT, DEFAULT, restore, load_adapter, respond


def restore_lookup(folder, device):
    spec = json.loads((folder / "weights.json").read_text())
    for p, field in [
        (folder / "lookup.safetensors", "sha256"),
        (ROOT / "models/visual_clause_roles/focus.safetensors", "parent_reader"),
        (ROOT / "models/visual_word_senses/writer.safetensors", "parent_writer"),
        (ROOT / "models/pronoun_writer/adapter.safetensors", "adapter"),
    ]:
        if digest(p) != spec[field]:
            raise ValueError(f"Weight checksum mismatch for {p.name}")
    state = load_file(str(folder / "lookup.safetensors"))
    if spec.get("architecture") == "visual_action_meaning_delta_v1":
        base = folder.parent / "aligned" / "lookup.safetensors"
        if digest(base) != spec["base_sha256"]:
            raise ValueError("Incorrect base for the exchanged reader")
        combined = load_file(str(base))
        if set(state) != {
            k for k in combined if k.split(".")[0] not in {"parent", "words"}
        }:
            raise ValueError("Incomplete or unexpected control tensors")
        combined.update(state)
        state = combined
    if spec.get("architecture") in {
        "visual_action_meaning_v1",
        "visual_action_meaning_delta_v1",
    }:
        from ilm.visual_lm.visual_action_meaning import VisualActionMeaning

        net = VisualActionMeaning().to(device)
    else:
        net = PicturedActionLookup().to(device)
    net.load_state_dict(state)
    return net.eval()


def contains(question, ink, stopped):
    if not stopped or not len(ink) or len(ink) > len(question):
        return False
    return any(
        np.array_equal(question[i : i + len(ink)], ink)
        for i in np.flatnonzero((question == ink[0]).all(-1))
    )


def read_lines(page):
    with Image.open(page) as im:
        if im.height != 64 or im.width % 32:
            raise ValueError(
                "Expected two 32-pixel-high rows with whole 32-pixel cells"
            )
    grid = read_image(page, None).reshape(2, -1, 128)
    lines = []
    for row in grid:
        occupied = np.flatnonzero(row.any(axis=1))
        if not len(occupied):
            raise ValueError("Each page row must contain ink")
        lines.append(row[: occupied[-1] + 1])
    return lines


def analyse(writer, reader, pixels, lengths):
    states = encoding_windows(writer, cell_features(pixels))
    score = reader.arc_scores(states, lengths)
    winner = score.flatten(1).argmax(-1)
    width = states.shape[1]
    return states, torch.stack((winner // width, winner % width), dim=-1)


@torch.inference_mode()
def answer(writer, reader, residual, lookup, lines, question):
    if len(lines) != 2 or not all(len(s) for s in lines) or not len(question):
        raise ValueError("Two nonempty lines and a question are required")
    device = next(lookup.parameters()).device
    lengths = torch.tensor([len(s) for s in lines], device=device)
    raw = np.zeros((2, int(lengths.max()), 128), np.uint8)
    for i, s in enumerate(lines):
        raw[i, : len(s)] = s
    states, arcs = analyse(writer, reader, torch.from_numpy(raw).to(device), lengths)
    scores = lookup.scores(
        states[torch.arange(2, device=device), arcs[:, 0]],
        cell_features(torch.tensor(question, device=device)[None]),
        torch.tensor([len(question)], device=device),
    )[:, 0]
    positions = arcs.cpu().tolist()
    hits = []
    for source, (v, _) in zip(lines, positions):
        ink, stopped = respond(writer, source, v, context_adapter=residual)
        hits.append(contains(question, ink, stopped))
    row = hits.index(True) if hits.count(True) == 1 else int(scores.argmax())
    obj = positions[row][1]
    ink, stopped = respond(writer, lines[row], obj, context_adapter=residual)
    return ink, dict(
        row=row + 1,
        source_cell=obj,
        stopped=stopped,
        scores=scores.tolist(),
        unique_ink_match=hits.count(True) == 1,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    pg = ap.add_mutually_exclusive_group(required=True)
    pg.add_argument("--page", type=Path)
    pg.add_argument("--lines", nargs=2, metavar=("FIRST", "SECOND"))
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--question", type=Path)
    q.add_argument("--question-text")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--reader-dir", type=Path, default=ROOT / "models/pictured_action_lookup"
    )
    ap.add_argument("--transcribe", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)

    def draw(s):
        return render_text(s, DEFAULT, 2).numpy()

    lines = read_lines(args.page) if args.page else [draw(s) for s in args.lines]
    question = (
        read_image(args.question, None) if args.question else draw(args.question_text)
    )
    dev = args.device
    ink, report = answer(
        restore(device=dev),
        load_focus(ROOT / "models/visual_clause_roles", dev),
        load_adapter(ROOT / "models/pronoun_writer", dev),
        restore_lookup(args.reader_dir, dev),
        lines,
        question,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_strip(ink, args.output)
    if args.transcribe:
        letters = "abcdefghijklmnopqrstuvwxyz-"
        table = {x.tobytes(): s for x, s in zip(draw(letters), letters)}
        report["text"] = "".join(table.get(x.tobytes(), "�") for x in ink)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
