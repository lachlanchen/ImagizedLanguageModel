"""Answer a pictured Chinese question using one or two facts from the supplied venue."""

import argparse
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.semantic_listener import beam_write
from ilm.visual_lm.ordered_copy import realize
from scripts.prepare_chinese_two_fact import MODEL, PARENT, WIDTH, FIELDS, MARKS, read
from scripts.train_chinese_two_fact import load, output_bank
from scripts.write_chinese_card_reply import (
    field_prefixes,
    load_page,
    make_page,
    read_fields,
    card_image,
    display,
)


def write_plans(
    net,
    question,
    labels,
    domains,
    available,
    palette,
    second_id,
    teacher=None,
    cover=False,
):
    _, _, plans, memory, visible = net.plan(
        question, labels, domains, available, teacher
    )
    allowed = torch.ones(
        len(question), len(palette), device=palette.device, dtype=torch.bool
    )
    allowed[:, second_id] = visible[:, 3]
    ids, lengths = beam_write(
        net, memory, visible, palette, None, limit=120, allowed=allowed
    )
    actions = [row[:n] for row, n in zip(ids.tolist(), lengths.tolist())]
    if cover:
        actions = realize(
            net, memory, visible, palette, actions, second_id - 1, second_id
        )
    return plans.tolist(), actions


def keys_for(plan):
    return [FIELDS[i] for i in plan if i < len(FIELDS)]


def expand_plan(actions, alphabet, palette, keys, values):
    bindings = dict(zip(MARKS, ["名称", *keys]))
    chunks = []
    for idx in actions:
        c = alphabet[idx]
        if not c:
            raise ValueError("Unexpected stop inside answer")
        if c in MARKS:
            if c not in bindings:
                raise ValueError("Unbound source role")
            chunks.append(values[bindings[c]])
        else:
            chunks.append(palette[idx : idx + 1])
    return np.concatenate(chunks) if chunks else np.empty((0, 128), np.uint8)


class TwoFactReply:
    def __init__(self, device="cpu", arm="cursor", *, aligned=True):
        if arm != "cursor" and not arm.startswith("ink-"):
            raise ValueError("An image page requires an optical arm")
        from scripts.train_field_alignment import load_alignment

        self.aligner = load_alignment("ink", device) if aligned else None
        if arm == "cursor" and not aligned:
            raise ValueError("The cursor writer requires a field aligner")
        self.device = device
        self.net = None if arm == "cursor" else load(arm, device)
        self.alphabet = read(MODEL / "alphabet.json")
        self.images = np.load(MODEL / "palette.npy")
        self.palette = output_bank(device)
        self.prefixes = field_prefixes()
        labels = np.load(PARENT / "label-pixels.npy")
        self.domains = {row.tobytes(): i for i, row in enumerate(labels[len(FIELDS) :])}
        features = observations({"tiles": labels.reshape(-1, 128)}, "ink")
        with torch.inference_mode():
            label_ink = torch.as_tensor(features, device=device).reshape(
                len(labels), WIDTH, 256
            )
            self.cursor = None
            if arm == "cursor":
                from scripts.train_field_cursor import load_cursor, palette

                self.cursor = load_cursor(device)
                self.labels = self.cursor.encode_labels(label_ink)
                self.embedding = self.cursor.glyph(palette(device))
            else:
                self.labels = self.net.encode_rows(label_ink)
        with np.load(PARENT / "transcription.npz") as bank:
            self.characters = dict(zip(map(bytes, bank["pixels"]), bank["characters"]))
        for c, tile in zip(self.alphabet, self.images):
            if c and c not in MARKS:
                self.characters.setdefault(bytes(tile), c)

    @torch.inference_mode()
    def reply(self, page, *, cover=True):
        domain, values = read_fields(page, self.domains, self.prefixes)
        question = torch.as_tensor(
            observations({"tiles": page[0]}, "ink"), device=self.device
        )[None]
        available = torch.tensor([[k in values for k in FIELDS]], device=self.device)
        domains = torch.tensor([domain], device=self.device)
        override = (
            self.aligner.choose(question, domains, available)
            if self.aligner is not None
            else None
        )
        if self.cursor is not None:
            from scripts.train_field_cursor import convert_plans, cursor_actions

            if not cover:
                raise ValueError("Cursor writing enforces the source plan")
            plans = override.tolist()
            seq = self.cursor.write(
                self.labels,
                domains + len(FIELDS),
                convert_plans(override, len(self.labels)),
                self.embedding,
            )
            actions = cursor_actions(seq, self.alphabet)
        else:
            plans, actions = write_plans(
                self.net,
                question,
                self.labels,
                domains,
                available,
                self.palette,
                self.alphabet.index(MARKS[-1]),
                teacher=override,
                cover=cover,
            )
        keys = keys_for(plans[0])
        pixels = expand_plan(actions[0], self.alphabet, self.images, keys, values)
        return keys, pixels

    def transcript(self, pixels):
        return "".join(self.characters.get(bytes(tile), "\ufffc") for tile in pixels)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path)
    g.add_argument("--record", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument(
        "--arm", choices=("cursor", "ink-natural", "ink-joined"), default="cursor"
    )
    ap.add_argument(
        "--pooled", action="store_true", help="Use the original whole-question planner"
    )
    ap.add_argument(
        "--unconstrained",
        action="store_true",
        help="Return the original freely stopped writer",
    )
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.device == "cuda":
        from scripts.verse_gpu_runtime import configure_cuda_cli

        configure_cuda_cli()
    torch.set_num_threads(2)
    page = load_page(args.input) if args.input else make_page(read(args.record))
    writer = TwoFactReply(args.device, args.arm, aligned=not args.pooled)
    keys, pixels = writer.reply(page, cover=not args.unconstrained)
    args.output.mkdir(parents=True)
    card_image(page).save(args.output / "input.png")
    display(pixels).save(args.output / "answer.png")
    (args.output / "answer.bin").write_bytes(pixels.tobytes())
    line = writer.transcript(pixels)
    (args.output / "answer.txt").write_text(line + "\n")
    print("、".join(keys), line)


if __name__ == "__main__":
    main()
