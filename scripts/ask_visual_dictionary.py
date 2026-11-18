"""Ask the image-stream writer about a locally archived Hanziyuan entry."""
import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.infer_pictured_entry import bitmap, page
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json
from scripts.prepare_source_plate_questions import CACHE, SPLITS, STAGES
from scripts.read_grounded_glyph_answer import respond
from scripts.train_general_entry_writer import PARENT, WORK, load_writer


def source_record(head, entry_id):
    info = load(SPLITS / "report.json")
    records = []
    for split in ("train", "development"):
        path = SPLITS / (split + ".json")
        if sha(path) != info["partitions"][split]["file_sha256"]:
            raise ValueError("Historical index digest changed")
        records.extend((r, split) for r in load(path) if r["char"] == head
                       and (entry_id is None or r["entry_id"] == entry_id))
    if len(records) != 1:
        choices = [r["entry_id"] for r, _ in records]
        raise ValueError(f"Expected one accessible entry for {head!r}. Available matches {choices}")
    if sha(CACHE) != info["raster_cache"]["cache_sha256"]:
        raise ValueError("Historical raster cache digest changed")
    return records[0]


def picture(record, stages, raster, names):
    def render(text):
        for char in text:
            tile = raster.tile(char)
            names[tile].add(char)
            yield tile

    fields = record["fields"]
    gloss = fields.get("original_meaning", "").strip().removeprefix("Meaning ").removesuffix(".")
    sound = fields.get("main_pronunciation", "").strip()
    cells = list(render("【" + record["char"] + "】"))
    for label, value in (("本义", gloss), ("读音", sound)):
        if value:
            cells.extend(render(label + "：" + value + "；"))
    cache = torch.load(CACHE, weights_only=True, map_location="cpu")["pixels"]
    sources = []
    for stage in stages:
        rows = [g for g in record["glyphs"] if g["stage"] == stage]
        if not rows:
            continue
        g = min(rows, key=lambda item: item["label"])
        raw = np.packbits(cache[g["raster_index"]].numpy().reshape(1024))
        if sha256(raw).hexdigest() != g["raster_sha256"]:
            raise ValueError(f"Source pixels differ for {g['label']}")
        ink = np.bitwise_not(raw).tobytes()
        names[ink].add("[" + g["label"] + "]")
        cells += [*render(STAGES[stage][0] + "："), ink, *render("；")]
        sources.append(dict(stage=stage, label=g["label"], raster_sha256=g["raster_sha256"],
                            original_sha256=g["sha256"], ink_sha256=sha256(ink).hexdigest()))
    return b"".join(cells), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char", required=True)
    parser.add_argument("--entry", help="Select the dictionary entry if the headword is ambiguous")
    parser.add_argument("--stages", choices=list(STAGES), nargs="+", default=list(STAGES))
    q = parser.add_mutually_exclusive_group(required=True)
    q.add_argument("--question")
    q.add_argument("--question-image", type=Path, help="Image of aligned 32 by 32 cells in reading order")
    parser.add_argument("--question-cells", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, default=WORK / "paired")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.stages)) != len(args.stages):
        parser.error("Choose a fresh output directory and distinct stages.")
    if args.question_cells is not None and not args.question_image:
        parser.error("A question cell count requires an image.")
    torch.set_num_threads(2)
    record, split = source_record(args.char, args.entry)
    raster, names = Rasterizer(), defaultdict(set)
    card, sources = picture(record, args.stages, raster, names)
    if args.question_image:
        question = bitmap(args.question_image, args.question_cells)
    else:
        question = b"".join(raster.tile(c) for c in args.question)
    memory = PARENT / "memory.npy"
    net = load_writer(args.checkpoint_dir, "cpu")
    p, q = (np.frombuffer(s, np.uint8).reshape(-1, 128) for s in (card, question))
    answer, stopped = respond(net, np.load(memory), p, q, "cpu")
    # Decode only after the complete output was frozen, preserving unmatched pixels.
    for c in set((args.question or "") + "的本义读音是，金文甲骨文篆書六書通作。"):
        names[raster.tile(c)].add(c)
    labels = []
    unknown = 0
    for j in range(0, len(answer), 128):
        cell = answer[j:j + 128]
        candidates = names.get(cell, set())
        if len(candidates) == 1:
            labels.append(next(iter(candidates)))
        else:
            labels.append("[bitmap-" + sha256(cell).hexdigest()[:12] + "]")
            unknown += 1
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        dest = Path(tmp) / "response"
        dest.mkdir()
        for name, stream in (("source", card), ("question", question), ("answer", answer)):
            page(stream).save(dest / (name + ".png"))
        (dest / "answer.bin").write_bytes(answer)
        (dest / "answer.txt").write_text("".join(labels) + "\n", encoding="utf-8")
        write_json(dest / "receipt.json", dict(entry=record["entry_id"], split=split, sources=sources,
                   cell_counts=[len(s)//128 for s in (card, question, answer)], stopped=stopped,
                   unidentified_or_ambiguous=unknown, source_sha256=sha256(card).hexdigest(),
                   answer_sha256=sha256(answer).hexdigest(), question_sha256=sha256(question).hexdigest(),
                   weights_sha256=sha(args.checkpoint_dir / "final.safetensors"), memory_sha256=sha(memory)))
        dest.rename(args.output)
    print("".join(labels), args.output / "answer.png")


if __name__ == "__main__":
    main()
