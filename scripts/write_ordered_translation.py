"""Write a translation image from a pictured sentence or rasterized typed input."""

import argparse
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.ordered_translation import OrderedTranslator
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import save_grid, read_image
from scripts.prepare_ordered_translation import Ink, MODEL, PUB, read, dump, sha
from scripts.verse_gpu_runtime import configure_cuda_cli


def restore(stride, device):
    if stride in (
        "proposition-likelihood",
        "proposition-contrast",
        "bilingual-likelihood",
        "bilingual-aligned",
        "paired-likelihood",
        "paired-assignment",
        "paraphrase-canonical",
        "paraphrase-varied",
    ):
        family, objective = stride.split("-")
        base = (
            MODEL.parent
            / {
                "proposition": "proposition_translation",
                "bilingual": "bilingual_ink",
                "paraphrase": "translation_paraphrases",
                "paired": "paired_ink",
            }[family]
        )
        path = base / objective
        stride = 1
    else:
        path = MODEL / str(stride)
    receipt = read(path / "fit.json")
    if sha(path / "weights.safetensors") != receipt["weights"]:
        raise ValueError("Weights differ")
    for name in ("inventories.json", "palette.npy"):
        if sha(MODEL / name) != read(PUB / "corpus.json")["files"][name]:
            raise ValueError(name)
    if stride == "specimen":
        from ilm.visual_lm.specimen_translation import SpecimenTranslator

        m = SpecimenTranslator()
    else:
        m = OrderedTranslator(1 if stride in ("replay", "news") else stride)
    m.load_state_dict(load_file(str(path / "weights.safetensors")))
    return m.to(device).eval()


def stopped(sequence, limit=128):
    values = sequence.tolist()
    n = values.index(1) if 1 in values else len(values)
    return values[: min(n, limit)], n <= limit and 1 in values


def infer(model, raw, direction, device, decode="greedy"):
    palette = np.load(MODEL / "palette.npy")
    # Every handle below names a physical bitmap local to this request.
    tiles = list(map(bytes, palette))
    index = {b: i for i, b in enumerate(tiles) if i >= 2}
    ids = []
    for cell in raw:
        b = bytes(cell)
        if b not in index:
            index[b] = len(tiles)
            tiles.append(b)
        ids.append(index[b])
    if not 0 < len(ids) <= 96:
        raise ValueError("Supply between 1 and 96 occupied glyph cells")
    features = observations(
        {"tiles": np.frombuffer(b"".join(tiles), np.uint8).reshape(-1, 128)}, "ink"
    )
    inventory = torch.tensor(read(MODEL / "inventories.json")[direction], device=device)
    feature_bank = torch.tensor(features, device=device)
    source = torch.tensor([ids], device=device)
    if decode == "order":
        from ilm.visual_lm.order_decoding import bitmap_ranks, write_with_order

        sequence = write_with_order(
            model,
            feature_bank,
            source,
            inventory,
            direction,
            bitmap_ranks(tiles, device),
        )[0]
    else:
        sequence = model.generate(feature_bank, source, inventory, direction)[0]
    out, stop = stopped(sequence)
    if decode in ("forward", "roundtrip"):
        from ilm.visual_lm.reverse_translation import rerank

        inventories = [
            torch.tensor(v, device=device) for v in read(MODEL / "inventories.json")
        ]
        alternatives = rerank(
            model,
            torch.tensor(features, device=device),
            [ids],
            inventories,
            direction,
            [(out, stop)],
        )
        out, stop = alternatives[0][decode], True
    elif decode not in ("greedy", "order"):
        raise ValueError("Unknown decoder")
    return b"".join(tiles[i] for i in out), stop


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--text")
    ap.add_argument(
        "--cells", type=int, help="Occupied cells in the input PNG, including spaces"
    )
    ap.add_argument("--to", choices=("en", "zh"), required=True)
    ap.add_argument(
        "--stride",
        "--model",
        choices=(
            "1",
            "4",
            "specimen",
            "replay",
            "news",
            "proposition-likelihood",
            "proposition-contrast",
            "bilingual-likelihood",
            "bilingual-aligned",
            "paired-likelihood",
            "paired-assignment",
            "paraphrase-canonical",
            "paraphrase-varied",
        ),
        default="1",
    )
    ap.add_argument(
        "--preserve-specimens",
        action="store_true",
        help="Carry unfamiliar image cells as opaque named specimens, using the pointer writer",
    )
    ap.add_argument(
        "--decode",
        choices=("greedy", "forward", "roundtrip", "order"),
        default="greedy",
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = ap.parse_args()
    if a.decode != "greedy" and (a.preserve_specimens or a.stride == "specimen"):
        ap.error("The selected decoder requires an ordinary translation writer")
    if a.device == "cuda":
        configure_cuda_cli()
    torch.set_num_threads(2)
    if a.image:
        if not a.cells:
            ap.error("--image needs its occupied --cells")
        raw = read_image(a.image, a.cells)
    else:
        ink = Ink()
        ids = ink.encode(a.text)
        if ink.bad:
            raise ValueError("Font lacks a supplied glyph")
        raw = ink.array()[ids]
    direction = int(a.to == "zh")
    if a.preserve_specimens:
        from ilm.visual_lm.specimen_plan import SpecimenPlanWriter

        path = MODEL / "specimen/role.npy"
        if sha(path) != read(PUB / "substitution.json")["descriptor"]:
            raise ValueError("Specimen descriptor differs")
        writer = SpecimenPlanWriter(
            restore("specimen", a.device),
            np.load(MODEL / "palette.npy"),
            read(MODEL / "inventories.json"),
            np.load(path),
            a.device,
        )
        cells, done = writer.translate(raw, direction)
        out = b"".join(cells)
    else:
        net = restore(int(a.stride) if a.stride.isdigit() else a.stride, a.device)
        out, done = infer(net, raw, direction, a.device, a.decode)
    a.output.mkdir(parents=True, exist_ok=False)
    save_grid(out, a.output / "answer.png")
    dump(
        a.output / "receipt.json",
        dict(cells=len(out) // 128, stopped=done, sha256=sha256(out).hexdigest()),
    )
