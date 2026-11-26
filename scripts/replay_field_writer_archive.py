"""Regenerate a published optical-writing experiment on CPU from its source PNGs."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from ilm.visual_lm.field_to_glyph import restore
from scripts.write_neural_glyphs import grid_bits, load_model, save_ink

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "publication/ilm-nature/evidence/field_writer_outputs.zip"
EVIDENCE = ROOT / "references/evidence/field_to_glyph_v1"
MODES = ("parent", "regression", "prediction", "exchange")


@torch.inference_mode()
def replay(archive, chunk):
    with ZipFile(archive) as bundle:
        rows = json.loads(bundle.read("request-outcomes.json"))
        if [row["id"] for row in rows] != list(range(192)):
            raise ValueError("Incomplete or reordered archive")
        grids = [grid_bits(BytesIO(bundle.read(f'{r["id"]:04d}/input.png')), len(r["source"])) for r in rows]
        counts, differences, model_hashes = {}, [], {}
        for mode in MODES:
            arm = "prediction" if mode == "exchange" else mode
            if mode == "parent":
                net = load_model("cpu")
                folder = ROOT / "models/neural_glyph_motor"
                parents = []
            else:
                net, parents = restore(arm, "cpu")
                folder = ROOT / "models/field_to_glyph" / arm
            model_hashes[mode] = [json.loads((folder/"fit.json").read_text())["weights"], *parents]
            matched = 0
            for direction, lang in enumerate(("en", "zh")):
                subset = [r for r in rows if r["target_language"] == lang]
                for j in range(0, len(subset), chunk):
                    batch = subset[j:j+chunk]
                    src = [grids[row["answers"][mode]["source_index"]] for row in batch]
                    lengths = torch.tensor([len(g) for g in src])
                    x = np.zeros((len(batch), int(lengths.max()), 32, 32), dtype=bool)
                    for b, g in enumerate(src):
                        x[b, :len(g)] = g
                    ink, sizes, stops = net.write(torch.from_numpy(x), lengths, direction)
                    for b, row in enumerate(batch):
                        index, expected = row["id"], row["answers"][mode]
                        original = bundle.read(f"{index:04d}/{mode}.png")
                        if sha256(original).hexdigest() != expected["png"]:
                            raise ValueError(f"Corrupt original PNG {index}/{mode}")
                        new = BytesIO()
                        n = int(sizes[b])
                        save_ink(ink[b, :n].numpy(), new, image_format="PNG")
                        ok = new.getvalue() == original and n == expected["cells"] and bool(stops[b]) == expected["stop"]
                        if ok:
                            matched += 1
                        else:
                            differences.append(dict(index=index, mode=mode, cells=n, stopped=bool(stops[b]),
                                                    replay_png=sha256(new.getvalue()).hexdigest()))
            counts[mode] = matched
    return dict(requests=len(rows), cpu_threads=torch.get_num_threads(), batch_limit=chunk,
                model_sha256=model_hashes, matched_png_length_stop=counts, differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    digest = sha256(args.archive.read_bytes()).hexdigest()
    if digest != json.loads((EVIDENCE/"evaluation.json").read_text())["archive"]:
        raise ValueError("Archive fingerprint differs from evaluation")
    result = replay(args.archive, 12)
    result.update(archive_sha256=digest, program_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    if args.receipt:
        args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(result["matched_png_length_stop"], flush=True)
    if result["differences"]:
        raise SystemExit("CPU/GPU output discrepancy")


if __name__ == "__main__":
    main()
