"""Measure optical transfer of the frozen writer on the same complete requests."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ilm.visual_lm.ink_transport import InkTransport
from ilm.visual_lm.portable_writer import read_text, DEFAULT_FONT, save_grid
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_book_rehearsal import BASE, WORK, PUB, read, dump, sha
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore, stopped

SERIF = Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc")


@torch.inference_mode()
def main(learned=False):
    torch.set_num_threads(2)
    record = read(WORK / "generated.json")
    rows, answers = record["rows"], record["answers"]
    palette = np.load(BASE / "palette.npy")
    guard = InkTransport(palette[2:])
    symbols = sorted({c for r in rows for c in r["source"]})
    normal = dict(
        zip(symbols, map(bytes, read_text("".join(symbols), DEFAULT_FONT)), strict=True)
    )
    serif = dict(
        zip(symbols, map(bytes, read_text("".join(symbols), SERIF)), strict=True)
    )
    rng = np.random.default_rng(71921)
    speckle = {}
    for c in symbols:
        bits = np.unpackbits(np.frombuffer(normal[c], np.uint8))
        bits[rng.choice(1024, 4, replace=False)] ^= 1
        speckle[c] = np.packbits(bits).tobytes()
    model = restore("book-rehearsal", "cpu")
    inventories = [torch.tensor(ids) for ids in read(BASE / "inventories.json")]
    variants = {"original": normal, "serif": serif, "four_flips": speckle}
    variants.update(
        {
            tag + "_guarded": {c: guard.resolve(b) for c, b in variants[tag].items()}
            for tag in ("serif", "four_flips")
        }
    )
    assert all(
        [normal[c] for c in r["source"]] == list(map(bytes, palette[r["input"]]))
        for r in rows
    )
    if learned:
        from scripts.train_rehearsal_typeface import render_face, FONTS, MODEL
        from ilm.visual_lm.glyph_typeface_reader import load_typeface_reader

        reader = load_typeface_reader(MODEL, palette[2:], "cpu")
        font_file = FONTS / "truetype/dejavu/DejaVuSerifCondensed.ttf"
        condensed = render_face(font_file, 0)
        covered = {
            c: condensed.tile(c)
            for c in symbols
            if c.isspace() or ord(c) in condensed.coverage
        }
        variants = {"condensed": covered}
        for tag, glyphs in (
            ("original", normal),
            ("serif", serif),
            ("four_flips", speckle),
            ("condensed", covered),
        ):
            ordered = list(glyphs)
            packed = np.frombuffer(
                b"".join(glyphs[c] for c in ordered), np.uint8
            ).reshape(-1, 128)
            variants[tag + "_learned"] = dict(
                zip(
                    ordered,
                    map(bytes, reader.transport(packed, palette[2:])),
                    strict=True,
                )
            )
    conditions = {}
    per_case = {}
    sample = WORK / "typography"
    sample.mkdir(exist_ok=True)
    for tag, glyphs in variants.items():
        bank = list(map(bytes, palette))
        ids = {b: i for i, b in enumerate(bank) if i >= 2}
        for b in glyphs.values():
            if b not in ids:
                ids[b] = len(bank)
                bank.append(b)
        features = torch.from_numpy(
            observations(
                {"tiles": np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128)},
                "ink",
            )
        )
        counts = defaultdict(Counter)
        decisions = [None] * len(rows)
        for t in (0, 1):
            selected = [
                i
                for i, r in enumerate(rows)
                if r["t"] == t and all(c in glyphs for c in r["source"])
            ]
            for batch in range(0, len(selected), 48):
                indices = selected[batch : batch + 48]
                inputs = [[ids[glyphs[c]] for c in rows[i]["source"]] for i in indices]
                pred = model.generate(features, pack(inputs, "cpu"), inventories[t], t)
                for i, y, source in zip(indices, pred, inputs, strict=True):
                    r = rows[i]
                    out, stop = stopped(y)
                    canonical = answers[i]["rehearsal"]
                    count = counts[r["group"]]
                    same = stop == canonical["stop"] and out == canonical["cells"]
                    exact = stop and out == r["target"]
                    count.update(
                        n=1,
                        exact=int(exact),
                        same_reply=int(same),
                        stopped=int(stop),
                        input_cells=len(source),
                        restored_cells=sum(
                            a == b for a, b in zip(source, r["input"], strict=True)
                        ),
                    )
                    decisions[i] = dict(same=bool(same), exact=bool(exact), stop=stop)
                    if i == 241:
                        save_grid(
                            b"".join(bank[a] for a in source),
                            sample / (tag + "-input.png"),
                        )
                        save_grid(
                            palette[out].tobytes(), sample / (tag + "-output.png")
                        )
        start = next(i for i, r in enumerate(rows) if r["group"].startswith("role-"))
        whole = sum(
            all(v is not None and v["exact"] for v in decisions[i : i + 8])
            for i in range(start, len(rows), 8)
        )
        conditions[tag] = dict(
            groups=counts,
            whole_role_families=whole,
            eligible_role_families=sum(
                all(v is not None for v in decisions[i : i + 8])
                for i in range(start, len(rows), 8)
            ),
            unique_source_shapes=len(set(glyphs.values())),
        )
        per_case[tag] = decisions
    if learned:
        if not all(r["same"] for r in per_case["original_learned"]):
            raise ValueError("Original answers changed")
        dump(WORK / "reader-decisions.json", per_case)
        dump(
            PUB / "typeface-reader.json",
            dict(
                conditions=conditions,
                decisions=sha(WORK / "reader-decisions.json"),
                generated=sha(WORK / "generated.json"),
                fit=sha(MODEL / "fit.json"),
                confirmatory_font=sha(font_file),
            ),
        )
        print({k: v["whole_role_families"] for k, v in conditions.items()}, flush=True)
        return
    if any(not r["same"] for r in per_case["original"]):
        raise ValueError("Canonical CPU replies differ from the original GPU run")
    unique_palette_incidents = {}
    for tag, vs in (("serif", serif), ("four_flips", speckle)):
        counts = Counter()
        for c, b in vs.items():
            got = guard.resolve(b)
            counts["already_exact"] += b == normal[c]
            counts["corrected"] += got != b and got == normal[c]
            counts["wrong_snap"] += got != b and got != normal[c]
            counts["unchanged_error"] += got == b and got != normal[c]
        unique_palette_incidents[tag] = counts
    dump(WORK / "font-decisions.json", per_case)
    dump(
        PUB / "typography.json",
        dict(
            source=sha(WORK / "generated.json"),
            font_sha256=sha(SERIF),
            conditions=conditions,
            unique_input_symbols=len(symbols),
            input_shape_outcomes=unique_palette_incidents,
            cases=sha(WORK / "font-decisions.json"),
        ),
    )
    print({k: v["whole_role_families"] for k, v in conditions.items()}, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", action="store_true")
    main(parser.parse_args().learned)
