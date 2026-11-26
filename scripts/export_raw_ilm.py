"""Replay shown image translations and export unchanged physical rasters for the article."""

from hashlib import sha256
from shutil import copyfile
import numpy as np
from PIL import Image
import torch
from scripts.prepare_paired_ink import ROOT, MODELS, WORK, PUB, read, dump, sha
from scripts.write_ordered_translation import restore, infer
from ilm.visual_lm.portable_writer import read_image

PAPER = ROOT / "publication/ilm-nature"
ASSETS = PAPER / "figures/raw_ilm"
EXAMPLES = ROOT / "models/translation_paraphrases/examples"


def export():
    torch.set_num_threads(2)
    ASSETS.mkdir(parents=True, exist_ok=True)
    outputs = read(WORK / "generated.json")
    family = read(WORK / "held.json")[0]
    files = {}
    pairs = []

    def put(tag, path):
        name = tag + ".png"
        copyfile(path, ASSETS / name)
        with Image.open(path) as im:
            w, h = im.size
        files[name] = dict(sha256=sha(path), width=w, height=h)
        return name

    def replay(tag, model, src, n, target, total, stop, t):
        a = put(tag + "-input", src)
        b = put(tag + "-output", target)
        with torch.inference_mode():
            raw, done = infer(restore(model, "cpu"), read_image(src, n), t, "cpu")
        assert raw == read_image(target, total).tobytes() and done == stop
        pairs.append(
            dict(
                checkpoint=model,
                weights=sha(
                    (MODELS if model == "paired-assignment" else EXAMPLES.parent)
                    / ("assignment" if model == "paired-assignment" else "varied")
                    / "weights.safetensors"
                ),
                source=a,
                answer=b,
                source_cells=n,
                emitted_cells=total,
                stop=done,
                packed_ink_sha256=sha256(raw).hexdigest(),
            )
        )

    for i in (54, 55):
        d = EXAMPLES / f"item-{i}"
        rec = read(d / "receipt.json")
        o = rec["outputs"]["paraphrase-varied"]
        replay(
            "role-" + str(i),
            "paraphrase-varied",
            d / "source.png",
            rec["source_cells"],
            d / "paraphrase-varied.png",
            o["cells"],
            o["stopped"],
            1,
        )
    for t in (1, 0):
        j = 2 * t
        out = outputs["out"][j]["assignment"]
        r = outputs["rows"][j]
        assert r["id"] == f"{family['id']}/{t}" and r["variant"] == 0
        d = WORK / "png" / str(family["id"])
        dest = d / f"assignment-0-{t}.png"
        assert sha(dest) == out["image"]
        replay(
            ("natural-en", "natural-zh")[t],
            "paired-assignment",
            d / f"source-0-{1-t}.png",
            len(family["cells"][0][1 - t]),
            dest,
            len(out["cells"]),
            out["stop"],
            t,
        )
    old = read(PAPER / "evidence/ranking_completed_figure_data.json")
    for i, r in enumerate(old["samples"]):
        z = (
            ROOT
            / "artifacts/global_glyph_ranking_v1/qualify/cases"
            / r["case"]
            / "pixels.npz"
        )
        if sha(z) != old["source_sha256"][str(z.relative_to(ROOT))]:
            raise ValueError(z)
        with np.load(z, allow_pickle=False) as arrays:
            for role, key in (("source", "gold_ink"), ("ranking", "candidate_ink")):
                bits = arrays[key][r["position"], 0]
                expected = r["images"][role]["original_ink_sha256"]
                assert (
                    bits.shape == (32, 32)
                    and sha256(bits.tobytes()).hexdigest() == expected
                )
                path = ASSETS / f"pixel-{i+1}-{role}.png"
                Image.fromarray(255 - 255 * bits).save(path)
                files[path.name] = dict(
                    width=32, height=32, sha256=sha(path), original_bits=expected
                )
    dump(
        PUB / "raw_rasters.json",
        dict(
            files=files,
            replays=pairs,
            formal=sha(ROOT / "formal/span_list/PairedInkOdds.lean"),
        ),
    )
    return files


if __name__ == "__main__":
    export()
