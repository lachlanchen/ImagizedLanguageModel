"""Index a representative attestation per headword, entry and script from the local plates."""

from collections import defaultdict
import json
import os
import shutil

from scripts.ordered_image_corpus import PLATES
from scripts.train_causal_glyph_writer import sha, record
from scripts.train_typeface_writer import WORK, PUBLIC


def build():
    dest = WORK / "catalogue"
    dest.mkdir()
    receipt = json.loads((PLATES / "report.json").read_text())
    assert (
        not any(receipt["overlap"].values())
        and sha(PLATES / "pixels.npy") == receipt["pixels_sha256"]
    )
    entries = defaultdict(lambda: defaultdict(dict))
    for split in ("train", "development"):
        path = PLATES / (split + "_cards.json")
        assert sha(path) == receipt["splits"][split]["cards_sha256"]
        for card in json.loads(path.read_text()):
            for r in card:
                forms = entries[r["char"]][r["entry"]]
                old = forms.get(r["stage"])
                if old is None or (r["label"], r["svg"]) < (old["label"], old["svg"]):
                    forms[r["stage"]] = {
                        k: r[k] for k in ("label", "svg", "tile", "family")
                    }
                    forms[r["stage"]]["split"] = split
    index = {
        c: [dict(entry=e, forms=forms) for e, forms in sorted(es.items())]
        for c, es in sorted(entries.items())
    }
    try:
        os.link(PLATES / "pixels.npy", dest / "pixels.npy")
    except OSError:
        shutil.copyfile(PLATES / "pixels.npy", dest / "pixels.npy")
    record(
        dest / "catalogue.json",
        dict(pixels_sha256=receipt["pixels_sha256"], entries=index),
    )
    record(
        PUBLIC / "catalogue.json",
        dict(
            headwords=len(index),
            entries=sum(map(len, index.values())),
            stage_examples=sum(len(e["forms"]) for es in index.values() for e in es),
            source_cards=receipt["splits"],
            index_sha256=sha(dest / "catalogue.json"),
            pixels_sha256=receipt["pixels_sha256"],
        ),
    )


if __name__ == "__main__":
    build()
