"""Compare matched historical inputs and physical outputs across ink conventions."""

import hashlib
from collections import defaultdict

import numpy as np

from scripts.evaluate_paired_student import read_streams
from scripts.prepare_visual_event_roles import OUT, ROOT, load, sha, write_new


def audit():
    first = OUT / "light_strokes"
    forms = load(OUT / "historical/forms.json")
    original = load(first / "historical/forms.json")
    assert forms.keys() == original.keys()
    entries = load(OUT / "historical/rows.json")
    assert entries == load(first / "historical/rows.json")
    replacements = {}
    for row, old, new in zip(
        entries,
        read_streams("source", first / "historical"),
        read_streams("source", OUT / "historical"),
        strict=True,
    ):
        for who in ("first", "second"):
            char = row[who]
            index = row["source"].index(char)
            form = forms[char]
            assert (
                form["sha256"]
                == original[char]["sha256"]
                == sha(ROOT.parent / "incoder" / form["source_local_path"])
            )
            assert hashlib.sha256(old[index]).hexdigest() == form["raster_sha256"]
            assert hashlib.sha256(new[index]).hexdigest() == form["ink_sha256"]
            assert np.array_equal(np.bitwise_not(old[index]), new[index])
            replacements[bytes(old[index])] = bytes(new[index])
    from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg

    for char, form in forms.items():
        path = ROOT.parent / "incoder" / form["source_local_path"]
        assert form["sha256"] == original[char]["sha256"] == sha(path)
        paper = np.packbits(render_historic_svg(path).numpy().reshape(-1))
        ink = np.bitwise_not(paper)
        assert (
            hashlib.sha256(paper).hexdigest()
            == form["raster_sha256"]
            == original[char]["raster_sha256"]
        )
        assert hashlib.sha256(ink).hexdigest() == form["ink_sha256"]
        replacements[bytes(paper)] = bytes(ink)
    assert len(replacements) == len(forms)
    equivalent = defaultdict(dict)
    for group, names in {
        "historical": (
            "source",
            "reference",
            "composer",
            "parent",
            "replay",
            "swapped",
        ),
        "binding": ("frequency", "context"),
        "predicates": ("source", "composer", "swapped", "context"),
    }.items():
        for name in names:
            a = read_streams(name, first / group)
            b = read_streams(name, OUT / group)
            count = sum(
                tuple(replacements.get(bytes(c), bytes(c)) for c in x)
                == tuple(map(bytes, y))
                for x, y in zip(a, b, strict=True)
            )
            equivalent[group][name] = dict(equal=count, total=len(a))
            if name in ("source", "reference"):
                assert count == len(a)
    assert sha(OUT / "binding/binding.msgpack.gz") == sha(
        first / "binding/binding.msgpack.gz"
    )
    read_differences = {}
    for group in ("binding", "predicates"):
        old = {
            (r["source"], c)
            for r in load(first / group / "review.json")
            for c in r["choices"]
        }
        current = [
            (r["source"], c)
            for r in load(OUT / group / "review.json")
            for c in r["choices"]
        ]
        read_differences[group] = dict(
            inherited=sum(pair in old for pair in current),
            new=sum(pair not in old for pair in current),
        )
    dst = ROOT / "references/evidence/event_ink_polarity_20261119.json"
    write_new(
        dst,
        dict(
            matched_forms=len(forms),
            forms={k: v["label"] for k, v in forms.items()},
            equivalent=equivalent,
            review_choices=read_differences,
            same_binding_sha256=sha(OUT / "binding/binding.msgpack.gz"),
            initial=load(first / "binding/summary.json")["historical"],
            corrected=load(OUT / "binding/summary.json")["historical"],
            discarded_exemplar_selection="Preliminary dark-ink export screened packed bytes as booleans and changed eight exemplars. Kept separately. Only the matched original exemplars enter the reported comparison.",
        ),
    )
    print(dst)


if __name__ == "__main__":
    audit()
