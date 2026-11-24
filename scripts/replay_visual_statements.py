"""Replay held pages from PNG and change single facts with the question fixed."""

from hashlib import sha256
import gzip
import json
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_image, read_text
from scripts.evaluate_visual_discourse import draw
from scripts.prepare_visual_statements import MODEL, PUB, WORK, read, dump
from scripts.write_visual_discourse import Writer, encode_pages
from scripts.write_visual_statements import editor, realize
from scripts.write_paired_evidence import reader, generate


@torch.inference_mode()
def main():
    torch.set_num_threads(3)
    w = Writer(device="cpu")
    rt = reader("evidence", "cpu")
    e = editor("cpu")
    space = next(i for i, v in enumerate(w.palette) if i >= 2 and not v.any())
    held = read(WORK / "fresh.json")
    baseline = json.loads(
        gzip.decompress((PUB / "trained-answers.json.gz").read_bytes())
    )
    verified = []

    def write(dest, q, source, ref, expected=None):
        dest.mkdir(parents=True, exist_ok=True)
        inputs = []
        for k, t in (("question", q), ("passage", source)):
            imgs = read_text(t, DEFAULT_FONT)
            draw(dest, k, imgs)
            inputs.append(read_image(dest / (k + ".png"), len(imgs)))
        ink, (a, b) = encode_pages(w.palette, inputs)
        x = torch.from_numpy(observations({"tiles": ink}, "ink"))
        ((fact, fs),) = generate(w.model, rt, x, len(w.palette), [a], [b], [b + a], 96)
        ((out, stop),) = realize(w.model, e, x, len(w.palette), [a], [fact], space)

        def save(k, ids):
            s = ink[np.asarray(ids, dtype=int)]
            draw(dest, k, s)
            return "".join(
                w.transcript[i] if i < len(w.transcript) else "�" for i in ids
            )

        t = save("answer", out)
        f = save("fact", fact)
        h = sha256(ink[np.asarray(out, dtype=int)].tobytes()).hexdigest()
        if expected is not None and (
            h != expected["shape_sha256"] or stop != expected["stopped"]
        ):
            raise ValueError("CPU and GPU answers differ at " + str(dest))
        draw(dest, "reference", read_text(ref, DEFAULT_FONT))
        row = dict(
            question=q,
            question_cells=len(a),
            passage_cells=len(b),
            fact_cells=len(fact),
            written=len(out),
            output=t,
            stopped=stop,
            emitted_fact=f,
            fact_stopped=fs,
            reference=ref,
            shape_sha256=h,
        )
        dump(dest / "readout.json", row)
        verified.append(
            dict(case=dest.name, gpu_comparison=expected is not None, **row)
        )

    r = held[12]
    assert r["question"] == "Who designed Salamanca?"
    public = MODEL / "examples/public"
    for name, source, old, new in (
        (
            "author",
            "Mira drew the map. Lena wrote the account of the bronze inscriptions. Oliver printed the plates.",
            "Lena",
            "Mira",
        ),
        (
            "place",
            "A fossil was found in Niger. The museum later displayed it in France.",
            "Niger",
            "Nepal",
        ),
        (
            "school",
            "The library opened in 1640. The school was founded in 1635. Its first classroom was on the hill.",
            "1635",
            "1735",
        ),
        ("locomotive", r["source"], "Matthew Murray", "Daniel Harper"),
    ):
        p = MODEL / "examples/intervention"
        if name == "locomotive":
            q = r["question"]
            control = baseline["fresh"][12]
        else:
            q = read(public / name / "readout.json")["question"]
            control = next(x for x in baseline["public"] if x["id"] == name)
        assert source.count(old) == 1
        write(p / (name + "-original"), q, source, old, control)
        write(p / (name + "-edited"), q, source.replace(old, new), new)
        assert (p / (name + "-original") / "question.png").read_bytes() == (
            p / (name + "-edited") / "question.png"
        ).read_bytes()
    dump(PUB / "physical-replays.json", verified)


if __name__ == "__main__":
    main()
