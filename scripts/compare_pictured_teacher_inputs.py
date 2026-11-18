"""Compare one local foundation model on typed sources, pictured sources and transcription."""

import base64
import fcntl
import io
import random
import time

from PIL import Image
from rapidfuzz.distance import Levenshtein

from scripts.local_translation_teacher import SYSTEM, local_teacher
from scripts.report_clause_support import seq
from scripts.infer_visual_translation import page
from scripts.train_pictured_revision import ROOT, OUT, load, sha, write_new

MODEL = "qwen3-vl:8b-instruct-q8_0"


def run():
    p = OUT / "vision_teacher"
    p.mkdir()
    e = OUT / "evaluation"
    packets = load(e / "packets.json")
    source = {
        m["row"]: r["source"]
        for r, m in zip(load(e / "review.json"), load(e / "mapping.json"))
    }
    records = []
    review = []
    mapping = []
    rng = random.Random(91704)
    started = time.monotonic()
    with local_teacher(MODEL, p) as (ask, digest, size):
        for i in sorted(packets["all"]):
            image = page(seq(e, "source", i))
            w, h = image.size
            image = image.resize((2 * w, 2 * h), Image.Resampling.NEAREST)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            payload = base64.b64encode(buf.getvalue()).decode()
            record = {}
            modes = ["typed", "image", "transcribe"]
            rng.shuffle(modes)
            for mode in modes:
                system = (
                    SYSTEM
                    if mode != "transcribe"
                    else "逐字抄录图片中的原文，保留标点。只写原文。"
                )
                user = dict(role="user", content=source[i] if mode == "typed" else "")
                if mode != "typed":
                    user["images"] = [payload]
                record[mode] = ask(
                    system,
                    user,
                    options=dict(
                        seed=331, temperature=0, num_ctx=4096, num_predict=160
                    ),
                )
            actual = "".join(record["transcribe"]["text"].split())
            expected = "".join(source[i].split())
            records.append(
                dict(
                    row=i,
                    responses=record,
                    ocr_edits=Levenshtein.distance(actual, expected),
                    source_length=len(expected),
                )
            )
            options = list(
                dict.fromkeys([record["typed"]["text"], record["image"]["text"]])
            )
            rng.shuffle(options)
            review.append(dict(source=source[i], choices=options))
            mapping.append(
                dict(
                    row=i,
                    options={
                        k: options.index(record[k]["text"]) for k in ("typed", "image")
                    },
                )
            )
    for n, v in [("responses", records), ("review", review), ("mapping", mapping)]:
        write_new(p / (n + ".json"), v)
    write_new(
        p / "receipt.json",
        dict(
            model=MODEL,
            manifest=digest,
            count=len(records),
            seconds=time.monotonic() - started,
            review_sha256=sha(p / "review.json"),
            bytes=size,
            source_png_original=sha(e / "source.npz"),
            source_characters=sum(r["source_length"] for r in records),
            transcription_edits=sum(r["ocr_edits"] for r in records),
            transcription_exact=sum(r["ocr_edits"] == 0 for r in records),
            length_stops={
                k: sum(r["responses"][k]["stop"] == "length" for r in records)
                for k in ("typed", "image", "transcribe")
            },
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
