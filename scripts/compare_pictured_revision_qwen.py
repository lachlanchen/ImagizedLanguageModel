"""Translate the blinded revision passages with the local pretrained Qwen3 8B text model."""

import fcntl
import random
import time

from scripts.train_pictured_revision import ROOT, OUT, load, sha, write_new

from scripts.local_translation_teacher import SYSTEM, local_teacher

MODEL = "qwen3:8b-q8_0"


def compare():
    directory = OUT / "qwen"
    directory.mkdir()
    blind = load(OUT / "evaluation/review.json")
    keys = load(OUT / "evaluation/mapping.json")
    outputs = []
    review = []
    mapping = []
    rng = random.Random(92191)
    start = time.monotonic()
    with local_teacher(MODEL, directory) as (ask, digest, size):
        for row, key in zip(blind, keys):
            result = ask(
                SYSTEM,
                dict(role="user", content=row["source"]),
                think=False,
                options=dict(seed=331, temperature=0, num_ctx=2048, num_predict=160),
            )
            value = result.pop("text")
            outputs.append(dict(row=key["row"], response=value, **result))
            alts = list(dict.fromkeys(row["choices"] + [value]))
            rng.shuffle(alts)
            assignments = {
                k: alts.index(row["choices"][v]) for k, v in key["options"].items()
            }
            assignments["qwen"] = alts.index(value)
            mapping.append(dict(row=key["row"], options=assignments))
            review.append(dict(source=row["source"], choices=alts))
    write_new(directory / "translations.json", outputs)
    write_new(directory / "review.json", review)
    write_new(directory / "mapping.json", mapping)
    write_new(
        directory / "receipt.json",
        dict(
            model=MODEL,
            manifest=digest,
            bytes=size,
            count=len(review),
            seconds=time.monotonic() - start,
            source_packet=sha(OUT / "evaluation/review.json"),
            prompt=SYSTEM,
            tokens=sum(r["tokens"] for r in outputs),
            length_stops=sum(r["stop"] == "length" for r in outputs),
            review_sha256=sha(directory / "review.json"),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compare()
