"""Serve Qwen3-VL locally for the same 32 originally rendered list questions."""

from concurrent.futures import ThreadPoolExecutor
from time import monotonic
import requests
from scripts.benchmark_competing_qwen import MODELS, STORE, png, server
from scripts.evaluate_multiple_ink_runs import aggregate, corpus, grade
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_multiple_ink_runs import PUB, dump, read, sha, multi
from scripts.train_paired_evidence import gpu_owner

MODEL = "qwen3-vl:8b-instruct-q4_K_M"
PROMPT = (
    "The first image is the passage, the second is a question. Read both. Write all distinct "
    "requested items using the passage wording and the question's language. Separate items with "
    "a vertical bar |. Include no explanation. If the page provides none, return an empty answer."
)


def main():
    rows, tiles = corpus()
    sample = [
        r
        for lang in ("en", "zh")
        for r in [r for r in rows if r["lang"] == lang and multi(r)][:16]
    ]
    out = PUB / "qwen-answers.json.gz"
    report = PUB / "qwen.json"
    if report.exists():
        if sha(out) != read(report)["sha256"]:
            raise ValueError("Published comparison changed")
        return
    if sha(STORE / "blobs" / ("sha256-" + MODELS[MODEL])) != MODELS[MODEL]:
        raise ValueError("Model identity")
    with gpu_owner(), server() as url:
        meta = requests.post(
            url + "/api/show", json={"model": MODEL}, timeout=30
        ).json()
        assert meta["details"]["quantization_level"] == "Q4_K_M"

        def ask(r):
            msg = dict(
                role="user",
                content=PROMPT,
                images=[png(tiles, r["pi"]), png(tiles, r["qi"])],
            )
            resp = requests.post(
                url + "/api/chat",
                json=dict(
                    model=MODEL,
                    stream=False,
                    keep_alive="5m",
                    messages=[msg],
                    options=dict(
                        temperature=0, seed=317091, num_ctx=4096, num_predict=256
                    ),
                ),
                timeout=240,
            )
            resp.raise_for_status()
            data = resp.json()
            answer = data["message"]["content"].strip()
            parts = [p.strip() for p in answer.split("|") if p.strip()]
            score = grade(parts, r["refs"])
            stop = data.get("done_reason") == "stop"
            score["exact"] &= stop
            return dict(
                id=r["id"],
                lang=r["lang"],
                answer=answer,
                parts=parts,
                multi=True,
                stopped=stop,
                visible=any(r["visible_parts"]),
                **score,
            )

        meter = snapshot()
        start = monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            detail = list(pool.map(ask, sample))
        elapsed = monotonic() - start
        energy = consumed(meter, snapshot())
        assert any(
            r["name"].lower() == MODEL.lower() and r["size_vram"] > 0
            for r in requests.get(url + "/api/ps").json()["models"]
        )
    keys = {(r["lang"], r["id"]) for r in sample}
    own = [
        r
        for r in unpack(PUB / "answers.json.gz")
        if (r["lang"], r["id"]) in keys
        and r["mode"] == "count"
        and r["condition"] == "original"
    ]
    assert len(own) == len(detail) == 32
    packed(out, detail)
    dump(
        report,
        dict(
            model=MODEL,
            weights=MODELS[MODEL],
            seconds=elapsed,
            joules=energy["joules"],
            sha256=sha(out),
            qwen=aggregate(detail),
            optical=aggregate(own),
            stopped=sum(d["stopped"] for d in detail),
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
