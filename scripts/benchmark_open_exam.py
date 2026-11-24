"""Compare free responses on identical printed questions and calibrate a separate meaning judge."""

import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
from time import monotonic
import numpy as np
import requests
from scripts.benchmark_competing_qwen import server, png, MODELS, STORE
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_open_exam import PUB, WORK, read, dump, sha, key
from scripts.prepare_multiple_ink_runs import normalize
from scripts.train_open_exam import corpus
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.write_visual_discourse import Writer

VLM = "qwen3-vl:8b-instruct-q4_K_M"
JUDGE = "qwen3:8b-q4_K_M"


def ask(url, model, msg, schema=None):
    payload = dict(
        model=model,
        messages=[msg],
        stream=False,
        keep_alive="5m",
        options=dict(temperature=0, seed=731231, num_ctx=4096, num_predict=128),
    )
    if schema:
        payload.update(format=schema, think=False)
    r = requests.post(url + "/api/chat", json=payload, timeout=240)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"].strip(), d.get("done_reason") == "stop"


def main(response_dir=WORK, work=WORK, public=PUB):
    work.mkdir(exist_ok=True, parents=True)
    public.mkdir(exist_ok=True, parents=True)
    rows, tiles, n = corpus("held")
    chosen = []
    for genre in ("d", "m"):
        docs = sorted(
            {r["doc"] for r in rows if r["genre"] == genre and r["cohort"] == "paired"},
            key=key,
        )[:6]
        chosen.extend(r for r in rows if r["doc"] in docs)
    assert len(chosen) == 24
    writer = Writer()
    text = {i: c for i, c in enumerate(writer.transcript)}
    for r in rows:
        for ix, c in zip(r["pi"] + r["qi"], r["p"] + r["q"], strict=True):
            text.setdefault(ix, c)
        for ys, v in zip(r["ys"], r["opts"], strict=True):
            for ix, c in zip(ys[:-1], v, strict=True):
                text.setdefault(ix, c)
    own = {}
    for a in ("parent", "matched", "exchanged"):
        rr = {
            r["id"]: r
            for r in unpack(response_dir / (a + "-responses.json.gz"))
            if r["condition"] == "intact"
        }
        own[a] = [
            dict(
                text="".join(text.get(i, "�") for i in rr[r["id"]]["ids"]),
                stop=rr[r["id"]]["stop"],
                exact=rr[r["id"]]["exact"],
            )
            for r in chosen
        ]
    for m in (VLM, JUDGE):
        if sha(STORE / "blobs" / ("sha256-" + MODELS[m])) != MODELS[m]:
            raise ValueError("Local model hash changed")
    selected = work / "comparison.json.gz"
    report = public / "qwen.json"
    if not report.exists():
        with gpu_owner(), server() as url:
            # Load before timing the real visual prompts.
            ask(url, VLM, dict(role="user", content="只回答好。"))
            meter = snapshot()
            start = monotonic()

            def reply(r):
                a, stop = ask(
                    url,
                    VLM,
                    dict(
                        role="user",
                        content="第一张图是文章，第二张图是问题。根据文章用中文给出最简短的完整答案。不要解释，只给答案。",
                        images=[png(tiles, r["pi"]), png(tiles, r["qi"])],
                    ),
                )
                return dict(
                    text=a,
                    stop=stop,
                    exact=stop and a == r["answer"],
                    normalized=stop and normalize(a) == normalize(r["answer"]),
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                vals = list(pool.map(reply, chosen))
            elapsed = monotonic() - start
            energy = consumed(meter, snapshot())
            assert any(
                r["name"].casefold() == VLM.casefold() and r["size_vram"] > 0
                for r in requests.get(url + "/api/ps", timeout=15).json()["models"]
            )
        own["qwen"] = vals
        packed(selected, own)
        dump(
            report,
            dict(
                sha256=sha(selected),
                weights=MODELS[VLM],
                n=len(chosen),
                seconds=elapsed,
                joules=energy["joules"],
                ids=[r["id"] for r in chosen],
                strict={k: sum(v["exact"] for v in vs) for k, vs in own.items()},
                qwen_normalized=sum(v["normalized"] for v in vals),
            ),
        )
    if sha(selected) != read(report)["sha256"]:
        raise ValueError("Changed comparison")
    own = unpack(selected)
    types = ["matched", "exchanged", "qwen", "reference", "distractor"]
    items = []
    for i, r in enumerate(chosen):
        names = list(
            np.random.default_rng(int.from_bytes(key(r["id"])[:8], "big")).permutation(
                types
            )
        )
        for order in (names, names[::-1]):
            choices = {a: own[a][i]["text"] for a in types[:3]}
            choices.update(
                reference=r["answer"],
                distractor=next(o for o in r["opts"] if o != r["answer"]),
            )
            items.append(
                (
                    r,
                    list(order),
                    {k: choices[a] for k, a in zip("ABCDE", order, strict=True)},
                )
            )
    schema = dict(
        type="object",
        properties={k: dict(type="boolean") for k in "ABCDE"},
        required=list("ABCDE"),
        additionalProperties=False,
    )

    def assess(item):
        r, order, choices = item
        content = (
            "按原文、问题、标准答案独立评判每个匿名回答是否完整、正确、切题。意思相同的改写可以，泛泛应答和只对一部分算错。按给定 JSON 输出。"
            + json.dumps(
                dict(
                    passage=r["p"],
                    question=r["q"],
                    reference=r["answer"],
                    responses=choices,
                ),
                ensure_ascii=False,
            )
        )
        s, stop = ask(url, JUDGE, dict(role="user", content=content), schema)
        obj = json.loads(s)
        if not stop or any(type(obj.get(k)) is not bool for k in "ABCDE"):
            raise ValueError("Invalid meaning review")
        return dict(
            id=r["id"], ratings={a: obj[k] for k, a in zip("ABCDE", order, strict=True)}
        )

    with gpu_owner(), server() as url:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(assess, items))
    counts = {}
    for a in types:
        pairs = [
            (u["ratings"][a], v["ratings"][a])
            for u, v in zip(outcomes[::2], outcomes[1::2], strict=True)
        ]
        counts[a] = dict(
            both=sum(x and y for x, y in pairs),
            either=sum(x or y for x, y in pairs),
            agreement=sum(x == y for x, y in pairs),
        )
    agreement = sum(v["agreement"] for v in counts.values())
    packed(work / "semantic.json.gz", outcomes)
    dump(
        public / "meaning.json",
        dict(
            judge=MODELS[JUDGE],
            n=len(chosen),
            counts=counts,
            agreement=agreement,
            qualified=counts["reference"]["both"] >= 22
            and counts["distractor"]["either"] <= 2
            and agreement >= 108,
            analysis="secondary blinded assessment after selection of checkpoints",
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--responses", type=Path, default=WORK)
    p.add_argument("--output", type=Path, default=WORK / "benchmark-replication")
    a = p.parse_args()
    main(a.responses, a.output, a.output)
