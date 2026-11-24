"""Qualify a local text teacher and collect fitting-only grounded dialogue answers."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

from scripts.dialogue_ink_data import read_data
from scripts.evaluate_optical_passage_qa import normalized, metrics
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_dialogue_ink import ROOT, WORK as BASE, dump, sha
from scripts.score_dialogue_ink_references import official_index
from scripts.train_span_choice import gpu_owner

WORK = ROOT / "artifacts/dialogue_teacher_v1"
PUB = ROOT / "references/evidence/dialogue_teacher_v1"
WEIGHT_HASH = "163553aea1b1de62de7c5eb2ef5afb756b4b3133308d9ae7e42e951d8d696ef5"
DEFAULT_MODEL = (
    ROOT.parent / "LocalLLM/.local/models/ollama/blobs" / ("sha256-" + WEIGHT_HASH)
)
DEFAULT_SERVER = (
    ROOT.parent / "LocalLLM/.local/build/llama.cpp-b10327-cuda-sm89/bin/llama-server"
)
INSTRUCTION = (
    "Read the passage and answer the current question using the previous exchange when relevant. "
    "Give the shortest natural answer that completely answers the question. Use yes or no for a yes/no question. "
    "If the passage cannot answer it, write unknown. Copy one short contiguous quote from the passage supporting "
    "your answer; use an empty quote if there is no support. Return JSON with string fields answer and quote."
)
SCHEMA = {
    "type": "object",
    "properties": {k: {"type": "string"} for k in ("answer", "quote")},
    "required": ["answer", "quote"],
    "additionalProperties": False,
}


def rank(row):
    return hashlib.sha256(f"dialogue-teacher-1/{row[1]}/{row[2]}".encode()).digest()


def hash_value(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def quote_spans(text, quote):
    if not quote:
        return []
    found = list(re.finditer(r"(?<!\w)" + re.escape(quote) + r"(?!\w)", text))
    words = list(re.finditer(r"\S+", text))
    return [
        [ids[0], ids[-1]]
        for m in found
        if (
            ids := [
                i
                for i, w in enumerate(words)
                if w.end() > m.start() and w.start() < m.end()
            ]
        )
    ]


def rpc(url, data=None, timeout=100):
    req = urllib.request.Request(
        url,
        data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


@contextmanager
def server(binary, model, wrapper, log):
    if sha(model) != WEIGHT_HASH:
        raise ValueError("Different Qwen 4B teacher")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = ([str(wrapper)] if wrapper else []) + [
        str(binary),
        "-m",
        str(model),
        "--alias",
        "ilm-grounded-teacher",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-ngl",
        "99",
        "-c",
        "24576",
        "-np",
        "12",
        "-t",
        "4",
        "-b",
        "2048",
        "-ub",
        "512",
        "--reasoning",
        "off",
        "--offline",
    ]
    with log.open("ab") as file:
        process = subprocess.Popen(
            command,
            stdout=file,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Teacher exited {process.returncode}, see {log}"
                    )
                try:
                    if rpc(url + "/health", timeout=2).get("status") == "ok":
                        break
                except OSError:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("Teacher did not become ready")
            if process.poll() is not None or not any(
                m["id"] == "ilm-grounded-teacher"
                for m in rpc(url + "/v1/models")["data"]
            ):
                raise ValueError("Teacher identity mismatch")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def request(url, item, order):
    fields = item["fields"]
    payload = (
        dict(passage=fields[0], previous_exchange=fields[2], current_question=fields[1])
        if order
        else dict(zip(("passage", "question", "previous_exchange"), fields))
    )
    prompt = (
        "<|im_start|>system\n"
        + INSTRUCTION
        + "<|im_end|>\n<|im_start|>user\n"
        + json.dumps(payload, ensure_ascii=False)
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    response = rpc(
        url + "/completion",
        {
            "prompt": prompt,
            "temperature": 0,
            "seed": 420121,
            "n_predict": 144,
            "json_schema": SCHEMA,
            "stop": ["<|im_end|>"],
            "cache_prompt": True,
        },
    )
    ended = (
        response.get("stop_type") in ("eos", "word")
        or response.get("stopped_eos")
        or response.get("stopped_word")
    )
    try:
        parsed = json.loads(response["content"])
        if not all(isinstance(parsed[k], str) for k in ("answer", "quote")):
            raise ValueError()
        answer, quote = (" ".join(parsed[k].split()) for k in ("answer", "quote"))
    except (ValueError, KeyError, TypeError):
        answer = quote = ""
    return dict(
        id=item["id"],
        input=hash_value([order, fields]) if order else hash_value(fields),
        stopped=bool(ended),
        answer=answer,
        quote=quote,
        spans=quote_spans(fields[0], quote),
        tokens=response["tokens_predicted"],
    )


def obtain(url, items, file, order=False):
    prior = {}
    if file.exists():
        with file.open("rb+") as cache:
            while True:
                pos = cache.tell()
                line = cache.readline()
                if not line:
                    break
                try:
                    r = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if line.endswith(b"\n"):
                        raise
                    # Only the final, unterminated record can be a partial write.
                    cache.truncate(pos)
                    break
                if r["id"] in prior:
                    raise ValueError("Duplicate cached response")
                prior[r["id"]] = r
                if not line.endswith(b"\n"):
                    cache.write(b"\n")
    expected = {item["id"] for item in items}
    if len(expected) != len(items) or not prior.keys() <= expected:
        raise ValueError("Different cached split")
    todo = []
    for item in items:
        if item["id"] not in prior:
            todo.append(item)
        elif prior[item["id"]]["input"] != (
            hash_value([order, item["fields"]]) if order else hash_value(item["fields"])
        ):
            raise ValueError("Different cached input")
    start = time.monotonic()
    meter = snapshot()
    with file.open("a") as dst, ThreadPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(lambda v: request(url, v, order), todo), 1):
            prior[r["id"]] = r
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
            dst.flush()
            if i % 128 == 0:
                print(
                    file.stem,
                    i,
                    "/",
                    len(todo),
                    round(time.monotonic() - start, 1),
                    flush=True,
                )
    return prior, dict(
        seconds=time.monotonic() - start,
        energy=consumed(meter, snapshot()),
        new=len(todo),
        cached=len(items) - len(todo),
        scope="new requests in this invocation; excludes server startup",
    )


def qualifications(data):
    rows = [r for r in data["rows"] if r[0] == "dev"]
    old = [
        r
        for r in json.loads(
            gzip.decompress((BASE / "evaluation/decisions.json.gz").read_bytes())
        )
        if r["group"] == "dev"
    ]
    evaluator, lookup, _, _, _ = official_index()
    hard = []
    regular = []
    markers = {}
    for i, (r, e) in enumerate(zip(rows, old, strict=True)):
        marker = lookup[r[1], r[2], r[3], r[4], r[5]]
        markers[i] = marker
        fields = [" " + normalized(s) + " " for s in (data["pages"][r[1]], r[3], r[4])]
        if e["strict"] and all(
            len(normalized(a).split()) >= 2
            and all(" " + normalized(a) + " " not in field for field in fields)
            for a in evaluator.gold_data[marker]
        ):
            hard.append(i)
        else:
            regular.append(i)
    if len(hard) != 83:
        raise ValueError("Difficult population changed")
    groups = {
        "absent": hard,
        "sample": sorted(regular, key=lambda i: rank(rows[i]))[:128],
    }
    items = []
    for group, indices in groups.items():
        for i in indices:
            for arm, qi, pi in [
                ("original", i, i),
                ("question", old[i]["question_control"], i),
                ("passage", i, old[i]["passage_control"]),
            ]:
                items.append(
                    dict(
                        id=f"{group}/{i}/{arm}",
                        fields=[data["pages"][rows[pi][1]], rows[qi][3], rows[qi][4]],
                    )
                )
    return items, groups, markers, evaluator


def training_items(data):
    rows = [r for r in data["rows"] if r[0] == "train"]
    order = sorted(range(len(rows)), key=lambda i: rank(rows[i]))
    selected = []
    for i in order:
        r = rows[i]
        answer = normalized(r[5])
        if len(answer.split()) >= 2 and all(
            " " + answer + " " not in " " + normalized(s) + " "
            for s in (data["pages"][r[1]], r[3], r[4])
        ):
            selected.append(i)
            if len(selected) == 1536:
                break
    chosen = set(selected)
    selected.extend([i for i in order if i not in chosen][: 3072 - len(selected)])
    return rows, [
        dict(id=str(i), fields=[data["pages"][rows[i][1]], rows[i][3], rows[i][4]])
        for i in selected
    ]


def study(args):
    WORK.mkdir(exist_ok=True)
    PUB.mkdir(exist_ok=True)
    data = read_data()
    items, groups, markers, evaluator = qualifications(data)
    private = WORK / "current_last" if args.current_last else WORK
    public = PUB / "current_last" if args.current_last else PUB
    private.mkdir(exist_ok=True)
    public.mkdir(exist_ok=True)
    if args.current_last:
        from scripts.dialogue_teacher_stories import new_stories, absent
        from scripts.evaluate_dialogue_ink import exchanges

        fresh, ms, _ = new_stories(data, "typography")
        rows = fresh["rows"]
        swapq, swapp = exchanges(rows)
        test = [
            i
            for i, r in enumerate(rows)
            if absent(evaluator.gold_data[ms[i]], (fresh["pages"][r[1]], r[3], r[4]))
        ]
        groups["replication"] = [i + len(markers) for i in test]
        for offset, i in zip(groups["replication"], test):
            markers[offset] = ms[i]
            for arm, qi, pi in [
                ("original", i, i),
                ("question", swapq[i], i),
                ("passage", i, swapp[i]),
            ]:
                items.append(
                    dict(
                        id=f"replication/{offset}/{arm}",
                        fields=[fresh["pages"][rows[pi][1]], rows[qi][3], rows[qi][4]],
                    )
                )
    summary_file = public / "qualification.json"
    summary = None
    model_sha = sha(args.model)
    binary_sha = sha(args.binary)
    if model_sha != WEIGHT_HASH:
        raise ValueError("Different model")
    binding = dict(
        model=model_sha,
        binary=binary_sha,
        prompt=INSTRUCTION,
        current_last=args.current_last,
        schema=SCHEMA,
        n_predict=144,
        temperature=0,
        seed=420121,
    )
    bound = private / "request_binding.json"
    if bound.exists() and json.loads(bound.read_text()) != binding:
        raise ValueError("Changed request binding")
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if (
            summary["prompt"],
            summary["model"],
            summary["binary"],
            summary.get("current_question_last", False),
            summary["requests"],
            summary["decisions"],
        ) != (
            INSTRUCTION,
            model_sha,
            binary_sha,
            args.current_last,
            len(items),
            sha(private / "qualification.jsonl"),
        ):
            raise ValueError("Changed completed qualification")
        if not summary["admitted"]:
            return
        if (public / "labels.json").exists():
            labels = json.loads((public / "labels.json").read_text())
            if (labels["decisions"], labels["accepted_sha"]) != (
                sha(private / "training.jsonl"),
                sha(private / "accepted.json"),
            ):
                raise ValueError("Changed accepted teacher data")
            return
    if not bound.exists():
        if summary is None and (private / "qualification.jsonl").exists():
            raise ValueError("Interrupted cache has no known request binding")
        dump(bound, binding)
    with server(args.binary, args.model, args.wrapper, private / "server.log") as url:
        if summary is None:
            records, cost = obtain(
                url, items, private / "qualification.jsonl", args.current_last
            )
            scores = {}
            for name, indices in groups.items():
                scores[name] = {}
                for arm in ("original", "question", "passage"):
                    totals = Counter()
                    for i in indices:
                        r = records[f"{name}/{i}/{arm}"]
                        marker = markers[i]
                        valid = r["stopped"] and bool(r["answer"])
                        if valid:
                            score = evaluator.compute_turn_score(*marker, r["answer"])
                            totals.update(score)
                            totals["exact"] += any(
                                normalized(r["answer"]) == normalized(a)
                                for a in evaluator.gold_data[marker]
                            )
                        totals["quoted"] += bool(valid and r["spans"])
                        totals["finished"] += valid
                    scores[name][arm] = dict(
                        n=len(indices),
                        f1=totals["f1"] / len(indices),
                        em=totals["em"] / len(indices),
                        any_exact=totals["exact"],
                        quoted=totals["quoted"],
                        finished=totals["finished"],
                    )
            h = scores["absent"]
            a = h["original"]
            passed = (
                a["any_exact"] >= 3
                and a["f1"] >= 0.35
                and min(a["f1"] - h[k]["f1"] for k in ("question", "passage")) >= 0.08
                and 2 * a["quoted"] >= a["n"]
            )
            if args.current_last:
                rep = scores["replication"]
                v = rep["original"]
                passed = (
                    passed
                    and v["any_exact"] >= 1
                    and v["f1"] >= 0.35
                    and min(v["f1"] - rep[k]["f1"] for k in ("question", "passage"))
                    >= 0.08
                    and 2 * v["quoted"] >= v["n"]
                )
            meta = dict(
                current_question_last=args.current_last,
                scores=scores,
                admitted=passed,
                requests=len(items),
                model=WEIGHT_HASH,
                binary=sha(args.binary),
                prompt=INSTRUCTION,
                decisions=sha(private / "qualification.jsonl"),
                cost=cost,
            )
            dump(public / "qualification.json", meta)
            print(meta, flush=True)
            if not passed:
                return
        rows, items = training_items(data)
        responses, cost = obtain(
            url, items, private / "training.jsonl", args.current_last
        )
    alphabet = {c for r in rows for c in r[5]}
    accepted = {}
    reasons = Counter()
    for key, r in responses.items():
        ans = r["answer"]
        f1 = metrics(ans, [rows[int(key)][5]])[1]
        if not r["stopped"] or not ans:
            reasons["unfinished_or_empty"] += 1
        elif (
            len(ans.split()) > 16
            or max(map(len, ans.split())) > 24
            or not set(ans) <= alphabet
        ):
            reasons["geometry_or_palette"] += 1
        elif len(r["spans"]) != 1:
            reasons["quote_not_unique"] += 1
        elif f1 < 0.35:
            reasons["human_agreement"] += 1
        else:
            accepted[key] = dict(answer=ans, evidence=r["spans"][0])
    dump(private / "accepted.json", accepted)
    dump(
        public / "labels.json",
        dict(
            attempts=len(items),
            accepted=len(accepted),
            excluded=reasons,
            changed_words=sum(
                v["answer"] != rows[int(i)][5] for i, v in accepted.items()
            ),
            decisions=sha(private / "training.jsonl"),
            accepted_sha=sha(private / "accepted.json"),
            cost=cost,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=Path, default=DEFAULT_SERVER)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--wrapper", type=Path)
    p.add_argument("--current-last", action="store_true")
    args = p.parse_args()
    with gpu_owner():
        study(args)
