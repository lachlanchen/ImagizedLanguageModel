#!/usr/bin/env python3
"""Prepare pinned, training-only preference sources; never admit a model fit.

Text is retained for offline rendering and label review. No tokenizer, model,
recognizer or image renderer is imported. Published preference labels remain
unverified; this command does not turn a rejected reward into a qualified one.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import unicodedata
import urllib.request


DATASET = "wenbopan/Chinese-dpo-pairs"
REVISION = "800b8cc6c8a1f7fa17ba1ab5690e50b663d832ca"
REMOTE_FILE = "train/train-00000-of-00001.parquet"
SOURCE_SHA256 = "aad344da8e25527368e0780d86ea1ce1595a2f739cef1e5261adae816ee661c6"
SOURCE_BYTES = 17430997
EXPECTED_ROWS = 10735
FIELDS = ("prompt", "system", "chosen", "rejected", "source", "id")
ROOT = Path(__file__).resolve().parents[1]
EVALUATION_REFS = ROOT / "artifacts/compact_visual_span_evaluation_v1/references.json"
MAX_SECONDS = 90
MAX_RSS_GIB = 1.0


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(text: str) -> str:
    """Conservative exact-query grouping, not semantic decontamination."""
    return "".join(unicodedata.normalize("NFKC", text).split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_budget(start: float) -> None:
    if time.monotonic() - start > MAX_SECONDS:
        raise RuntimeError("Preparation exceeded its 90-second ceiling")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    if rss > MAX_RSS_GIB:
        raise RuntimeError("Preparation exceeded its 1-GiB peak RSS ceiling")
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) / 1024**2 < 24:
        raise RuntimeError("Host available RAM fell below 24 GiB")


def source_file(cache: Path, fetch: bool, start: float) -> Path:
    path = cache / "source.parquet"
    if path.exists():
        if path.stat().st_size != SOURCE_BYTES or digest(path) != SOURCE_SHA256:
            raise ValueError("Existing source differs from the pinned public object")
        return path
    if not fetch:
        raise ValueError("Source absent; use --fetch to download the pinned public file")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / "source.parquet.part"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{REMOTE_FILE}"
    # Exclusive creation preserves partial failures rather than overwriting them.
    with temporary.open("xb") as output, urllib.request.urlopen(url, timeout=20) as response:
        total = 0
        while block := response.read(1024 * 1024):
            total += len(block)
            if total > SOURCE_BYTES:
                raise ValueError("Download exceeds pinned byte count")
            output.write(block)
            check_budget(start)
    if total != SOURCE_BYTES or digest(temporary) != SOURCE_SHA256:
        raise ValueError("Download does not match pinned byte count and SHA-256")
    os.replace(temporary, path)
    return path


def prepare(cache: Path, output: Path, fetch: bool) -> dict:
    start = time.monotonic()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing evidence is preserved")
    check_budget(start)
    reference_hash = digest(EVALUATION_REFS)
    refs = json.loads(EVALUATION_REFS.read_text())["reference_rows"]
    questions = {canonical(r["question"]) for r in refs if r["kind"] == "qa"}
    if len(questions) != 32:
        raise ValueError("Expected all 32 existing QA evaluation questions")
    source = source_file(cache, fetch, start)
    # Serial batches avoid loading a dataset framework, network workers or GPUs.
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != EXPECTED_ROWS or set(parquet.schema.names) != set(FIELDS):
        raise ValueError("Pinned dataset schema or count is unexpected")
    output.mkdir(parents=True)
    temporary = output / "pairs.jsonl.partial"
    counts: Counter = Counter()
    sources: Counter = Counter()
    partitions: Counter = Counter()
    lengths = {k: [] for k in ("prompt", "chosen", "rejected")}
    group_partitions: dict[str, str] = {}
    cap_counts = {str(n): Counter() for n in (32, 128, 513, 640)}
    with temporary.open("x", encoding="utf-8") as stream:
        for batch in parquet.iter_batches(batch_size=128, columns=list(FIELDS), use_threads=False):
            check_budget(start)
            for raw in batch.to_pylist():
                if not all(isinstance(raw[k], str) for k in FIELDS if k != "id") or not (
                    raw["id"] is None or isinstance(raw["id"], str)
                ):
                    raise ValueError("Non-string field; preserve the partial export for inspection")
                row_number = counts["rows"]
                counts["rows"] += 1
                counts["null_source_ids"] += raw["id"] is None
                group = fingerprint(canonical(raw["prompt"]))
                original_partition = "calibration" if int(group[:16], 16) % 10 == 0 else "train"
                overlap = any(q in canonical(raw["system"] + "\n" + raw["prompt"]) for q in questions)
                partition = "quarantine_evaluation_query" if overlap else original_partition
                previous = group_partitions.setdefault(group, original_partition)
                if previous != original_partition:
                    raise AssertionError("Exact normalized prompt crossed proposed partitions")
                empty_prompt = not raw["prompt"].strip()
                empty_chosen = not raw["chosen"].strip()
                empty_rejected = not raw["rejected"].strip()
                same = canonical(raw["chosen"]) == canonical(raw["rejected"])
                counts["empty_prompt"] += empty_prompt
                counts["empty_chosen"] += empty_chosen
                counts["empty_rejected"] += empty_rejected
                counts["identical_normalized_answers"] += same
                counts["evaluation_query_matches"] += overlap
                usable_structure = not (empty_prompt or empty_chosen or empty_rejected or same or overlap)
                counts["structurally_eligible_not_qualified"] += usable_structure
                sources[raw["source"]] += 1
                partitions[partition] += 1
                for field in lengths:
                    lengths[field].append(len(raw[field]))
                for cap, tally in cap_counts.items():
                    n = int(cap)
                    tally["both_answers_fit"] += max(len(raw["chosen"]), len(raw["rejected"])) <= n
                    tally["at_least_one_answer_exceeds"] += max(len(raw["chosen"]), len(raw["rejected"])) > n
                encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                row = {"row": row_number, "row_sha256": fingerprint(encoded), "prompt_group": group,
                       "prospective_partition": partition, "structurally_eligible_not_qualified": usable_structure,
                       "labels_qualified": False, "raw": raw}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    if counts["rows"] != EXPECTED_ROWS or digest(EVALUATION_REFS) != reference_hash:
        raise ValueError("Row count or protected evaluation references changed")
    target = output / "pairs.jsonl"
    os.replace(temporary, target)

    def summary(values: list[int]) -> dict:
        values = sorted(values)
        return {"minimum": values[0], "median": values[len(values)//2],
                "p90": values[int(.9*(len(values)-1))], "maximum": values[-1], "sum": sum(values)}

    check_budget(start)
    report = {
        "schema": "chinese-image-preference-source-preparation-v1",
        "script_sha256": digest(Path(__file__)),
        "dataset": DATASET, "revision": REVISION, "source_file": REMOTE_FILE,
        "source_sha256": SOURCE_SHA256, "source_bytes": SOURCE_BYTES,
        "dataset_card_url": f"https://huggingface.co/datasets/{DATASET}/blob/{REVISION}/README.md",
        "card_license": "mit", "labels": "GPT-3.5-translated preferences from multiple upstream datasets; unqualified",
        "counts": dict(counts), "source_counts": dict(sources), "prospective_partitions": dict(partitions),
        "unique_normalized_prompt_groups": len(group_partitions),
        "grouping": "SHA256(NFKC prompt without whitespace), 10 percent hash buckets reserved for calibration",
        "evaluation_reference_sha256": reference_hash,
        "quarantine_scope": "32 exact normalized QA question substrings in system plus prompt only; not semantic or passage decontamination",
        "length_unit": "Python Unicode code points, not image glyphs, font coverage or language tokens",
        "lengths": {k: summary(v) for k, v in lengths.items()},
        "answer_cap_counts_all_source_rows": {k: dict(v) for k, v in cap_counts.items()},
        "all_source_rows_preserved": True, "source_strings_normalized_in_export": False,
        "answers_truncated": False, "pairs_jsonl_sha256": digest(target),
        "pairs_jsonl_bytes": target.stat().st_size, "elapsed_seconds": time.monotonic()-start,
        "cost_scope": "This invocation only; prior failed preparation and its download are preserved separately, without captured timing or RSS",
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "ceilings": {"seconds": MAX_SECONDS, "rss_gib": MAX_RSS_GIB, "host_available_gib": 24},
        "images_rendered": 0, "model_calls": 0, "optimizer_updates": 0,
        "semantic_labels_qualified": False, "physical_rendering_qualified": False,
        "preference_fit_admitted": False, "changes_running_trial": False, "goal_achieved": False,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/external/chinese_dpo_pairs_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    result = prepare(args.cache, args.output, args.fetch)
    print(json.dumps({k: result[k] for k in ("counts", "prospective_partitions", "lengths", "answer_cap_counts_all_source_rows", "elapsed_seconds", "peak_process_rss_gib", "preference_fit_admitted")}, ensure_ascii=False))
