#!/usr/bin/env python3
"""Publish verified hybrid-memory PNGs without duplicating the baseline gallery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = read(args.run / "result.json")
    verified = read(args.run / "verification.json")
    if (verified["result_sha256"] != sha(args.run / "result.json")
            or verified["physical_outputs_reconstructed"] != 192
            or verified["cases"] != 96 or not verified["groups_recomputed"]):
        raise ValueError("complete independent physical verification required")
    for name, digest in result["files_sha256"].items():
        if sha(args.run / name) != digest:
            raise ValueError("verified retained file changed: " + name)
    baseline_root = Path("references/evidence/contextual_pixel_spans_result")
    baseline = read(baseline_root / "result.json")
    if baseline["source_result_sha256"] != result["source_result_sha256"]:
        raise ValueError("published predecessor identity differs")
    old = {r["id"]: r for r in baseline["cases"]}
    args.output.mkdir(parents=True, exist_ok=False)
    files, cases, gallery = {}, [], []

    def link(name):
        path = baseline_root / name
        if sha(path) != baseline["files_sha256"][name]:
            raise ValueError("published original bytes changed")
        relative = os.path.relpath(path, args.output)
        files[relative] = sha(path)
        return relative

    for row in result["cases"]:
        previous = old[row["id"]]
        source = args.run / row["swapped_output"]
        destination = args.output / row["swapped_output"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != result["files_sha256"][row["swapped_output"]]:
            raise ValueError("public swapped PNG differs")
        files[row["swapped_output"]] = sha(destination)
        if (sha(baseline_root / previous["input"]) != result["files_sha256"][row["input"]]
                or sha(baseline_root / previous["output"]) != result["files_sha256"][row["own_output"]]):
            raise ValueError("published owning input/output differs")
        public = {k: v for k, v in row.items() if k not in
            ("input", "own_output", "references", "partner_references", "probabilities")}
        public.update(input=link(previous["input"]), own_output=link(previous["output"]),
            references=[link(n) for n in previous["references"]],
            partner_references=[link(n) for n in old[row["partner_id"]]["references"]],
            local_probabilities=str(args.run / row["probabilities"]),
            local_probabilities_sha256=result["files_sha256"][row["probabilities"]])
        cases.append(public)
        targets = ", ".join(f"[ref {i + 1}]({name})" for i, name in enumerate(public["references"]))
        partner_targets = ", ".join(f"[partner {i + 1}]({name})" for i, name in enumerate(public["partner_references"]))
        gallery.append(f"| {row['id']} | {row['split']} | [input]({public['input']}) | "
            f"[own]({public['own_output']}) | [swap]({public['swapped_output']}) | "
            f"{targets}; {partner_targets} | {row['own_exact']} / {row['swapped_exact']} / {row['swapped_partner_exact']} |")
    if len(cases) != 96:
        raise ValueError("complete cohort required")
    preferences = {}
    for split in ("train", "development"):
        rows = [r for r in cases if r["split"] == split]
        preferences[split] = dict(cases=len(rows),
            own_reference_outranks_partner=sum(r["own_reference_log_odds"] > 0 for r in rows),
            partner_reference_outranks_own_after_swap=sum(r["swapped_reference_log_odds"] < 0 for r in rows),
            log_odds_moves_toward_partner=sum(r["swapped_reference_log_odds"] < r["own_reference_log_odds"] for r in rows))
    export = {k: v for k, v in result.items() if k not in ("cases", "files_sha256")}
    export.update(cases=cases, files_sha256=files, derived_preferences=preferences,
        independent_verification=verified, local_run=str(args.run),
        local_result_sha256=sha(args.run / "result.json"), exporter_sha256=sha(__file__),
        copyright="CMRC2018 Cui et al. (2019), CC-BY-SA-4.0; Qwen3-VL Alibaba, Apache-2.0.")
    (args.output / "result.json").write_text(json.dumps(export, ensure_ascii=False, indent=2) + "\n")
    prose = """# All task-memory interventions

These 96 swaps are head-level hybrid features, not native encodings of new
questions. They do not qualify a reader or a free image-language generator.
The existing gallery supplies byte-identical original inputs, answers and
reference alternatives; every swapped PNG is retained unmodified here.

See the [interpretation](@RESULT@),
[prospective protocol](@PROTOCOL@) and
[complete receipt](result.json). Large span-probability arrays remain in the
local run recorded by that receipt. Independent NumPy verification checks
those arrays and all 192 physical outputs; it does not reimplement the neural
forward pass.

The final column is exactness against the owning reference before the swap,
against the owning reference after the swap, and against the partner reference
after the swap. Answers absent from the source cannot be generated.

| ID | Split | Original input | Original answer | Swapped answer | References | Own / swapped-own / swapped-partner |
| --- | --- | --- | --- | --- | --- | --- |
"""
    prose = prose.replace("@RESULT@", os.path.relpath("references/contextual_query_memory_result.md", args.output))
    prose = prose.replace("@PROTOCOL@", os.path.relpath("references/contextual_query_memory_protocol.md", args.output))
    (args.output / "README.md").write_text(prose + "\n".join(gallery) + "\n")
    print(json.dumps(dict(cases=len(cases), new_raw_pngs=96, byte_bound_public_files=len(files),
        derived_preferences=preferences), indent=2))


if __name__ == "__main__":
    main()
