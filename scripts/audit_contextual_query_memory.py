#!/usr/bin/env python3
"""CPU-only task-memory intervention on the completed physical span head."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixels(path):
    with Image.open(path) as image:
        a = np.array(image.convert("L"), copy=True)
    if a.ndim != 2 or a.shape[0] != 32 or not np.isin(a, [0, 255]).all():
        raise ValueError("original binary writing image required")
    return a


def trimmed(a):
    x = np.flatnonzero((a == 0).any(0))
    return a[:, x[0]:x[-1] + 1] if len(x) else a[:, :0]


def exact(a, targets):
    return any(np.array_equal(trimmed(a), trimmed(t)) for t in targets)


def mass(logp, positions):
    values = np.array([logp[i, j] for i, j in positions], dtype=np.float64)
    m = float(values.max())
    return m + math.log(float(np.exp(values - m).sum()))


def summary(cases):
    groups = {}
    for split in ("train", "development"):
        rows = [r for r in cases if r["split"] == split]
        pairs = defaultdict(list)
        for row in rows:
            pairs[row["pair"]].append(row)
        if any(len(v) != 2 for v in pairs.values()):
            raise ValueError("complete assessed question pairs required")
        groups[split] = dict(cases=len(rows), pairs=len(pairs),
            own_exact=sum(r["own_exact"] for r in rows),
            swapped_exact=sum(r["swapped_exact"] for r in rows),
            own_partner_exact=sum(r["own_partner_exact"] for r in rows),
            swapped_partner_exact=sum(r["swapped_partner_exact"] for r in rows),
            changed_selected_images=sum(r["selected_image_changed"] for r in rows),
            mean_span_total_variation=float(np.mean([r["span_total_variation"] for r in rows])),
            mean_own_log_odds=float(np.mean([r["own_reference_log_odds"] for r in rows])),
            mean_swapped_log_odds=float(np.mean([r["swapped_reference_log_odds"] for r in rows])),
            mean_own_log_odds_change=float(np.mean([r["swapped_reference_log_odds"] - r["own_reference_log_odds"] for r in rows])),
            own_both_correct=sum(all(r["own_exact"] for r in pair) for pair in pairs.values()),
            swapped_both_partner_correct=sum(all(r["swapped_partner_exact"] for r in pair) for pair in pairs.values()),
            overlapping_reference_sets=sum(r["reference_sets_overlap"] for r in rows))
    return groups


def run(args):
    began = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("set CUDA_VISIBLE_DEVICES='' for this CPU-only diagnostic")
    import torch
    from ilm.visual_lm.contextual_pixel_spans import ContextualPixelSpans, matches, units

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026110708)
    root = args.run
    if read(root / "status.json").get("state") != "completed":
        raise ValueError("completed predecessor required; do not inspect live checkpoints")
    prep = root / "prepared"
    report = read(prep / "report.json")
    for name in ("inputs", "references"):
        if sha(prep / f"{name}.json") != report[f"{name}_sha256"]:
            raise ValueError("prepared inventories changed")
    for name, digest in report["identity"]["sources"].items():
        if sha(name) != digest:
            raise ValueError("executed source changed: " + name)
    feature_report = read(root / "features/report.json")
    if feature_report["preparation_sha256"] != sha(prep / "report.json"):
        raise ValueError("feature/preparation binding changed")
    training = read(root / "training/report.json")
    checkpoint = root / "training/checkpoint.pt"
    if training["updates"] != 3200 or sha(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError("final completed head required")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["completed_updates"] != 3200
            or payload["preparation_sha256"] != sha(prep / "report.json")
            or payload["feature_report_sha256"] != sha(root / "features/report.json")):
        raise ValueError("checkpoint identity differs")
    head = ContextualPixelSpans().eval().requires_grad_(False)
    head.load_state_dict(payload["model"], strict=True)
    del payload
    if any(p.device.type != "cpu" for p in head.parameters()):
        raise ValueError("head must stay on CPU")
    original = read(root / "result.json")
    published = read("references/evidence/contextual_pixel_spans_result/result.json")
    if (sha(root / "result.json") != published["source_result_sha256"]
            or training["checkpoint_sha256"] != published["training"]["checkpoint_sha256"]):
        raise ValueError("completed source result differs from published evidence")
    assessment = read(root / "assessment/report.json")
    if (sha(root / "assessment/outputs.json") != assessment["outputs_sha256"]
            or assessment != original["assessment"] or training != original["training"]
            or feature_report != original["source_features"]
            or sha(prep / "report.json") != original["preparation_sha256"]
            or assessment["training_report_sha256"] != sha(root / "training/report.json")):
        raise ValueError("original assessment changed")
    chosen = [r for r in original["cases"] if r["task"] == "qa" and r["condition"] == "normal"]
    if len(chosen) != 96 or sum(r["split"] == "train" for r in chosen) != 32:
        raise ValueError("fixed 64 development plus32 training cohort required")
    pair_index = defaultdict(list)
    for row in chosen:
        pair_index[(row["split"], row["pair"])].append(row)
    if len(pair_index) != 48 or any(len(v) != 2 for v in pair_index.values()):
        raise ValueError("48 complete same-passage pairs required")
    feature_index = {r["id"]: r for r in feature_report["records"]}
    reference_index = {r["id"]: r for r in read(prep / "references.json")}
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}

    def retain(source, name, expected=None):
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError("source bytes changed: " + str(source))
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(dest) != digest:
            raise ValueError("retained path collision")
        if not dest.exists():
            shutil.copy2(source, dest)
        files[name] = digest
        return name

    def load(row):
        f = feature_index[row["id"]]
        path = root / "features" / f["file"]
        if sha(path) != f["sha256"] or f["sha256"] != row["input_feature_sha256"]:
            raise ValueError("saved feature bytes changed")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value["id"] != row["id"] or value["input_sha256"] != row["input_sha256"]:
            raise ValueError("feature/input ownership differs")
        image = pixels(prep / row["input"])
        if not np.array_equal(value["source_units"].numpy(), units(image)):
            raise ValueError("saved units differ from actual source ink")
        return value, image

    cases = []
    start = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(chosen):
            pair = pair_index[(row["split"], row["pair"])]
            other = next(r for r in pair if r["id"] != row["id"])
            value, image = load(row)
            partner, _ = load(other)
            if value["memory"].shape != partner["memory"].shape:
                raise ValueError("paired task memory shapes differ")
            length = torch.tensor([len(value["visual"])])

            def predict(memory):
                logp = head(value["visual"][None].float(), memory[None].float(),
                    value["geometry"][None].float(), length)[0]
                answer, event = head.select(logp, image, value["source_units"].numpy())
                return logp, answer, event

            # Neither prediction receives a reference image or answer position.
            lp, own, own_event = predict(value["memory"])
            sp, swapped, swapped_event = predict(partner["memory"])
            original_path = root / "assessment" / row["output"]
            if (sha(original_path) != row["output_sha256"]
                    or not np.array_equal(own, pixels(original_path))
                    or (own_event["start"], own_event["length"]) != (row["event"]["start"], row["event"]["length"])):
                raise ValueError("CPU owning replay differs: " + row["id"])
            refs, refs_other, ref_paths = [], [], []
            for key, collection in ((row["id"], refs), (other["id"], refs_other)):
                names = []
                for ref in reference_index[key]["targets"]:
                    name = retain(prep / ref["file"], "references/" + ref["sha256"] + ".png", ref["sha256"])
                    names.append(name)
                    collection.append(pixels(args.output / name))
                ref_paths.append(names)
            # Both reference sets must be physically representable in the owner.
            labels = matches(image, value["source_units"].numpy(), refs)
            other_labels = matches(image, value["source_units"].numpy(), refs_other)
            a, b = lp.numpy(), sp.numpy()
            odds_own = mass(a, labels) - mass(a, other_labels)
            odds_swapped = mass(b, labels) - mass(b, other_labels)
            tv = .5 * float(np.abs(np.exp(a.astype(np.float64)) - np.exp(b.astype(np.float64))).sum())
            q = head.query(value["memory"].float()).mean(0).numpy()
            q_other = head.query(partner["memory"].float()).mean(0).numpy()
            logits_name = f"probabilities/{index:03d}.npz"
            (args.output / "probabilities").mkdir(exist_ok=True)
            np.savez_compressed(args.output / logits_name, own=a, swapped=b,
                own_projected_query=q, partner_projected_query=q_other)
            files[logits_name] = sha(args.output / logits_name)
            own_name = retain(original_path, f"own/{index:03d}.png", row["output_sha256"])
            swapped_name = f"swapped/{index:03d}.png"
            (args.output / "swapped").mkdir(exist_ok=True)
            Image.fromarray(swapped).save(args.output / swapped_name)
            files[swapped_name] = sha(args.output / swapped_name)
            cases.append(dict(id=row["id"], partner_id=other["id"], pair=row["pair"], split=row["split"],
                input=retain(prep / row["input"], f"inputs/{index:03d}.png", row["input_sha256"]),
                input_feature_sha256=row["input_feature_sha256"], partner_feature_sha256=other["input_feature_sha256"],
                own_output=own_name, swapped_output=swapped_name, probabilities=logits_name,
                own_event=own_event, swapped_event=swapped_event, references=ref_paths[0], partner_references=ref_paths[1],
                owning_native_probability_difference=abs(own_event["marginal_probability"] - row["event"]["marginal_probability"]),
                own_exact=exact(own, refs), swapped_exact=exact(swapped, refs),
                own_partner_exact=exact(own, refs_other), swapped_partner_exact=exact(swapped, refs_other),
                selected_image_changed=not np.array_equal(trimmed(own), trimmed(swapped)),
                reference_sets_overlap=any(exact(t, refs_other) for t in refs),
                own_reference_log_odds=odds_own, swapped_reference_log_odds=odds_swapped,
                span_total_variation=tv, projected_query_l2=float(np.linalg.norm(q - q_other)),
                projected_query_relative_l2=float(np.linalg.norm(q - q_other) / max(np.linalg.norm(q), 1e-12))))
            if (index + 1) % 16 == 0:
                print(json.dumps(dict(completed=index + 1, total=96)), flush=True)
    groups = summary(cases)
    for split in ("train", "development"):
        if groups[split]["own_exact"] != original["groups"][f"{split}/qa/normal/zh"]["exact"]:
            raise ValueError("owning CPU quality differs from original score")
    result = dict(groups=groups, cases=cases, files_sha256=files,
        checkpoint_sha256=training["checkpoint_sha256"], source_result_sha256=sha(root / "result.json"),
        head_source_sha256=sha("ilm/visual_lm/contextual_pixel_spans.py"), script_sha256=sha(__file__),
        protocol_sha256=sha("references/contextual_query_memory_protocol.md"),
        cpu_head_parameters=sum(p.numel() for p in head.parameters()), complete_runtime_parameters=training["full_runtime_parameters"],
        head_calls=192, original_answers_replayed=96, source_encoder_calls=0, optimizer_updates=0,
        cpu_threads=torch.get_num_threads(), torch_version=torch.__version__, elapsed_seconds=time.monotonic() - start,
        setup_seconds=start - began, total_seconds=time.monotonic() - began,
        peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        cached_feature_module_intervention=True, native_image_intervention=False,
        model_or_gate_selected=False, language_quality_qualification=False, general_ilm_goal_complete=False)
    (args.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["groups"], ensure_ascii=False, indent=2), flush=True)


def verify(folder):
    """Independent physical-column scan and scoring; imports no Torch."""
    result = read(folder / "result.json")
    for name, digest in result["files_sha256"].items():
        if sha(folder / name) != digest:
            raise ValueError("retained bytes changed: " + name)
    recomputed = []
    for row in result["cases"]:
        image = pixels(folder / row["input"])
        edges = np.diff(np.r_[False, (image == 0).any(0), False].astype(np.int8))
        left, right = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        refs = [pixels(folder / name) for name in row["references"]]
        partner_refs = [pixels(folder / name) for name in row["partner_references"]]
        with np.load(folder / row["probabilities"], allow_pickle=False) as p:
            probs = {k: p[k].astype(np.float64) for k in ("own", "swapped")}
            query = p["own_projected_query"].astype(np.float64)
            partner_query = p["partner_projected_query"].astype(np.float64)
        distance = float(np.linalg.norm(query - partner_query))
        relative_distance = distance / max(float(np.linalg.norm(query)), 1e-12)
        if not (math.isclose(distance, row["projected_query_l2"], rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(relative_distance, row["projected_query_relative_l2"], rel_tol=1e-6, abs_tol=1e-6)):
            raise ValueError("retained query-distance summary differs")
        for p in probs.values():
            if p.shape != (len(left), 128) or abs(float(np.exp(p).sum()) - 1) > 2e-5:
                raise ValueError("invalid complete source-span distribution")
            for start in range(len(left)):
                valid = min(128, len(left) - start)
                if not np.isfinite(p[start, :valid]).all() or not np.isneginf(p[start, valid:]).all():
                    raise ValueError("valid/invalid span support differs")
        found = [[], []]
        content_mass = {name: defaultdict(float) for name in probs}
        first_event = {}
        # Separate hashing implementation; hashes group identical physical images.
        columns = image.T.copy().tobytes()
        for start, lo in enumerate(left):
            for length in range(1, min(128, len(left) - start) + 1):
                candidate = image[:, lo:right[start + length - 1]]
                hi = int(right[start + length - 1])
                key = (hi - int(lo), hashlib.sha512(columns[int(lo) * 32:hi * 32]).digest())
                first_event.setdefault(key, (start, length))
                for name, p in probs.items():
                    content_mass[name][key] += float(np.exp(p[start, length - 1]))
                for positions, targets in zip(found, (refs, partner_refs)):
                    if any(np.array_equal(candidate, trimmed(t)) for t in targets):
                        positions.append((start, length - 1))
        if not all(found):
            raise ValueError("both answers must remain available in the owning image")
        actual = []
        for name in ("own", "swapped"):
            event = row[name + "_event"]
            start, length = event["start"], event["length"]
            if not 0 <= start < start + length <= len(left):
                raise ValueError("invalid selected source span")
            best = max(content_mass[name], key=content_mass[name].get)
            if first_event[best] != (start, length) or not math.isclose(
                    content_mass[name][best], event["marginal_probability"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("selected image is not the independently computed maximum-mass image")
            reconstructed = np.pad(image[:, left[start]:right[start + length - 1]],
                ((0, 0), (2, 2)), constant_values=255)
            actual.append(pixels(folder / row[name + "_output"]))
            if not np.array_equal(actual[-1], reconstructed):
                raise ValueError("output differs from declared physical source columns")
        values = dict(own_exact=any(np.array_equal(trimmed(actual[0]), trimmed(t)) for t in refs),
            swapped_exact=any(np.array_equal(trimmed(actual[1]), trimmed(t)) for t in refs),
            own_partner_exact=any(np.array_equal(trimmed(actual[0]), trimmed(t)) for t in partner_refs),
            swapped_partner_exact=any(np.array_equal(trimmed(actual[1]), trimmed(t)) for t in partner_refs),
            reference_sets_overlap=any(np.array_equal(trimmed(a), trimmed(b)) for a in refs for b in partner_refs),
            selected_image_changed=not np.array_equal(trimmed(actual[0]), trimmed(actual[1])),
            span_total_variation=.5 * float(np.abs(np.exp(probs["own"]) - np.exp(probs["swapped"])).sum()))
        for name in ("own", "swapped"):
            p = probs[name]
            # Accumulate probability in log space independently of production labels.
            totals = []
            for positions in found:
                total = np.float64(-np.inf)
                for i, j in positions:
                    total = np.logaddexp(total, p[i, j])
                totals.append(float(total))
            values[name + "_reference_log_odds"] = totals[0] - totals[1]
        for key, value in values.items():
            if isinstance(value, bool):
                if value != row[key]:
                    raise ValueError("physical image score differs: " + key)
            elif not math.isclose(value, row[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("probability score differs: " + key)
        recomputed.append(dict(row, **values))
    if summary(recomputed) != result["groups"] or result["original_answers_replayed"] != 96:
        raise ValueError("complete paired summary differs")
    receipt = dict(cases=96, physical_outputs_reconstructed=192, groups_recomputed=True,
        source_encoder_calls=0, neural_forward_reimplemented=False,
        result_sha256=sha(folder / "result.json"), verifier_script_sha256=sha(__file__),
        general_ilm_goal_complete=False)
    (folder / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=Path("artifacts/contextual_pixel_spans_v1"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    verify(args.output) if args.verify else run(args)


if __name__ == "__main__":
    main()
