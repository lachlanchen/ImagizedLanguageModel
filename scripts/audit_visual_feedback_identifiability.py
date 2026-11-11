#!/usr/bin/env python3
"""Measure feedback-loss limits from identical span pixels in the actual schedule."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
from scripts.cache_visual_sequence_teacher_states import batches, read_sequences
from scripts.train_compact_visual_span import source_targets


def arrays(directory, shard, suffix):
    name, = [name for name in shard["files"] if name.endswith(suffix)]
    if Path(name).name != name or sha(directory / name) != shard["files"][name]:
        raise ValueError("Training shard identity changed")
    with np.load(directory / name, allow_pickle=False) as archive:
        return dict(archive)


def audit(data, preparation, output, budget):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA during the feedback identifiability measurement")
    torch.set_num_threads(2)
    plan = json.loads(preparation.read_text())
    if sha(data / "report.json") != plan["data_report_sha256"]:
        raise ValueError("Use the prepared complete training corpus")
    if budget < plan["maximum_native_sequence"]:
        raise ValueError("Batch budget must fit the longest complete source sequence")
    appearances = Counter()
    vector_mass = defaultdict(float)
    count_mass = defaultdict(float)
    images = {}
    total_batches = total_spans = total_units = 0
    for shard in plan["shards"]:
        pixels = arrays(data, shard, "_pixels.npz")
        alignment = arrays(data, shard, "_teacher.npz")
        records = read_sequences(data, shard)
        span_offsets = pixels["record_span_offsets"]
        groupings = list(batches(records, budget))
        for group in groupings:
            sample_counts = Counter()
            spans = units = 0
            for record in group:
                first, end = span_offsets[record:record + 2]
                glyph_start = int(pixels["record_glyph_offsets"][record])
                unit_start = int(alignment["record_unit_offsets"][record])
                for index in range(first, end):
                    glyph_end = int(pixels["span_glyph_ends"][index])
                    unit_end = int(alignment["span_unit_ends"][index])
                    source = tuple(map(int, alignment["unit_ids"][unit_start:unit_end]))
                    image = pixels["ink"][glyph_start:glyph_end].tobytes()
                    if not source or not image:
                        raise ValueError("Empty source update or visual span")
                    if images.setdefault(source, image) != image:
                        raise ValueError("One source sequence rendered to different pixel spans")
                    sample_counts[source] += 1
                    spans += 1
                    units += len(source)
                    glyph_start, unit_start = glyph_end, unit_end
            total_batches += 1
            total_spans += spans
            total_units += units
            for source, count in sample_counts.items():
                appearances[source] += count
                vector_mass[source] += count / units
                count_mass[source] += count / spans
        print(json.dumps({"completed_records": shard["first_record"] + len(records),
                          "batches": total_batches, "unique_native_sequences": len(images)}), flush=True)
    if total_units != sum(shard["native_units"] for shard in plan["shards"]):
        raise ValueError("Audited feedback positions differ from the scheduled training pass")
    if (not np.isclose(sum(mass * len(key) for key, mass in vector_mass.items()) / total_batches, 1)
            or not np.isclose(sum(count_mass.values()) / total_batches, 1)):
        raise ValueError("Scheduled objective weights do not sum to one")
    by_image = defaultdict(list)
    for source, image in images.items():
        by_image[image].append(source)
    archive = open_compact_archive(ROOT.parent / "LocalLLM")
    lookup, vectors = source_targets(archive, plan["training_native_units"])
    tokens = archive.reader.fields["tokenizer.ggml.tokens"].contents()
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    byte_map = {value: key for key, value in bytes_to_unicode().items()}

    def spelling(source):
        raw = bytes(byte_map[char] for unit in source for char in tokens[unit])
        return raw.decode("utf-8")

    lower_bound = count_entropy = count_error = residual_disagreement = 0.0
    conflicts = []
    for image, sources in by_image.items():
        if len(sources) == 1:
            continue
        source_vectors = {key: vectors[lookup[list(key)]].double() for key in sources}
        minimum = 0.0
        for position in range(max(map(len, sources))):
            targets = [(key, value[position]) for key, value in source_vectors.items() if position < len(value)]
            weighted_sum = torch.zeros(vectors.shape[1], dtype=torch.float64)
            weight_sum = torch.zeros((), dtype=torch.float64)
            energy = torch.zeros_like(weight_sum)
            terms = []
            for key, target in targets:
                mass = vector_mass[key] / total_batches
                norm = target.square().sum().clamp_min(1e-8)
                weight = mass / norm
                weighted_sum += weight * target
                weight_sum += weight
                energy += weight * target.square().sum()
                terms.append((weight, target))
            optimum = weighted_sum / weight_sum
            direct = sum(weight * (target - optimum).square().sum() for weight, target in terms)
            closed = (energy - weighted_sum.square().sum() / weight_sum).clamp_min(0)
            difference = float((direct - closed).abs())
            if difference > 1e-12:
                raise ValueError("Closed-form minimum differs from the actual weighted residual")
            residual_disagreement = max(residual_disagreement, difference)
            minimum += float(direct)
        counts = defaultdict(float)
        for key in sources:
            counts[len(key)] += count_mass[key] / total_batches
        distribution = np.asarray(list(counts.values()), dtype=np.float64)
        mass = distribution.sum()
        probabilities = distribution / mass
        entropy = float(-(distribution * np.log(probabilities)).sum())
        error = float(mass - distribution.max())
        lower_bound += minimum
        count_entropy += entropy
        count_error += error
        conflicts.append({"span_image_sha256": hashlib.sha256(image).hexdigest(), "glyphs": len(image) // 128,
                          "source_spellings": [{"text": spelling(key), "native_updates": len(key),
                                                "appearances": appearances[key]} for key in sources],
                          "appearances": sum(appearances[key] for key in sources),
                          "all_pixels_blank": not any(image),
                          "contribution_to_feedback_loss_floor": minimum,
                          "contribution_to_count_entropy_nats": entropy,
                          "contribution_to_count_error": error})
    conflicts.sort(key=lambda row: row["contribution_to_feedback_loss_floor"], reverse=True)
    receipt = {"corpus_report_sha256": sha(data / "report.json"), "teacher_preparation_sha256": sha(preparation),
               "padded_unit_budget": budget, "actual_scheduled_batches": total_batches, "span_appearances": total_spans,
               "native_feedback_positions": total_units, "distinct_native_sequences": len(images),
               "distinct_visual_spans": len(by_image), "conflicting_visual_spans": len(conflicts),
               "conflicting_span_appearances": sum(row["appearances"] for row in conflicts),
               "scheduled_feedback_relative_squared_error_floor": lower_bound,
               "scheduled_feedback_count_cross_entropy_floor_nats": count_entropy,
               "scheduled_feedback_count_error_floor": count_error,
               "maximum_direct_vs_closed_form_residual_difference": residual_disagreement,
               "conflicts": conflicts, "source_sha256": sha(Path(__file__)),
               "vector_targets": "BF16 source rows used by the actual compact trainer",
               "fit_or_inference_performed": False, "cuda_initialized": torch.cuda.is_initialized()}
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "report.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "conflicts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/visual_sequence_corpus_v1")
    parser.add_argument("--preparation", type=Path, default=ROOT / "artifacts/visual_sequence_teacher_v1/preparation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/visual_feedback_identifiability_v1")
    parser.add_argument("--padded-unit-budget", type=int, default=1024)
    args = parser.parse_args()
    if args.padded_unit_budget < 1:
        parser.error("Use a positive batch budget")
    audit(args.data, args.preparation, args.output, args.padded_unit_budget)
