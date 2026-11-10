#!/usr/bin/env python3
"""Measure whether the existing glyph pixels can represent native teacher spans."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ilm.image_edit.baseline import ROOT, save_json, sha


PARENT = ROOT / "artifacts/balanced_glyph_writer_v1/preparation.json"
TRAIN = ROOT / "data/visual_qa/cmrc2018_short/train.jsonl"


def summarize(texts, tokenizer, spellings, chunks, ambiguous, glyph_ids):
    reasons = Counter()
    lengths = Counter()
    result = Counter()
    seen_units = set()
    for text in texts:
        units = tokenizer.encode(text, add_special_tokens=False).ids
        result["examples"] += 1
        result["native_steps"] += len(units)
        result["character_steps"] += len(text)
        result["whole_source_roundtrips"] += tokenizer.decode(units, skip_special_tokens=False) == text
        individual_roundtrip = "".join(spellings[i] for i in units) == text
        result["individual_unit_roundtrips"] += individual_roundtrip
        result["all_characters_have_glyphs"] += all(char in glyph_ids for char in text)
        eligible = True
        bounded = True
        for index in units:
            seen_units.add(index)
            spelling = spellings[index]
            if not spelling or "\ufffd" in spelling:
                reasons["empty_or_replacement_character"] += 1
                eligible = False
            elif chunks[index] is None:
                reasons["missing_canonical_glyph"] += 1
                eligible = False
            elif index in ambiguous:
                reasons["same_pixels_as_another_native_unit"] += 1
                eligible = False
            else:
                length = len(chunks[index])
                lengths[length] += 1
                bounded = bounded and length <= 8
        if eligible and individual_roundtrip:
            result["lossless_unambiguous_examples"] += 1
            result["lossless_unambiguous_examples_with_maximum_8_glyphs"] += bounded
            result["eligible_native_steps"] += len(units)
            result["eligible_character_steps"] += len(text)
    return {
        **dict(result),
        "distinct_native_units": len(seen_units),
        "eligible_step_glyph_lengths": dict(sorted(lengths.items())),
        "ineligible_steps_by_reason": dict(reasons),
    }


def anchor_composition(answers, tokenizer, chunks, ambiguous, archive, primitive_path):
    import torch
    from torch.nn import functional as F

    from scripts.check_native_visual_transfer_v71 import embed_rows

    unit_counts = Counter(index for answer in answers
                          for index in tokenizer.encode(answer, add_special_tokens=False).ids)
    units = sorted(index for index in unit_counts
                   if chunks[index] is not None and len(chunks[index]) > 1 and index not in ambiguous)
    anchors = torch.load(primitive_path, weights_only=True, map_location="cpu")["semantic"]
    cosine_first = []
    cosine_mean = []
    relative_mean_errors = []
    for start in range(0, len(units), 256):
        selected = units[start:start + 256]
        targets = embed_rows(archive, selected)
        first = torch.stack([anchors[chunks[index][0]] for index in selected])
        mean = torch.stack([anchors[list(chunks[index])].mean(0) for index in selected])
        cosine_first.append(F.cosine_similarity(first, targets))
        cosine_mean.append(F.cosine_similarity(mean, targets))
        relative_mean_errors.append((mean - targets).norm(dim=1) / targets.norm(dim=1).clamp_min(1e-12))
    first = torch.cat(cosine_first)
    mean = torch.cat(cosine_mean)
    errors = torch.cat(relative_mean_errors)
    counts = torch.tensor([unit_counts[index] for index in units], dtype=torch.float32)
    return {
        "distinct_observed_multi_glyph_units": len(units),
        "total_occurrences": int(counts.sum()),
        "mean_glyph_anchor_cosine_median": float(mean.median()),
        "first_glyph_anchor_cosine_median": float(first.median()),
        "mean_glyph_anchor_cosine_occurrence_weighted": float((mean * counts).sum() / counts.sum()),
        "mean_glyph_anchor_relative_l2_error_median": float(errors.median()),
        "mean_glyph_anchor_cosine_at_least_point9": int((mean >= .9).sum()),
        "train_only_teacher_embedding_geometry": True,
        "actual_model_trajectory_tested": False,
    }


def audit(output):
    import torch

    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    from ilm.visual_lm.grounded_image_weights import MODEL_SHA

    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    parent = json.loads(PARENT.read_text())
    inventory_path = ROOT / parent["primitive"] / "inventory.json"
    primitive = json.loads((inventory_path.parent / "preparation.json").read_text())
    primitive_path = inventory_path.parent / "training.pt"
    if sha(inventory_path) != primitive["inventory_sha256"]:
        raise ValueError("Glyph inventory differs from the verified primitive")
    if sha(primitive_path) != primitive["training_sha256"]:
        raise ValueError("Glyph semantic targets differ from the verified primitive")
    inventory = json.loads(inventory_path.read_text())["records"]
    glyph_ids = {char: i for i, row in enumerate(inventory) for char in row["chars"]}
    training = {row["id"]: row for row in map(json.loads, TRAIN.read_text().splitlines())}
    answers = []
    paragraphs = {}
    for job in parent["training_jobs"]:
        row = training[job["id"]]
        answer = row["references"][0]["text"]
        if [glyph_ids[char] for char in answer] != job["labels"]:
            raise ValueError("Canonical training answer differs from the prepared target")
        answers.append(answer)
        original = row["original_context"]
        paragraphs[original] = None

    archive, tokenizer, markers = archive_and_tokenizer()
    spellings = [tokenizer.decode([i], skip_special_tokens=False)
                 for i in range(len(archive.field("tokenizer.ggml.tokens")))]
    special = set(map(int, archive.field("tokenizer.ggml.eos_token_ids"))) | set(markers.values())
    special.update(index for index, token in tokenizer.get_added_tokens_decoder().items() if token.special)
    chunks = []
    owners = defaultdict(list)
    for index, spelling in enumerate(spellings):
        chunk = None
        if index not in special and spelling and "\ufffd" not in spelling and all(char in glyph_ids for char in spelling):
            chunk = tuple(glyph_ids[char] for char in spelling)
            owners[chunk].append(index)
        chunks.append(chunk)
    ambiguous = {index for indices in owners.values() if len(indices) > 1 for index in indices}
    supported = [len(chunk) for chunk in chunks if chunk is not None]
    report = {
        "source_archive_sha256": MODEL_SHA,
        "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (PARENT, TRAIN, inventory_path, primitive_path)},
        "source_code_sha256": sha(Path(__file__)),
        "canonical_glyph_patterns": len(inventory),
        "native_vocabulary_units": len(spellings),
        "units_with_all_glyphs": len(supported),
        "multi_glyph_units_with_all_glyphs": sum(n > 1 for n in supported),
        "exact_pixel_alias_groups": sum(len(indices) > 1 for indices in owners.values()),
        "units_in_exact_pixel_alias_groups": len(ambiguous),
        "maximum_glyphs_per_supported_unit": max(supported),
        "source_only_train_answers": summarize(answers, tokenizer, spellings, chunks, ambiguous, glyph_ids),
        "source_only_unique_original_train_paragraphs": summarize(
            paragraphs, tokenizer, spellings, chunks, ambiguous, glyph_ids
        ),
        "observed_multi_glyph_unit_anchor_composition": anchor_composition(
            answers, tokenizer, chunks, ambiguous, archive, primitive_path
        ),
        "canonical_raster_ids_are_deduplicated_binary_images": True,
        "uses_training_records_only": True,
        "analysis_uses_source_tokenizer": True,
        "neural_model_forward_or_fit_performed": False,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    output.mkdir(parents=True)
    save_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/visual_chunk_coverage_v1")
    audit(parser.parse_args().output)
