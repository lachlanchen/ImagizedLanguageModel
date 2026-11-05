"""Physical-image data and scoring for the registered V68 autonomous audit.

These utilities belong to offline evaluation. Generation receives an owning
prompt tensor only; reference images are accessed separately after generation.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .causal_visual_prefix import fold_writing_prompt

AUDIT_DATA_SHA = "4f90406b51670e4e82327d1396e1c954e62bfb5d7b31c437b5c59e88a9180fe5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CodecLanguageAuditData:
    def __init__(self, path):
        self.path = Path(path)
        if sha(self.path / "report.json") != AUDIT_DATA_SHA:
            raise ValueError("fixed autonomous audit data changed")
        self.report = json.loads((self.path / "report.json").read_text())
        self.cases = self.report["cases"]
        if (len(self.cases) != 192 or self.report["model_calls"] != 0
                or not self.report["evaluation_only"] or self.report["sealed_examples"] != 0):
            raise ValueError("complete, evaluation-only audit data required")
        if len({(r["source"], r["id"]) for r in self.cases}) != 192:
            raise ValueError("fixed questions must be unique")
        for row in self.cases:
            relative = Path(row["file"])
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pt":
                raise ValueError("unsafe audit image path")
        chinese = [r for r in self.cases if r["source"] == "cmrc"]
        english = [r for r in self.cases if r["source"] == "dolly"]
        if len(chinese) != 128 or len(english) != 64 or len(self.report["chosen_pairs"]) != 64:
            raise ValueError("registered source counts differ")
        articles = {r["article"] for r in chinese}
        if len(articles) != 64 or any(sum(r["article"] == a for r in chinese) != 2 for a in articles):
            raise ValueError("64 distinct two-question articles required")
        categories = {r["category"] for r in english}
        if len(categories) != 8 or any(sum(r["category"] == c for r in english) != 8 for c in categories):
            raise ValueError("eight English cases per category required")

    def payload(self, index):
        row = self.cases[index]
        path = self.path / row["file"]
        if sha(path) != row["sha256"]:
            raise ValueError("fixed audit image changed")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if set(value) != {"prompt", "blank_prompt", "references"} or not value["references"]:
            raise ValueError("physical prompt/control/reference images required")
        for image in (value["prompt"], value["blank_prompt"], *value["references"]):
            if (image.dtype != torch.uint8 or image.ndim != 2 or image.shape[0] != 32
                    or image.shape[1] % 64 or not bool(((image == 0) | (image == 255)).all())):
                raise ValueError("complete binary physical strips required")
        if (list(value["prompt"].shape) != row["prompt_shape"]
                or value["blank_prompt"].shape != value["prompt"].shape
                or [list(x.shape) for x in value["references"]] != row["reference_shapes"]):
            raise ValueError("registered image shapes differ")
        if not bool((value["blank_prompt"] >= value["prompt"]).all()):
            raise ValueError("blanking cannot add foreground ink")
        return value

    def prompt(self, index, *, blank=False):
        value = self.payload(index)
        pixels = value["prompt"].to(torch.float32).div(127.5).sub(1)[None, None].clone()
        normal = fold_writing_prompt(pixels)
        if not blank:
            return normal.pixels.clone()
        # Erasure can create new wrap candidates. Keep the observed normal
        # layout so the intervention changes ink alone, never coordinates.
        erased = value["blank_prompt"].to(torch.float32).div(127.5).sub(1)[None, None].clone()
        page = torch.ones_like(normal.pixels)
        for row, (start, end) in enumerate(normal.source_ranges):
            page[..., row * 32:(row + 1) * 32, :end - start] = erased[..., start:end]
        return page

    def references(self, index):
        return tuple(x.clone() for x in self.payload(index)["references"])


def output_strip(result, *, maximum_blocks=128):
    """Validate actual structural events and retain every physical output bit."""
    images, events = result["images"], result["events"]
    if (not isinstance(images, (tuple, list)) or len(images) > maximum_blocks
            or type(result["stopped"]) is not bool or type(result["capped"]) is not bool
            or result["stopped"] == result["capped"]):
        raise ValueError("invalid generated image stream or termination")
    if len(events) != len(images) + int(result["stopped"]):
        raise ValueError("generation event count differs from actual output")
    if result["capped"] and len(images) != maximum_blocks:
        raise ValueError("capped generation must consume the fixed capacity")
    for index, event in enumerate(events):
        if (event["block"] != index or not np.isfinite(event["stop_probability"])
                or not 0 <= event["stop_probability"] <= 1
                or event["stopped"] != (result["stopped"] and index == len(images))):
            raise ValueError("invalid learned-stop event")
        if event["stopped"] != (event["stop_probability"] >= .5):
            raise ValueError("registered learned-stop threshold changed")
    for image in images:
        if (image.device.type != "cpu" or image.dtype != torch.float32
                or image.shape != (1, 1, 32, 64)
                or not bool(((image == -1) | (image == 1)).all())):
            raise ValueError("unmodified generated physical blocks required")
    if not images:
        return torch.empty((32, 0), dtype=torch.uint8)
    return ((torch.cat(tuple(images), -1)[0, 0] + 1) * 127.5).to(torch.uint8)


def reference_scores(strip, references, *, stopped):
    """Agreement with full physical references, including width and stopping."""
    prediction = strip.numpy() < 128
    candidates = []
    for reference in references:
        target = reference.numpy() < 128
        overlap = min(prediction.shape[1], target.shape[1])
        numerator = 2 * int((prediction[:, :overlap] & target[:, :overlap]).sum())
        denominator = int(prediction.sum() + target.sum())
        candidates.append(dict(foreground_f1=numerator / denominator if denominator else 1.,
            exact=bool(stopped and prediction.shape == target.shape and np.array_equal(prediction, target))))
    return dict(foreground_f1=max(x["foreground_f1"] for x in candidates),
        exact=any(x["exact"] for x in candidates), blank=not bool(prediction.any()),
        blocks=strip.shape[1] // 64, stopped=bool(stopped), references=candidates)


def summarize(data, scores):
    """Apply the prospectively registered development gates without selection."""
    if len(scores) != 192 or set(scores) != {r["source"] + "/" + r["id"] for r in data.cases}:
        raise ValueError("all fixed questions required")
    means = {}
    for source in ("cmrc", "dolly"):
        rows = [scores[r["source"] + "/" + r["id"]] for r in data.cases if r["source"] == source]
        means[source] = {route:{key:float(np.mean([x[route][key] for x in rows]))
            for key in ("foreground_f1", "exact", "blank", "blocks", "stopped")}
            for route in ("normal", "blank")}
    by_id = {r["id"]:r for r in data.cases if r["source"] == "cmrc"}
    pair_scores = [dict(ids=pair, both_correct=all(scores["cmrc/" + identity]["normal"]["exact"]
        for identity in pair)) for pair in data.report["chosen_pairs"]]
    articles = sorted({r["article"] for r in by_id.values()})
    cluster = []
    for article in articles:
        values = [scores["cmrc/" + r["id"]]["normal"]["foreground_f1"]
            - scores["cmrc/" + r["id"]]["blank"]["foreground_f1"]
            for r in by_id.values() if r["article"] == article]
        cluster.append(np.mean(values))
    cluster = np.asarray(cluster)
    draws = np.random.default_rng(20266831).integers(0, 64, (2000, 64))
    bootstrap = cluster[draws].mean(1)
    effect = dict(mean=float(cluster.mean()), ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
        articles=64, replicates=2000, seed=20266831)
    paired = float(np.mean([x["both_correct"] for x in pair_scores]))
    normal = means["cmrc"]["normal"]
    gates = dict(chinese_exact=normal["exact"] >= .5,
        chinese_foreground_f1=normal["foreground_f1"] >= .8,
        both_correct_pairs=paired >= .35, question_gain=effect["mean"] >= .1,
        question_gain_ci_positive=effect["ci95"][0] > 0, learned_stopping=normal["stopped"] >= .9)
    english_categories = {}
    for category in sorted({r["category"] for r in data.cases if r["source"] == "dolly"}):
        rows = [scores["dolly/" + r["id"]]["normal"] for r in data.cases
            if r["source"] == "dolly" and r["category"] == category]
        english_categories[category] = {key:float(np.mean([x[key] for x in rows]))
            for key in ("foreground_f1", "exact", "blank", "stopped")}
    return dict(measurements=means, paired_both_correct=paired, pairs=pair_scores,
        question_effect=effect, gates=gates, bounded_chinese_development_pass=all(gates.values()),
        english_categories=english_categories, english_scores_are_reference_agreement_only=True,
        development_reused=True, sealed_examples=0, general_language_qualified=False,
        general_ilm_goal_complete=False)
