"""Qualify and encode with the frozen, training-only English sentence teacher."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from scripts.prepare_questioned_events import PUB, ROOT, read, dump, sha

DATA = ROOT / "data/raw/visual_semantic_teacher"
WEIGHTS = DATA / "minilm"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EVIDENCE = ROOT / "references/evidence/optical_passage_qa_v1"


class Teacher:
    def __init__(self, device):
        if (
            sha(WEIGHTS / "model.safetensors")
            != "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
        ):
            raise ValueError("Unexpected MiniLM checkpoint")
        self.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, local_files_only=True)
        self.model = (
            AutoModel.from_pretrained(WEIGHTS, local_files_only=True).to(device).eval()
        )
        self.device = device

    @torch.no_grad()
    def encode(self, sentences, batch=128):
        vectors = []
        for i in range(0, len(sentences), batch):
            tokens = self.tokenizer(
                sentences[i : i + batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            h = self.model(**tokens).last_hidden_state
            m = tokens.attention_mask.unsqueeze(-1)
            vectors.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu())
        return torch.cat(vectors)


def cases():
    with gzip.open(PUB / "description_evaluation/decisions.json.gz", "rt") as f:
        decisions = json.load(f)
    rows = read(PUB / "held.json")
    pages = defaultdict(list)
    for ix, mode, target, qi, arm, *_ in decisions["records"]:
        if arm == "word_aligned" and mode in ("natural_unseen", "webster", "natural"):
            names = [e[2].replace("-", " ") for e in rows[ix]["events"]]
            group = "pu" if rows[ix]["group"] == "literature" else "kyoto"
            pages[group + "_" + mode, ix].append(
                (decisions["questions"][qi], names, target)
            )
    return pages


def score(pages, vectors):
    groups = defaultdict(list)
    for (group, page), items in pages.items():
        margins = []
        for question, names, target in items:
            s = torch.stack([vectors[t] for t in names]) @ vectors[question]
            best_other = max(x.item() for j, x in enumerate(s) if j != target)
            margins.append(float(s[target]) - best_other)
        groups[group].append(
            dict(
                page=page,
                min_margin=min(margins),
                correct=sum(x > 0 for x in margins),
                questions=len(items),
            )
        )
    return {
        g: dict(
            pages=len(rr),
            complete=sum(r["min_margin"] > 0 for r in rr),
            queries=sum(r["questions"] for r in rr),
            correct=sum(r["correct"] for r in rr),
            decisions=rr,
        )
        for g, rr in groups.items()
    }


def strings(pages):
    return sorted(
        {s for rows in pages.values() for q, names, _ in rows for s in (q, *names)}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    pages = cases()
    sentences = strings(pages)
    result = score(pages, dict(zip(sentences, Teacher("cpu").encode(sentences))))
    passed = (
        result["kyoto_natural_unseen"]["complete"] > 4
        and result["kyoto_webster"]["complete"] > 48
    )
    dump(args.output, dict(revision=REVISION, qualified=passed, groups=result))
    print(
        {k: (v["complete"], v["pages"]) for k, v in result.items()}, "qualified", passed
    )
