"""Trace two upstream training rows through MiniMind's native loader and full-SFT weights on CPU."""

import json
import random
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import torch
import transformers
from transformers import AutoTokenizer

from scripts.evaluate_visual_story_writer import UPSTREAM, WEIGHT
from scripts.prepare_visual_story_writer import ROOT, PUB, sha, dump

DATA = "312afb4f76391145c6902f765bb51691c09a12f5"


def first_line(task):
    url = f"https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/{DATA}/{task}_t2t_mini.jsonl"
    with requests.get(
        url, headers={"Range": "bytes=0-65535"}, stream=True, timeout=45
    ) as response:
        response.raise_for_status()
        line = next(response.iter_lines())
    json.loads(line)
    return line


def main():
    repo = ROOT.parent / "minimind"
    if (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        != UPSTREAM
    ):
        raise ValueError("Checkout differs")
    weights = ROOT / "artifacts/minimind_reference/full_sft_768.pth"
    if sha(weights) != WEIGHT:
        raise ValueError("Checkpoint differs")
    sys.path.insert(0, str(repo))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from dataset.lm_dataset import PretrainDataset, SFTDataset

    torch.set_num_threads(2)
    torch.manual_seed(42)
    random.seed(42)
    tok = AutoTokenizer.from_pretrained(repo / "model", local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig()).eval()
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    receipt = dict(
        code=UPSTREAM,
        data=DATA,
        checkpoint=WEIGHT,
        torch=torch.__version__,
        transformers=transformers.__version__,
        params=sum(p.numel() for p in model.parameters()),
        examples={},
        tokenizer={
            s: tok(s, add_special_tokens=False).input_ids
            for s in ("言", "𠮷", "中文写成图像。")
        },
    )
    with TemporaryDirectory() as tmp, torch.inference_mode():
        for task, loader, width in (
            ("pretrain", PretrainDataset, 340),
            ("sft", SFTDataset, 768),
        ):
            path = Path(tmp) / (task + ".jsonl")
            path.write_bytes(first_line(task) + b"\n")
            x, y = loader(str(path), tok, max_length=width)[0]
            gold = y[1:] != -100
            out = model(x[None], labels=y[None])
            receipt["examples"][task] = dict(
                raw_sha256=sha(path),
                width=len(x),
                nonpad=int((x != tok.pad_token_id).sum()),
                scored=int(gold.sum()),
                first_scored_label=int(gold.nonzero()[0]) + 1,
                mean_nll=float(out.loss),
            )
    dump(PUB / "minimind_trace.json", receipt)
    print(receipt)


if __name__ == "__main__":
    main()
