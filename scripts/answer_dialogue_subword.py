"""Generate a reply with the released from-scratch CoQA token baseline."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from ilm.visual_lm.dialogue_subword import DialogueSubword


def load(folder, device):
    model = folder / "transformer.safetensors"
    tokenizer = folder / "tokenizer.json"
    meta = json.loads((folder / "weights.json").read_text())
    for path, field in [(model, "model_sha256"), (tokenizer, "tokenizer_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[field]:
            raise ValueError(f"Checkpoint mismatch for {path.name}")
    tok = Tokenizer.from_file(str(tokenizer))
    net = DialogueSubword(tok.get_vocab_size()).to(device)
    net.load_state_dict(load_file(str(model)))
    return net.eval(), tok


def answer(net, tok, passage, question, history=""):
    p, q, h = [s.ids for s in tok.encode_batch([passage, question, history])]
    seq = [1] + q + [3] + h + [3] + p
    seg = [0] * (len(q) + 2) + [1] * (len(h) + 1) + [2] * len(p)
    device = net.frequency.device
    output, stopped = net.generate(
        torch.tensor([seq], device=device), torch.tensor([seg], device=device)
    )
    words = output[0].tolist()
    if stopped:
        words = words[: words.index(2)]
    return {"answer": tok.decode(words), "stopped": bool(stopped)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("passage", "question", "history"):
        p.add_argument("--" + name, required=name != "history", default="")
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/dialogue_subword",
    )
    a = p.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(answer(*load(a.model, "cpu"), a.passage, a.question, a.history)))
