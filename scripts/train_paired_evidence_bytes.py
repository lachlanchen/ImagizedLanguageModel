"""Matched byte-input representation control for the complete-answer routes."""

from scripts.train_paired_evidence import (
    Pairs,
    Corpus,
    load_model,
    gpu_owner,
    byte_features,
    fit,
)
from scripts.verse_gpu_runtime import configure_cuda_cli

if __name__ == "__main__":
    configure_cuda_cli()
    data = Pairs()
    old = Corpus()
    with gpu_owner():
        x = byte_features(data.a["chars"]).cuda()
        base = load_model("byte-pair", "cuda").requires_grad_(False)
        fit(data, old, x, base, base.encode(x), "byte-evidence")
