"""Audit the final writer's physical probability of complete answers and emitted replies."""

from scripts.probe_phrase_likelihood import study
from scripts.prepare_marginal_confirmation import WORK, PUB
from scripts.train_marginal_phrase import MODEL
from scripts.verse_gpu_runtime import configure_cuda_cli

if __name__ == "__main__":
    configure_cuda_cli()
    study(WORK, PUB, MODEL, "new-marginal.json.gz")
