"""Prepare frozen-reader predictions before seeding the matched answer writer."""

import torch
from hashlib import sha256
from scripts.prepare_fact_channel import PUB, dump
from scripts.train_fact_channel import (
    Data,
    predictions,
    fit,
    RegionWriter,
    SEED,
    gpu_owner,
)
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    with gpu_owner():
        torch.set_float32_matmul_precision("highest")
        # Loading a trained reader constructs its CPU modules before the weights
        # are restored. Finish this work before fit() resets the writer's seed.
        predictions(Data("writer", "cuda"), "cuda")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            state = RegionWriter().state_dict()
            h = sha256()
            for key, tensor in state.items():
                h.update(key.encode())
                h.update(tensor.contiguous().numpy().tobytes())
        dump(
            PUB / "writer-initialization.json",
            dict(
                seed=SEED,
                initial_tensors=len(state),
                ordered_cpu_weight_sha256=h.hexdigest(),
                prediction_cache_prepared_before_seed=True,
            ),
        )
        fit("predicted")


if __name__ == "__main__":
    configure_cuda_cli()
    main()
