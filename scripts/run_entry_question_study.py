"""Reproduce the two training and generation runs in a new output directory."""

import argparse
import fcntl
from pathlib import Path

import torch

from scripts import evaluate_entry_question_reader as evaluation
from scripts import train_entry_question_reader as training
from scripts.prepare_pictured_entry_answers import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Choose a new directory for this run")
    registration = training.registration()
    if registration != training.load(training.OUT / "registration.json"):
        parser.error(
            "Source images, implementation or initial model differ from the published comparison"
        )
    if not torch.cuda.is_available():
        parser.error("Training both readers requires a CUDA GPU")
    with (training.ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination.mkdir(parents=True)
        training.MODELS = evaluation.MODELS = destination / "models"
        training.OUT = evaluation.OUT = destination / "evidence"
        training.OUT.mkdir()
        write_json(training.OUT / "registration.json", registration)
        torch.set_num_threads(2)
        training.fit()
        evaluation.evaluate()
    print(destination / "evidence/evaluation.json")


if __name__ == "__main__":
    main()
