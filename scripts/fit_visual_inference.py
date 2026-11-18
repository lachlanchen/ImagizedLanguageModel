"""Run a fresh paired/control visual writing comparison into a private output directory."""
import argparse
import fcntl
import os
from pathlib import Path

import numpy as np
import torch

from scripts import train_visual_inference_writer as experiment
from scripts.complete_visual_inference_control import repaired, AMEND


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="New directory for checkpoints, images and aggregate results")
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if experiment.binding()!=experiment.load(experiment.REG):
        raise ValueError("Preparation or inherited weights differ from the registered experiment")
    if experiment.load(AMEND)["program_sha256"]!=experiment.sha(Path(__file__).with_name("complete_visual_inference_control.py")):
        raise ValueError("Changed control construction")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or not torch.cuda.is_available():
        raise ValueError("Select the project training GPU with CUDA_VISIBLE_DEVICES=0")
    with (experiment.ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True
        output.mkdir(parents=True)
        # The archived experiment writes relative to these two locations.
        # Configure them before importing its evaluator, which shares the same layout.
        experiment.WORK=output
        experiment.PUBLIC=output/"results"
        experiment.PUBLIC.mkdir()
        experiment.mispair=repaired
        bank=experiment.memory_bank()
        train=experiment.Corpus(bank,"train")
        replay=experiment.replay_data(bank)
        with np.load(experiment.PRIVATE) as split:
            rows=split["fit"]
        tables=experiment.physical_tables(bank,experiment.DEVICE)
        for arm in experiment.ARMS:
            experiment.fit(arm,train,rows,replay,tables)
        from scripts.evaluate_visual_inference_writer import evaluate
        evaluate()
    print(output/"results/result.json")


if __name__=="__main__":
    main()
