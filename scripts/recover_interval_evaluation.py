"""Evaluate the completed calibration after the recorded import repair, without another fit."""
import fcntl
import torch

from scripts.calibrate_interval_writer import DEST, PUBLIC, ROOT, load, sha
from scripts.evaluate_calibrated_writer import evaluate


def main():
    recovery = load(PUBLIC/"calibration_eval_recovery.json")
    assert sha(ROOT/"scripts/evaluate_calibrated_writer.py") == recovery["corrected_eval"]
    assert sha(DEST/"fit.json") == recovery["fit"]
    assert sha(DEST/"interval.safetensors") == load(DEST/"fit.json")["weights"]
    if (PUBLIC/"calibrated.json").exists():
        raise FileExistsError("Completed calibration evaluation")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        evaluate()


if __name__=="__main__":
    main()
