"""Complete the registered comparison with the unavoidable singleton-genre exception."""
import argparse
import fcntl
import os
from pathlib import Path

import numpy as np
import torch

from scripts import train_visual_inference_writer as trial

AMEND=trial.PUBLIC/"control_amendment.json"
original=trial.mispair


def repaired(ds,rows,rng):
    # The three missing-genre rows all cite the same premise. Identity is the
    # only within-genre reassignment preserving their empirical marginal.
    has_genre=np.array([ds.meta[i]["genre"] is not None for i in rows])
    unknown=rows[~has_genre]
    assert len(unknown)==3 and len({ds.meta[i]["premise"] for i in unknown})==1
    order=original(ds,rows[has_genre],rng)
    assert np.array_equal(order[unknown],unknown)
    assert all(ds.meta[i]["genre"]==ds.meta[k]["genre"] for i,k in zip(rows,order[rows],strict=True))
    return order


def binding():
    folder=trial.WORK/"paired"
    receipt=trial.load(folder/"fit.json")
    assert trial.binding()==trial.load(trial.REG) and receipt["sha256"]==trial.sha(folder/"final.safetensors")
    assert receipt["steps"]==4184
    return dict(program_sha256=trial.sha(Path(__file__)),registration_sha256=trial.sha(trial.REG),
                paired_final=receipt["sha256"],paired_fit_sha256=trial.sha(folder/"fit.json"),
                original_log_sha256=trial.sha(trial.ROOT/"artifacts/visual_inference_writer_32719.log"),
                kept_same_per_epoch=3,distinct_per_epoch=33438,steps_before_control_exception=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    if parser.parse_args().register:
        if AMEND.exists():raise FileExistsError(AMEND)
        trial.write_json(AMEND,binding());return
    if binding()!=trial.load(AMEND) or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Changed comparison")
    with (trial.ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        folder=trial.WORK/"mispaired"
        if not folder.is_dir() or any(folder.iterdir()):
            raise ValueError("Expected the untouched control directory from initialization")
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
        bank=trial.memory_bank()
        ds=trial.Corpus(bank,"train")
        with np.load(trial.PRIVATE) as split:rows=split["fit"]
        for epoch in range(trial.EPOCHS):
            m=repaired(ds,rows,np.random.default_rng(trial.SEED+100+epoch))
            assert int((m[rows]==rows).sum())==3
        replay=trial.replay_data(bank)
        tables=trial.physical_tables(bank,trial.DEVICE)
        folder.rename(trial.WORK/"control_initialization_before_exception")
        trial.mispair=repaired
        trial.fit("mispaired",ds,rows,replay,tables)
        from scripts.evaluate_visual_inference_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()
